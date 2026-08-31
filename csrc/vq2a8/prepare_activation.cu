#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/layout/matrix.h>

#include <tuple>

namespace {

constexpr int kRhtBlock = 128;
constexpr int kQuantThreads = 256;
constexpr float kInvSqrt128 = 0.08838834764831843f;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ void vq2a8_fwht_transform_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const int64_t* __restrict__ expert_ids,
    const float* __restrict__ weight_scale,
    const float* __restrict__ weight_bias,
    const int8_t* __restrict__ rht_sign,
    float* __restrict__ transformed,
    float* __restrict__ partial_amax,
    float* __restrict__ partial_bias,
    int size_k,
    int num_rht_blocks) {
  const int assignment = blockIdx.x;
  const int rht_block = blockIdx.y;
  const int lane = threadIdx.x;
  const int column = rht_block * kRhtBlock + lane;
  const int64_t expert = expert_ids[assignment];
  const int expert_offset = static_cast<int>(expert) * size_k + column;
  const int activation_offset = assignment * size_k + column;

  __shared__ float values[kRhtBlock];
  __shared__ float reduce_max[kRhtBlock];
  __shared__ float reduce_sum[kRhtBlock];

  float value = __bfloat162float(activation[activation_offset]);
  value *= static_cast<float>(rht_sign[expert_offset]);
  values[lane] = value;
  __syncthreads();

#pragma unroll
  for (int stride = 1; stride < kRhtBlock; stride <<= 1) {
    const float current = values[lane];
    const float other = values[lane ^ stride];
    const float next = (lane & stride) ? other - current : current + other;
    __syncthreads();
    values[lane] = next;
    __syncthreads();
  }

  const float rotated = values[lane] * kInvSqrt128;
  const float scaled = rotated * weight_scale[expert_offset];
  transformed[activation_offset] = scaled;
  reduce_max[lane] = fabsf(scaled);
  reduce_sum[lane] = rotated * weight_bias[expert_offset];
  __syncthreads();

#pragma unroll
  for (int stride = kRhtBlock / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      reduce_max[lane] = fmaxf(reduce_max[lane], reduce_max[lane + stride]);
      reduce_sum[lane] += reduce_sum[lane + stride];
    }
    __syncthreads();
  }
  if (lane == 0) {
    const int partial_offset = assignment * num_rht_blocks + rht_block;
    partial_amax[partial_offset] = reduce_max[0];
    partial_bias[partial_offset] = reduce_sum[0];
  }
}

__global__ void vq2a8_quantize_kernel(
    const float* __restrict__ transformed,
    const float* __restrict__ partial_amax,
    const float* __restrict__ partial_bias,
    __nv_fp8_storage_t* __restrict__ quantized,
    float* __restrict__ activation_scale,
    float* __restrict__ bias_correction,
    int size_k,
    int num_rht_blocks) {
  const int assignment = blockIdx.x;
  const int thread = threadIdx.x;
  float local_amax = 0.0f;
  float local_bias = 0.0f;
  if (thread < num_rht_blocks) {
    const int partial_offset = assignment * num_rht_blocks + thread;
    local_amax = partial_amax[partial_offset];
    local_bias = partial_bias[partial_offset];
  }
  if (thread < 32) {
    local_amax = warp_max(local_amax);
    local_bias = warp_sum(local_bias);
  }

  __shared__ float scale_shared;
  if (thread == 0) {
    const float scale = fmaxf(local_amax / kFp8Max, 1.0e-12f);
    scale_shared = scale;
    activation_scale[assignment] = scale;
    bias_correction[assignment] = local_bias;
  }
  __syncthreads();

  const int row_offset = assignment * size_k;
  for (int column = thread; column < size_k; column += blockDim.x) {
    float value = transformed[row_offset + column] / scale_shared;
    value = fminf(fmaxf(value, -kFp8Max), kFp8Max);
    quantized[row_offset + column] =
        __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
  }
}

template <bool RoutedReduce>
__global__ void vq2a8_batched_gemm_kernel(
    const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const int64_t* __restrict__ expert_ids,
    const int32_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ codebooks,
    const uint8_t* __restrict__ codebook_tile_ids,
    const float* __restrict__ bias_correction,
    const int64_t* __restrict__ token_ids,
    const float* __restrict__ routing_weights,
    void* __restrict__ output,
    int size_n,
    int size_k,
    int num_codebook_tiles) {
  const int assignment = blockIdx.x;
  const int n_block = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int expert = static_cast<int>(expert_ids[assignment]);
  const int packed_n = size_n / 2;
  const int packed_k = size_k / 8;
  const int output_tiles = size_n / 32;

  using Element = cutlass::float_e4m3_t;
  using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::gemm::GemmShape<16, 8, 32>,
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::RowMajor>::Type;

  __shared__ __align__(16) uint8_t activation_tile[4][16 * 32];
  __shared__ __align__(16) uint8_t weight_tile[4][32 * 8];
  __shared__ __align__(16) float output_tile_shared[4][16 * 8];
  typename WarpMma::FragmentC accumulator;
  accumulator.clear();
  WarpMma warp_mma;

  for (int start_k = 0; start_k < size_k; start_k += 32) {
#pragma unroll
    for (int index = lane; index < 16 * 32; index += 32) {
      activation_tile[warp][index] =
          activation[assignment * size_k + start_k + index % 32];
    }
#pragma unroll
    for (int index = lane; index < 32 * 8; index += 32) {
      const int local_n = index / 32;
      const int local_k = index % 32;
      const int output_column = n_block * 32 + warp * 8 + local_n;
      const int column = start_k + local_k;
      const int vector_row = output_column / 2;
      const int component = output_column & 1;
      const int32_t packed_word =
          packed_indices[(expert * packed_n + vector_row) * packed_k +
                         column / 8];
      const int code = (packed_word >> ((column & 7) * 4)) & 15;
      const int codebook_tile =
          codebook_tile_ids[expert * size_k + column];
      const int codebook_offset =
          (((expert * num_codebook_tiles + codebook_tile) *
                 output_tiles +
             n_block) *
                16 +
            code) *
               2 +
          component;
      weight_tile[warp][index] = codebooks[codebook_offset];
    }
    __syncwarp();

    typename WarpMma::IteratorA iterator_a(
        typename WarpMma::IteratorA::TensorRef(
            reinterpret_cast<Element*>(activation_tile[warp]),
            cutlass::layout::RowMajor(32)),
        lane);
    typename WarpMma::IteratorB iterator_b(
        typename WarpMma::IteratorB::TensorRef(
            reinterpret_cast<Element*>(weight_tile[warp]),
            cutlass::layout::ColumnMajor(32)),
        lane);
    typename WarpMma::FragmentA fragment_a;
    typename WarpMma::FragmentB fragment_b;
    typename WarpMma::TransformedFragmentA transformed_a;
    typename WarpMma::TransformedFragmentB transformed_b;
    iterator_a.load(fragment_a);
    iterator_b.load(fragment_b);
    warp_mma.transform(transformed_a, transformed_b, fragment_a, fragment_b);
    warp_mma(accumulator, transformed_a, transformed_b, accumulator);
    __syncwarp();
  }

  typename WarpMma::IteratorC iterator_c(
      typename WarpMma::IteratorC::TensorRef(
          output_tile_shared[warp], cutlass::layout::RowMajor(8)),
      lane);
  iterator_c.store(accumulator);
  __syncwarp();
  const float scale = activation_scale[assignment];
  const float bias = bias_correction[assignment];
  if (lane < 8) {
    const int column = n_block * 32 + warp * 8 + lane;
    const float value = output_tile_shared[warp][lane] * scale + bias;
    if constexpr (RoutedReduce) {
      const float rounded =
          __bfloat162float(__float2bfloat16_rn(value));
      atomicAdd(reinterpret_cast<float*>(output) +
                    token_ids[assignment] * size_n + column,
                rounded * routing_weights[assignment]);
    } else {
      reinterpret_cast<__nv_bfloat16*>(output)[assignment * size_n + column] =
          __float2bfloat16_rn(value);
    }
  }
}

__global__ void vq2a8_gate_up_kernel(
    const uint8_t* __restrict__ activation,
    const float* __restrict__ activation_scale,
    const int64_t* __restrict__ expert_ids,
    const int32_t* __restrict__ packed_indices,
    const uint8_t* __restrict__ codebooks,
    const uint8_t* __restrict__ codebook_tile_ids,
    const float* __restrict__ bias_correction,
    __nv_bfloat16* __restrict__ output,
    int intermediate_size,
    int size_k,
    int num_codebook_tiles,
    float swiglu_limit) {
  const int assignment = blockIdx.x;
  const int n_block = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int expert = static_cast<int>(expert_ids[assignment]);
  const int packed_n = intermediate_size;
  const int packed_k = size_k / 8;
  const int output_tiles = intermediate_size / 16;

  using Element = cutlass::float_e4m3_t;
  using WarpMma = typename cutlass::gemm::warp::DefaultMmaTensorOp<
      cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::gemm::GemmShape<16, 8, 32>,
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::RowMajor>::Type;

  __shared__ __align__(16) uint8_t activation_tile[4][16 * 32];
  __shared__ __align__(16) uint8_t weight_tile[4][2][32 * 8];
  __shared__ __align__(16) float output_tile_shared[4][2][16 * 8];
  typename WarpMma::FragmentC gate_accumulator;
  typename WarpMma::FragmentC up_accumulator;
  gate_accumulator.clear();
  up_accumulator.clear();
  WarpMma warp_mma;

  for (int start_k = 0; start_k < size_k; start_k += 32) {
#pragma unroll
    for (int index = lane; index < 16 * 32; index += 32) {
      activation_tile[warp][index] =
          activation[assignment * size_k + start_k + index % 32];
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
#pragma unroll
      for (int index = lane; index < 32 * 8; index += 32) {
        const int local_n = index / 32;
        const int local_k = index % 32;
        const int output_column =
            n_block * 32 + warp * 8 + local_n + part * intermediate_size;
        const int column = start_k + local_k;
        const int vector_row = output_column / 2;
        const int component = output_column & 1;
        const int32_t packed_word =
            packed_indices[(expert * packed_n + vector_row) * packed_k +
                           column / 8];
        const int code = (packed_word >> ((column & 7) * 4)) & 15;
        const int codebook_tile =
            codebook_tile_ids[expert * size_k + column];
        const int codebook_offset =
            (((expert * num_codebook_tiles + codebook_tile) * output_tiles +
              output_column / 32) *
                 16 +
             code) *
                2 +
            component;
        weight_tile[warp][part][index] = codebooks[codebook_offset];
      }
    }
    __syncwarp();

    typename WarpMma::IteratorA iterator_a(
        typename WarpMma::IteratorA::TensorRef(
            reinterpret_cast<Element*>(activation_tile[warp]),
            cutlass::layout::RowMajor(32)),
        lane);
    typename WarpMma::IteratorB gate_iterator_b(
        typename WarpMma::IteratorB::TensorRef(
            reinterpret_cast<Element*>(weight_tile[warp][0]),
            cutlass::layout::ColumnMajor(32)),
        lane);
    typename WarpMma::IteratorB up_iterator_b(
        typename WarpMma::IteratorB::TensorRef(
            reinterpret_cast<Element*>(weight_tile[warp][1]),
            cutlass::layout::ColumnMajor(32)),
        lane);
    typename WarpMma::FragmentA fragment_a;
    typename WarpMma::FragmentB gate_fragment_b;
    typename WarpMma::FragmentB up_fragment_b;
    typename WarpMma::TransformedFragmentA transformed_a;
    typename WarpMma::TransformedFragmentB transformed_gate_b;
    typename WarpMma::TransformedFragmentB transformed_up_b;
    iterator_a.load(fragment_a);
    gate_iterator_b.load(gate_fragment_b);
    up_iterator_b.load(up_fragment_b);
    warp_mma.transform(
        transformed_a, transformed_gate_b, fragment_a, gate_fragment_b);
    warp_mma.transform(
        transformed_a, transformed_up_b, fragment_a, up_fragment_b);
    warp_mma(
        gate_accumulator,
        transformed_a,
        transformed_gate_b,
        gate_accumulator);
    warp_mma(
        up_accumulator,
        transformed_a,
        transformed_up_b,
        up_accumulator);
    __syncwarp();
  }

  typename WarpMma::IteratorC gate_iterator_c(
      typename WarpMma::IteratorC::TensorRef(
          output_tile_shared[warp][0], cutlass::layout::RowMajor(8)),
      lane);
  typename WarpMma::IteratorC up_iterator_c(
      typename WarpMma::IteratorC::TensorRef(
          output_tile_shared[warp][1], cutlass::layout::RowMajor(8)),
      lane);
  gate_iterator_c.store(gate_accumulator);
  up_iterator_c.store(up_accumulator);
  __syncwarp();
  if (lane < 8) {
    const int column = n_block * 32 + warp * 8 + lane;
    const float scale = activation_scale[assignment];
    const float bias = bias_correction[assignment];
    float gate = output_tile_shared[warp][0][lane] * scale + bias;
    float up = output_tile_shared[warp][1][lane] * scale + bias;
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const float value = gate / (1.0f + expf(-gate)) * up;
    output[assignment * intermediate_size + column] =
        __float2bfloat16_rn(value);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> prepare_activation_cuda(
    const at::Tensor& activation,
    const at::Tensor& expert_ids,
    const at::Tensor& weight_scale,
    const at::Tensor& weight_bias,
    const at::Tensor& rht_sign,
    int64_t rht_block_size) {
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  TORCH_CHECK(activation.scalar_type() == at::kBFloat16,
              "activation must use bfloat16");
  TORCH_CHECK(activation.dim() == 2, "activation must be two-dimensional");
  TORCH_CHECK(activation.is_contiguous(), "activation must be contiguous");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == at::kLong,
              "expert_ids must be CUDA int64");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(weight_scale.is_cuda() && weight_scale.scalar_type() == at::kFloat,
              "weight_scale must be CUDA float32");
  TORCH_CHECK(weight_bias.is_cuda() && weight_bias.scalar_type() == at::kFloat,
              "weight_bias must be CUDA float32");
  TORCH_CHECK(rht_sign.is_cuda() && rht_sign.scalar_type() == at::kChar,
              "rht_sign must be CUDA int8");
  TORCH_CHECK(weight_scale.is_contiguous() && weight_bias.is_contiguous() &&
                  rht_sign.is_contiguous(),
              "expert transform tensors must be contiguous");
  TORCH_CHECK(rht_block_size == kRhtBlock,
              "CUDA VQ2A8 preparation requires rht_block_size=128");

  const auto size_m = activation.size(0);
  const auto size_k = activation.size(1);
  TORCH_CHECK(size_m > 0, "activation must contain at least one assignment");
  TORCH_CHECK(size_k % kRhtBlock == 0,
              "activation width must be divisible by 128");
  TORCH_CHECK(expert_ids.numel() == size_m,
              "expert_ids count must match activation rows");
  TORCH_CHECK(weight_scale.size(1) == size_k &&
                  weight_bias.sizes() == weight_scale.sizes() &&
                  rht_sign.sizes() == weight_scale.sizes(),
              "expert transform shapes must agree with activation width");

  c10::cuda::CUDAGuard device_guard(activation.device());
  const int num_rht_blocks = static_cast<int>(size_k / kRhtBlock);
  const auto float_options = activation.options().dtype(at::kFloat);
  auto transformed = at::empty({size_m, size_k}, float_options);
  auto partial_amax = at::empty({size_m, num_rht_blocks}, float_options);
  auto partial_bias = at::empty({size_m, num_rht_blocks}, float_options);
  auto quantized = at::empty(
      {size_m, size_k},
      activation.options().dtype(at::ScalarType::Float8_e4m3fn));
  auto activation_scale = at::empty({size_m}, float_options);
  auto bias_correction = at::empty({size_m}, float_options);

  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned int>(size_m), num_rht_blocks);
  vq2a8_fwht_transform_kernel<<<grid, kRhtBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(activation.data_ptr()),
      expert_ids.data_ptr<int64_t>(),
      weight_scale.data_ptr<float>(),
      weight_bias.data_ptr<float>(),
      rht_sign.data_ptr<int8_t>(),
      transformed.data_ptr<float>(),
      partial_amax.data_ptr<float>(),
      partial_bias.data_ptr<float>(),
      static_cast<int>(size_k),
      num_rht_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  vq2a8_quantize_kernel<<<size_m, kQuantThreads, 0, stream>>>(
      transformed.data_ptr<float>(),
      partial_amax.data_ptr<float>(),
      partial_bias.data_ptr<float>(),
      reinterpret_cast<__nv_fp8_storage_t*>(quantized.data_ptr()),
      activation_scale.data_ptr<float>(),
      bias_correction.data_ptr<float>(),
      static_cast<int>(size_k),
      num_rht_blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {quantized, activation_scale, bias_correction};
}

at::Tensor batched_gemm_cuda(
    const at::Tensor& activation,
    const at::Tensor& activation_scale,
    const at::Tensor& expert_ids,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& bias_correction) {
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous(),
              "activation must be contiguous CUDA");
  TORCH_CHECK(
      activation.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "activation must use float8_e4m3fn");
  TORCH_CHECK(activation.dim() == 2, "activation must be two-dimensional");
  TORCH_CHECK(activation_scale.is_cuda() &&
                  activation_scale.scalar_type() == at::kFloat &&
                  activation_scale.is_contiguous(),
              "activation_scale must be contiguous CUDA float32");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == at::kLong &&
                  expert_ids.is_contiguous(),
              "expert_ids must be contiguous CUDA int64");
  TORCH_CHECK(packed_indices.is_cuda() &&
                  packed_indices.scalar_type() == at::kInt &&
                  packed_indices.is_contiguous(),
              "packed_indices must be contiguous CUDA int32");
  TORCH_CHECK(codebooks.is_cuda() && codebooks.is_contiguous() &&
                  codebooks.scalar_type() ==
                      at::ScalarType::Float8_e4m3fn,
              "codebooks must be contiguous CUDA float8_e4m3fn");
  TORCH_CHECK(codebook_tile_ids.is_cuda() &&
                  codebook_tile_ids.scalar_type() == at::kByte &&
                  codebook_tile_ids.is_contiguous(),
              "codebook_tile_ids must be contiguous CUDA uint8");
  TORCH_CHECK(bias_correction.is_cuda() &&
                  bias_correction.scalar_type() == at::kFloat &&
                  bias_correction.is_contiguous(),
              "bias_correction must be contiguous CUDA float32");

  const int64_t size_m = activation.size(0);
  const int64_t size_k = activation.size(1);
  TORCH_CHECK(packed_indices.dim() == 3 && codebooks.dim() == 5 &&
                  codebook_tile_ids.dim() == 2,
              "invalid VQ2A8 packed tensor ranks");
  const int64_t size_n = packed_indices.size(1) * 2;
  TORCH_CHECK(size_n % 32 == 0 && size_k % 32 == 0,
              "VQ2A8 GEMM requires N%32=0 and K%32=0");
  TORCH_CHECK(packed_indices.size(2) * 8 == size_k,
              "packed indices do not cover K");
  TORCH_CHECK(codebooks.size(0) == packed_indices.size(0) &&
                  codebooks.size(2) * 32 == size_n &&
                  codebooks.size(3) == 16 && codebooks.size(4) == 2,
              "codebook shape does not match packed indices");
  TORCH_CHECK(codebook_tile_ids.size(0) == packed_indices.size(0) &&
                  codebook_tile_ids.size(1) == size_k,
              "codebook tile IDs do not match packed indices");
  TORCH_CHECK(expert_ids.numel() == size_m &&
                  activation_scale.numel() == size_m &&
                  bias_correction.numel() == size_m,
              "assignment metadata size mismatch");

  c10::cuda::CUDAGuard device_guard(activation.device());
  auto output = at::empty(
      {size_m, size_n}, activation.options().dtype(at::kBFloat16));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned int>(size_m),
                  static_cast<unsigned int>(size_n / 32));
  vq2a8_batched_gemm_kernel<false><<<grid, 128, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(activation.data_ptr()),
      activation_scale.data_ptr<float>(),
      expert_ids.data_ptr<int64_t>(),
      packed_indices.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(codebooks.data_ptr()),
      codebook_tile_ids.data_ptr<uint8_t>(),
      bias_correction.data_ptr<float>(),
      nullptr,
      nullptr,
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      static_cast<int>(codebooks.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor gate_up_cuda(
    const at::Tensor& activation,
    const at::Tensor& activation_scale,
    const at::Tensor& expert_ids,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& bias_correction,
    double swiglu_limit) {
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous() &&
                  activation.scalar_type() ==
                      at::ScalarType::Float8_e4m3fn,
              "activation must be contiguous CUDA float8_e4m3fn");
  TORCH_CHECK(activation.dim() == 2, "activation must be two-dimensional");
  TORCH_CHECK(activation_scale.is_cuda() &&
                  activation_scale.scalar_type() == at::kFloat &&
                  activation_scale.is_contiguous(),
              "activation_scale must be contiguous CUDA float32");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == at::kLong &&
                  expert_ids.is_contiguous(),
              "expert_ids must be contiguous CUDA int64");
  TORCH_CHECK(packed_indices.is_cuda() &&
                  packed_indices.scalar_type() == at::kInt &&
                  packed_indices.is_contiguous(),
              "packed_indices must be contiguous CUDA int32");
  TORCH_CHECK(codebooks.is_cuda() && codebooks.is_contiguous() &&
                  codebooks.scalar_type() ==
                      at::ScalarType::Float8_e4m3fn,
              "codebooks must be contiguous CUDA float8_e4m3fn");
  TORCH_CHECK(codebook_tile_ids.is_cuda() &&
                  codebook_tile_ids.scalar_type() == at::kByte &&
                  codebook_tile_ids.is_contiguous(),
              "codebook_tile_ids must be contiguous CUDA uint8");
  TORCH_CHECK(bias_correction.is_cuda() &&
                  bias_correction.scalar_type() == at::kFloat &&
                  bias_correction.is_contiguous(),
              "bias_correction must be contiguous CUDA float32");

  const int64_t size_m = activation.size(0);
  const int64_t size_k = activation.size(1);
  const int64_t intermediate_size = packed_indices.size(1);
  TORCH_CHECK(intermediate_size % 32 == 0 && size_k % 32 == 0 &&
                  packed_indices.size(2) * 8 == size_k,
              "invalid VQ2A8 gate/up shape");
  TORCH_CHECK(codebooks.dim() == 5 &&
                  codebooks.size(0) == packed_indices.size(0) &&
                  codebooks.size(2) * 16 == intermediate_size &&
                  codebooks.size(3) == 16 && codebooks.size(4) == 2,
              "codebook shape does not match packed indices");
  TORCH_CHECK(codebook_tile_ids.size(0) == packed_indices.size(0) &&
                  codebook_tile_ids.size(1) == size_k,
              "codebook tile IDs do not match packed indices");
  TORCH_CHECK(expert_ids.numel() == size_m &&
                  activation_scale.numel() == size_m &&
                  bias_correction.numel() == size_m,
              "assignment metadata size mismatch");

  c10::cuda::CUDAGuard device_guard(activation.device());
  auto output = at::empty(
      {size_m, intermediate_size}, activation.options().dtype(at::kBFloat16));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned int>(size_m),
                  static_cast<unsigned int>(intermediate_size / 32));
  vq2a8_gate_up_kernel<<<grid, 128, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(activation.data_ptr()),
      activation_scale.data_ptr<float>(),
      expert_ids.data_ptr<int64_t>(),
      packed_indices.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(codebooks.data_ptr()),
      codebook_tile_ids.data_ptr<uint8_t>(),
      bias_correction.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int>(intermediate_size),
      static_cast<int>(size_k),
      static_cast<int>(codebooks.size(1)),
      static_cast<float>(swiglu_limit));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor down_reduce_cuda(
    const at::Tensor& activation,
    const at::Tensor& activation_scale,
    const at::Tensor& expert_ids,
    const at::Tensor& token_ids,
    const at::Tensor& routing_weights,
    int64_t num_tokens,
    const at::Tensor& packed_indices,
    const at::Tensor& codebooks,
    const at::Tensor& codebook_tile_ids,
    const at::Tensor& bias_correction) {
  TORCH_CHECK(activation.is_cuda() && activation.is_contiguous() &&
                  activation.scalar_type() ==
                      at::ScalarType::Float8_e4m3fn,
              "activation must be contiguous CUDA float8_e4m3fn");
  TORCH_CHECK(activation.dim() == 2, "activation must be two-dimensional");
  TORCH_CHECK(activation_scale.is_cuda() &&
                  activation_scale.scalar_type() == at::kFloat &&
                  activation_scale.is_contiguous(),
              "activation_scale must be contiguous CUDA float32");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == at::kLong &&
                  expert_ids.is_contiguous(),
              "expert_ids must be contiguous CUDA int64");
  TORCH_CHECK(token_ids.is_cuda() && token_ids.scalar_type() == at::kLong &&
                  token_ids.is_contiguous(),
              "token_ids must be contiguous CUDA int64");
  TORCH_CHECK(routing_weights.is_cuda() &&
                  routing_weights.scalar_type() == at::kFloat &&
                  routing_weights.is_contiguous(),
              "routing_weights must be contiguous CUDA float32");
  TORCH_CHECK(packed_indices.is_cuda() &&
                  packed_indices.scalar_type() == at::kInt &&
                  packed_indices.is_contiguous(),
              "packed_indices must be contiguous CUDA int32");
  TORCH_CHECK(codebooks.is_cuda() && codebooks.is_contiguous() &&
                  codebooks.scalar_type() ==
                      at::ScalarType::Float8_e4m3fn,
              "codebooks must be contiguous CUDA float8_e4m3fn");
  TORCH_CHECK(codebook_tile_ids.is_cuda() &&
                  codebook_tile_ids.scalar_type() == at::kByte &&
                  codebook_tile_ids.is_contiguous(),
              "codebook_tile_ids must be contiguous CUDA uint8");
  TORCH_CHECK(bias_correction.is_cuda() &&
                  bias_correction.scalar_type() == at::kFloat &&
                  bias_correction.is_contiguous(),
              "bias_correction must be contiguous CUDA float32");

  const int64_t size_m = activation.size(0);
  const int64_t size_k = activation.size(1);
  const int64_t size_n = packed_indices.size(1) * 2;
  TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
  TORCH_CHECK(expert_ids.numel() == size_m && token_ids.numel() == size_m &&
                  routing_weights.numel() == size_m &&
                  activation_scale.numel() == size_m &&
                  bias_correction.numel() == size_m,
              "assignment metadata size mismatch");
  TORCH_CHECK(size_n % 32 == 0 && size_k % 32 == 0 &&
                  packed_indices.size(2) * 8 == size_k,
              "invalid VQ2A8 down shape");
  TORCH_CHECK(codebooks.dim() == 5 &&
                  codebooks.size(0) == packed_indices.size(0) &&
                  codebooks.size(2) * 32 == size_n &&
                  codebooks.size(3) == 16 && codebooks.size(4) == 2,
              "codebook shape does not match packed indices");
  TORCH_CHECK(codebook_tile_ids.size(0) == packed_indices.size(0) &&
                  codebook_tile_ids.size(1) == size_k,
              "codebook tile IDs do not match packed indices");

  c10::cuda::CUDAGuard device_guard(activation.device());
  auto output = at::zeros(
      {num_tokens, size_n}, activation.options().dtype(at::kFloat));
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const dim3 grid(static_cast<unsigned int>(size_m),
                  static_cast<unsigned int>(size_n / 32));
  vq2a8_batched_gemm_kernel<true><<<grid, 128, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(activation.data_ptr()),
      activation_scale.data_ptr<float>(),
      expert_ids.data_ptr<int64_t>(),
      packed_indices.data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(codebooks.data_ptr()),
      codebook_tile_ids.data_ptr<uint8_t>(),
      bias_correction.data_ptr<float>(),
      token_ids.data_ptr<int64_t>(),
      routing_weights.data_ptr<float>(),
      output.data_ptr<float>(),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      static_cast<int>(codebooks.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.to(at::kBFloat16);
}

}  // namespace

TORCH_LIBRARY(vq2a8_cuda, m) {
  m.def(
      "prepare_activation(Tensor activation, Tensor expert_ids, "
      "Tensor weight_scale, Tensor weight_bias, Tensor rht_sign, "
      "int rht_block_size=128) -> (Tensor, Tensor, Tensor)");
  m.def(
      "batched_gemm(Tensor activation, Tensor activation_scale, "
      "Tensor expert_ids, Tensor packed_indices, Tensor codebooks, "
      "Tensor codebook_tile_ids, Tensor bias_correction) -> Tensor");
  m.def(
      "gate_up(Tensor activation, Tensor activation_scale, Tensor expert_ids, "
      "Tensor packed_indices, Tensor codebooks, Tensor codebook_tile_ids, "
      "Tensor bias_correction, float swiglu_limit=0.0) -> Tensor");
  m.def(
      "down_reduce(Tensor activation, Tensor activation_scale, "
      "Tensor expert_ids, Tensor token_ids, Tensor routing_weights, "
      "int num_tokens, Tensor packed_indices, Tensor codebooks, "
      "Tensor codebook_tile_ids, Tensor bias_correction) -> Tensor");
}

TORCH_LIBRARY_IMPL(vq2a8_cuda, CUDA, m) {
  m.impl("prepare_activation", &prepare_activation_cuda);
  m.impl("batched_gemm", &batched_gemm_cuda);
  m.impl("gate_up", &gate_up_cuda);
  m.impl("down_reduce", &down_reduce_cuda);
}
