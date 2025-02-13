// Copyright (c) Megvii Inc. All rights reserved.
#include <torch/script.h>

#include <vector>

#define CHECK_CUDA(x)
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor voxel_pooling_inference_forward_wrapper(
    int64_t batch_size, int64_t num_cams, int64_t num_depth, int64_t num_height, int64_t num_width,
    int64_t num_channels, int64_t num_voxel_x, int64_t num_voxel_y, int64_t num_voxel_z,
    torch::Tensor geom_xyz_tensor, torch::Tensor depth_features_tensor,
    torch::Tensor context_features_tensor, torch::Tensor output_features_tensor);

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *depth_features,
    const float *context_features, float *output_features) {
  const int total_samples = batch_size * num_cams * num_depth * num_height * num_width;
  for (int sample_idx = 0; sample_idx < total_samples; ++sample_idx) {
    const int sample_x = geom_xyz[sample_idx * 3 + 0];
    const int sample_y = geom_xyz[sample_idx * 3 + 1];
    const int sample_z = geom_xyz[sample_idx * 3 + 2];

    if (sample_x < 0 || sample_x >= num_voxel_x ||
        sample_y < 0 || sample_y >= num_voxel_y ||
        sample_z < 0 || sample_z >= num_voxel_z) {
        continue;
    }

    const int batch_idx = sample_idx / (num_cams * num_depth * num_height * num_width);
    const int width_idx = sample_idx % num_width;
    const int height_idx = (sample_idx / num_width) % num_height;
    const int cam_idx = (sample_idx / (num_depth * num_height * num_width)) % num_cams;
    const float depth_val = depth_features[sample_idx];
    const int base_out_idx = 
            (batch_idx * num_voxel_y * num_voxel_x +
             sample_y * num_voxel_x +
             sample_x) 
             * num_channels;

    for (int c = 0; c < num_channels; ++c) {
      const int context_idx =
                batch_idx * (num_cams * num_channels * num_height * num_width) +
                cam_idx   * (num_channels * num_height * num_width) +
                c         * (num_height * num_width) +
                height_idx * num_width +
                width_idx;

      const float context_val = context_features[context_idx];
      const float prod = depth_val * context_val;

      output_features[base_out_idx + c] += prod;
    }
  }
}

torch::Tensor voxel_pooling_inference_forward_wrapper(
    int64_t batch_size, int64_t num_cams, int64_t num_depth, int64_t num_height, int64_t num_width,
    int64_t num_channels, int64_t num_voxel_x, int64_t num_voxel_y, int64_t num_voxel_z,
    torch::Tensor geom_xyz_tensor, torch::Tensor depth_features_tensor,
    torch::Tensor context_features_tensor, torch::Tensor output_features_tensor) {
  assert(depth_features_tensor.dtype() == at::kFloat);
  CHECK_INPUT(geom_xyz_tensor);
  CHECK_INPUT(depth_features_tensor);
  CHECK_INPUT(context_features_tensor);
  const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
  const float *depth_features = depth_features_tensor.data_ptr<float>();
  const float *context_features = context_features_tensor.data_ptr<float>();
  float *output_features = output_features_tensor.data_ptr<float>();
  voxel_pooling_inference_forward_kernel_launcher(
        batch_size, num_cams, num_depth, num_height, num_width, num_channels,
        num_voxel_x, num_voxel_y, num_voxel_z, geom_xyz, depth_features,
        context_features, output_features);
  return output_features_tensor.clone();
}

static auto registry =
  torch::RegisterOperators("customop::voxel_pooling_inference_forward_wrapper",
                            &voxel_pooling_inference_forward_wrapper);
