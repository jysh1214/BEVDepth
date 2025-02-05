// Copyright (c) Megvii Inc. All rights reserved.
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <torch/script.h>

#include <vector>

#define CHECK_CUDA(x)
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void voxel_pooling_inference_forward_wrapper(
    int64_t batch_size, int64_t num_cams, int64_t num_depth, int64_t num_height, int64_t num_width,
    int64_t num_channels, int64_t num_voxel_x, int64_t num_voxel_y, int64_t num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor depth_features_tensor,
    at::Tensor context_features_tensor, at::Tensor output_features_tensor);

void voxel_pooling_inference_forward_kernel_launcher(
    int batch_size, int num_cams, int num_depth, int num_height, int num_width,
    int num_channels, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *depth_features,
    const float *context_features, float *output_features) {

}

void voxel_pooling_inference_forward_wrapper(
    int64_t batch_size, int64_t num_cams, int64_t num_depth, int64_t num_height, int64_t num_width,
    int64_t num_channels, int64_t num_voxel_x, int64_t num_voxel_y, int64_t num_voxel_z,
    at::Tensor geom_xyz_tensor, at::Tensor depth_features_tensor,
    at::Tensor context_features_tensor, at::Tensor output_features_tensor) {
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

  return;
}

static auto registry =
  torch::RegisterOperators("customop::voxel_pooling_inference_forward_wrapper",
                            &voxel_pooling_inference_forward_wrapper);
