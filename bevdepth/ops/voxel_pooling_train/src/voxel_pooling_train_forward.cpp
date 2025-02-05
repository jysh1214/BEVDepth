// Copyright (c) Megvii Inc. All rights reserved.
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>
#define CHECK_CUDA(x)
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void voxel_pooling_train_forward_wrapper(int64_t batch_size, int64_t num_points,
                                        int64_t num_channels, int64_t num_voxel_x,
                                        int64_t num_voxel_y, int64_t num_voxel_z,
                                        at::Tensor geom_xyz_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor output_features_tensor,
                                        at::Tensor pos_memo_tensor);

void voxel_pooling_train_forward_kernel_launcher(
    int batch_size, int num_points, int num_channels, int num_voxel_x,
    int num_voxel_y, int num_voxel_z, const int *geom_xyz,
    const float *input_features, float *output_features, int *pos_memo) {

}

void voxel_pooling_train_forward_wrapper(int64_t batch_size, int64_t num_points,
                                        int64_t num_channels, int64_t num_voxel_x,
                                        int64_t num_voxel_y, int64_t num_voxel_z,
                                        at::Tensor geom_xyz_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor output_features_tensor,
                                        at::Tensor pos_memo_tensor) {
  assert(input_features_tensor.dtype() == at::kFloat);
  CHECK_INPUT(geom_xyz_tensor);
  CHECK_INPUT(input_features_tensor);
  const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
  int *pos_memo = pos_memo_tensor.data_ptr<int>();

  const float *input_features = input_features_tensor.data_ptr<float>();
  float *output_features = output_features_tensor.data_ptr<float>();
  voxel_pooling_train_forward_kernel_launcher(
        batch_size, num_points, num_channels, num_voxel_x, num_voxel_y,
        num_voxel_z, geom_xyz, input_features, output_features, pos_memo);

  return;
}

static auto registry =
  torch::RegisterOperators("customop::voxel_pooling_train_forward_wrapper",
                            &voxel_pooling_train_forward_wrapper);
