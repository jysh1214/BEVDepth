# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

torch.ops.load_library(
    "/home/BEVDepth/bevdepth/ops/voxel_pooling_inference/voxel_pooling_inference_ext.cpython-39-x86_64-linux-gnu.so"
)

def _voxel_pooling_inference_forward_wrapper(
    g,
    batch_size,
    num_cams,
    num_depth,
    num_height,
    num_width,
    num_channels,
    num_voxel_x,
    num_voxel_y,
    num_voxel_z,
    geom_xyz_tensor,
    depth_features_tensor,
    context_features_tensor,
    output_features_tensor,
):
    return g.op(
        "sifive::VoxelPooling",
        batch_size,
        num_cams,
        num_depth,
        num_height,
        num_width,
        num_channels,
        num_voxel_x,
        num_voxel_y,
        num_voxel_z,
        geom_xyz_tensor,
        depth_features_tensor,
        context_features_tensor,
        output_features_tensor,
    )


from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic(
    "customop::voxel_pooling_inference_forward_wrapper",
    _voxel_pooling_inference_forward_wrapper, 11)


def voxel_pooling_inference(
    geom_xyz: torch.Tensor,
    depth_features: torch.Tensor,
    context_features: torch.Tensor,
    voxel_num: torch.Tensor
) -> torch.Tensor:
    batch_size = geom_xyz.shape[0]
    num_cams = geom_xyz.shape[1]
    num_depth = geom_xyz.shape[2]
    num_height = geom_xyz.shape[3]
    num_width = geom_xyz.shape[4]
    num_channels = context_features.shape[1]
    output_features = depth_features.new_zeros(
        (batch_size, voxel_num[1], voxel_num[0], num_channels))
    return torch.ops.customop.voxel_pooling_inference_forward_wrapper(
        batch_size,
        num_cams,
        num_depth,
        num_height,
        num_width,
        num_channels,
        voxel_num[0],
        voxel_num[1],
        voxel_num[2],
        geom_xyz,
        depth_features,
        context_features,
        output_features,
    ).permute(0, 3, 1, 2)
