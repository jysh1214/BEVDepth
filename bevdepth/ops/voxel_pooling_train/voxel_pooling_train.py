# Copyright (c) Megvii Inc. All rights reserved.
import torch
from torch.autograd import Function

torch.ops.load_library(
    "/home/BEVDepth/bevdepth/ops/voxel_pooling_train/voxel_pooling_train_ext.cpython-39-x86_64-linux-gnu.so"
)


def _voxel_pooling_train_forward_wrapper(
    g,
    batch_size,
    num_points,
    num_channels,
    num_voxel_x,
    num_voxel_y,
    num_voxel_z,
    geom_xyz_tensor,
    input_features_tensor,
    output_features_tensor,
    pos_memo_tensor,
):
    return g.op(
        "customop::voxel_pooling_train_forward_wrapper",
        batch_size,
        num_points,
        num_channels,
        num_voxel_x,
        num_voxel_y,
        num_voxel_z,
        geom_xyz_tensor,
        input_features_tensor,
        output_features_tensor,
        pos_memo_tensor,
    )


from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic(
    "customop::voxel_pooling_train_forward_wrapper",
    _voxel_pooling_train_forward_wrapper, 11)


class VoxelPoolingTrain(Function):

    @staticmethod
    def forward(ctx, geom_xyz: torch.Tensor, input_features: torch.Tensor,
                voxel_num: torch.Tensor) -> torch.Tensor:
        """Forward function for `voxel pooling.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape
                of [B, N, 3].
            input_features (Tensor): feature for each voxel with the
                shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the
                shape of [3].

        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        grad_input_features = torch.zeros_like(input_features)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1]))
        assert geom_xyz.shape[1] == input_features.shape[1]
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(batch_size, voxel_num[1],
                                                   voxel_num[0], num_channels)
        # Save the position of bev_feature_map for each input point.
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        torch.ops.customop.voxel_pooling_train_forward_wrapper(
            batch_size,
            num_points,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            geom_xyz,
            input_features,
            output_features,
            pos_memo,
        )
        # save grad_input_features and pos_memo for backward
        ctx.save_for_backward(grad_input_features, pos_memo)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        kept = (pos_memo != -1)[..., 0]
        grad_input_features_shape = grad_input_features.shape
        grad_input_features = grad_input_features.reshape(
            grad_input_features.shape[0], -1, grad_input_features.shape[-1])
        grad_input_features[kept] = grad_output_features[
            pos_memo[kept][..., 0].long(), :, pos_memo[kept][..., 1].long(),
            pos_memo[kept][..., 2].long()]
        grad_input_features = grad_input_features.reshape(
            grad_input_features_shape)
        return None, grad_input_features, None


voxel_pooling_train = VoxelPoolingTrain.apply
