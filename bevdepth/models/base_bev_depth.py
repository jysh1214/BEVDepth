from torch import nn

from bevdepth.layers.backbones.base_lss_fpn import BaseLSSFPN
from bevdepth.layers.heads.bev_depth_head import BEVDepthHead

__all__ = ['BaseBEVDepth']


class BaseBEVDepth(nn.Module):
    """Source code of `BEVDepth`, `https://arxiv.org/abs/2112.11790`.

    Args:
        backbone_conf (dict): Config of backbone.
        head_conf (dict): Config of head.
        is_train_depth (bool): Whether to return depth.
            Default: False.
    """

    # TODO: Reduce grid_conf and data_aug_conf
    def __init__(self, backbone_conf, head_conf, is_train_depth=False):
        super(BaseBEVDepth, self).__init__()
        self.backbone = BaseLSSFPN(**backbone_conf)
        self.head = BEVDepthHead(**head_conf)
        self.is_train_depth = is_train_depth

    def forward(
        self,
        x,
        mats_dict,
        timestamps=None,
    ):
        """Forward function for BEVDepth

        Args:
            x (Tensor): Input ferature map.
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            timestamps (long): Timestamp.
                Default: None.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        if self.is_train_depth and self.training:
            x, depth_pred = self.backbone(x,
                                          mats_dict,
                                          timestamps,
                                          is_return_depth=True)
            preds = self.head(x)
            return preds, depth_pred
        else:
            x = self.backbone(x, mats_dict, timestamps)
            preds = self.head(x)
            return preds

    def get_targets(self, gt_boxes, gt_labels):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        return self.head.get_targets(gt_boxes, gt_labels)

    def loss(self, targets, preds_dicts):
        """Loss function for BEVDepth.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        return self.head.loss(targets, preds_dicts)

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        return self.head.get_bboxes(preds_dicts, img_metas, img, rescale)


import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
import torchvision.models as models
from torch_mlir import torchscript

@MODELS.register_module()
class ResNet(nn.Module):
    def __init__(self,
            depth=50,
            out_indices=[0, 1, 2, 3],
            norm_eval=False,
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            frozen_stages=0,
    ):
        super().__init__()
        self.model = models.resnet50()
        self.model.eval()
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    backbone = BaseLSSFPN(
        [-51.2, 51.2, 0.8],
        [-51.2, 51.2, 0.8],
        [-5, 3, 8],
        [2.0, 58.0, 0.5],
        (256, 704),
        16,
        80,
        {'type': 'ResNet', 'depth': 50, 'frozen_stages': 0, 'out_indices': [0, 1, 2, 3], 'norm_eval': False, 'init_cfg': {'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}},
        {'type': 'SECONDFPN', 'in_channels': [256, 512, 1024, 2048], 'upsample_strides': [0.25, 0.5, 1, 2], 'out_channels': [128, 128, 128, 128]},
        {'in_channels': 512, 'mid_channels': 512},
        False,
    )
    print(backbone)

    backbone.eval()
    backbone(
        torch.randn(1, 2, 6, 3, 256, 704, dtype=torch.float32),
        {
            "sensor2ego_mats": torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
            "intrin_mats": torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
            "ida_mats": torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
            "sensor2sensor_mats": torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
            "bda_mat": torch.randn(1, 4, 4, dtype=torch.float32),
        },
        None,
        False,
    )
    # backbone_mlir_model = torchscript.compile(
    #     backbone,
    #     [
    #         torch.randn(1, 2, 6, 3, 256, 704, dtype=torch.float32),
    #         torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
    #         torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
    #         torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
    #         torch.randn(1, 2, 6, 4, 4, dtype=torch.float32),
    #         torch.randn(1, 4, 4, dtype=torch.float32),
    #         # None,
    #         # False,
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(backbone_mlir_model, "BaseLSSFPN.mlirbc")
