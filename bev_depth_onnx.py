from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.base_exp import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel
from bevdepth.models.base_bev_depth import BaseBEVDepth

import torch
from onnxruntime.tools import pytorch_export_contrib_ops

class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.key_idxes = [-1]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.key_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.key_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weights'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = BaseBEVDepth(self.backbone_conf,
                                  self.head_conf,
                                  is_train_depth=True)

        # Export ONNX model
        x = torch.randn(1, 2, 6, 3, 256, 704, dtype=torch.float32)
        sensor2ego_mats = torch.randn(1, 2, 6, 4, 4, dtype=torch.float32)
        intrin_mats = torch.randn(1, 2, 6, 4, 4, dtype=torch.float32)
        ida_mats = torch.randn(1, 2, 6, 4, 4, dtype=torch.float32)
        sensor2sensor_mats = torch.randn(1, 2, 6, 4, 4, dtype=torch.float32)
        bda_mat = torch.randn(1, 4, 4, dtype=torch.float32)

        bevdepth = BaseBEVDepth(
            self.backbone_conf,
            self.head_conf,
            is_train_depth=True
        )

        pytorch_export_contrib_ops.register()
        onnx_model_path = "bevdepth.onnx"
        torch.onnx.export(
            bevdepth,
            (x, sensor2ego_mats, intrin_mats, ida_mats, sensor2sensor_mats, bda_mat),
            onnx_model_path,
            verbose=True,
            opset_version=11,
        )

        print(f"Saved ONNX model to '{onnx_model_path}'.")
        exit(0)


"""
python3 bev_depth_onnx.py --ckpt_path bev_depth_lss_r50_256x704_128x128_24e_2key.pth -e -b 1
"""
if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_lss_r50_256x704_128x128_24e_2key')
