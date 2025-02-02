import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer

from mmdet3d.registry import MODELS

import torchvision.models as models

import sys
import os
import io


# class Mlp(nn.Module):

#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act_layer=nn.ReLU,
#                  drop=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


# class SELayer(nn.Module):

#     def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
#         super().__init__()
#         self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
#         self.act1 = act_layer()
#         self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
#         self.gate = gate_layer()

#     def forward(self, x, x_se):
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         return x * self.gate(x_se)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                          size=x4.size()[2:],
                          mode='bilinear',
                          align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# class ConvBnReLU3D(nn.Module):
#     """Implements of 3d convolution + batch normalization + ReLU."""

#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int = 3,
#         stride: int = 1,
#         pad: int = 1,
#         dilation: int = 1,
#     ) -> None:
#         """initialization method for convolution3D +
#             batch normalization + relu module
#         Args:
#             in_channels: input channel number of convolution layer
#             out_channels: output channel number of convolution layer
#             kernel_size: kernel size of convolution layer
#             stride: stride of convolution layer
#             pad: pad of convolution layer
#             dilation: dilation of convolution layer
#         """
#         super(ConvBnReLU3D, self).__init__()
#         self.conv = nn.Conv3d(in_channels,
#                               out_channels,
#                               kernel_size,
#                               stride=stride,
#                               padding=pad,
#                               dilation=dilation,
#                               bias=False)
#         self.bn = nn.BatchNorm3d(out_channels)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """forward method"""
#         return F.relu(self.bn(self.conv(x)), inplace=True)


def save_model(model, output):
    bytecode_stream = io.BytesIO()
    model.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()
    with open(output, "wb") as file:
        file.write(bytecode)


if __name__ == "__main__":
    mid_channels = 512
    depth_channels = 112
    basic_block = BasicBlock(mid_channels, mid_channels)

    print(basic_block)

    x = torch.randn(6, 512, 16, 44)
    # basic_block(x)

    # compiled_basic_block = torch.compile(basic_block, backend="turbine_cpu")
    # compiled_basic_block(x)


    # depth_conv = nn.Sequential(
    #     BasicBlock(mid_channels, mid_channels),
    #     BasicBlock(mid_channels, mid_channels),
    #     BasicBlock(mid_channels, mid_channels),
    #     ASPP(mid_channels, mid_channels),
    #     build_conv_layer(cfg=dict(
    #         type='DCN',
    #         in_channels=mid_channels,
    #         out_channels=mid_channels,
    #         kernel_size=3,
    #         padding=1,
    #         groups=4,
    #         im2col_step=128,
    #     )),
    #     nn.Conv2d(mid_channels,
    #         depth_channels,
    #         kernel_size=1,
    #         stride=1,
    #         padding=0
    #     ),
    # )

    # print(depth_conv)
    # compiled_depth_conv = torch.compile(depth_conv, backend="turbine_cpu")
    # compiled_depth_conv(x)

    # dcn = build_conv_layer(cfg=dict(
    #     type='DCN',
    #     in_channels=mid_channels,
    #     out_channels=mid_channels,
    #     kernel_size=3,
    #     padding=1,
    #     groups=4,
    #     im2col_step=128,
    # ))

    # class DCN(nn.Module):
    #     def __init__(self, mid_channels=512):
    #         super(DCN, self).__init__()
    #         model = build_conv_layer(cfg=dict(
    #             type='DCN',
    #             in_channels=mid_channels,
    #             out_channels=mid_channels,
    #             kernel_size=3,
    #             padding=1,
    #             groups=4,
    #             im2col_step=128,
    #         ))
    #     def forward(self, x):
    #         return self.model(x)

    # dcn = DCN()

    # dcn = DeformConv2dPack(in_channels=512,
    #     out_channels=512,
    #     kernel_size=(3, 3),
    #     stride=(1, 1),
    #     padding=(1, 1),
    #     dilation=(1, 1),
    #     groups=4,
    #     deform_groups=1,
    #     bias=False
    # )
    # print(dcn)

    # compiled_dcn = torch.compile(dcn, backend="turbine_cpu")
    # compiled_dcn(x) # Segmentation fault (core dumped)



    # mlp = Mlp(27, 512, 512)
    # mlp.eval()
    # mlp_mlir_model = torchscript.compile(
    #     mlp,
    #     [torch.randn(6, 27)],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(mlp_mlir_model, "Mlp.mlirbc")

    # selayer = SELayer(512)
    # selayer.eval()
    # selayer_mlir_model = torchscript.compile(
    #     selayer,
    #     [
    #         torch.randn(6, 512, 16, 44, dtype=torch.float16),
    #         # torch.randn(6, 512, 1, 1, dtype=torch.float16),
    #         torch.randn(6, 512, 1, 1, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    #     verbose=True,
    # )
    # save_model(selayer_mlir_model, "SELayer.mlirbc")

    aspp = ASPP(512)
    aspp.eval()
    # aspp_mlir_model = torchscript.compile(
    #     aspp,
    #     [
    #         # torch.randn(6, 512, 16, 44, dtype=torch.float16)
    #         torch.randn(6, 512, 16, 44, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(aspp_mlir_model, "ASPP.mlirbc")

    compiled_aspp = torch.compile(aspp, backend="turbine_cpu")
    compiled_aspp(
        torch.randn(6, 512, 16, 44, dtype=torch.float32),
    )


    # convbnrelu3d_0 = ConvBnReLU3D(8, 16, 1, 1, 0, 1)
    # convbnrelu3d_0.eval()
    # convbnrelu3d_0_mlir_model = torchscript.compile(
    #     convbnrelu3d_0,
    #     [
    #         torch.randn(6, 8, 3, 64, 176, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(convbnrelu3d_0_mlir_model, "ConvBnReLU3D_0.mlirbc")

    # convbnrelu3d_1 = ConvBnReLU3D(16, 8, 1, 1, 0, 1)
    # convbnrelu3d_1.eval()
    # convbnrelu3d_1_mlir_model = torchscript.compile(
    #     convbnrelu3d_1,
    #     [
    #         # torch.randn(6, 16, 3, 64, 176, dtype=torch.float16),
    #         torch.randn(6, 16, 3, 64, 176, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(convbnrelu3d_1_mlir_model, "ConvBnReLU3D_1.mlirbc")

    # in_channels = 512
    # mid_channels = 512
    # depthnet_sequential = nn.Sequential(
    #     nn.Conv2d(in_channels,
    #         mid_channels,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1),
    #     nn.BatchNorm2d(mid_channels),
    #     nn.ReLU(inplace=True),
    # )
    # depthnet_sequential.eval()
    # depthnet_sequential_mlir_model = torchscript.compile(
    #     depthnet_sequential,
    #     [
    #         # torch.randn(6, 512, 16, 44, dtype=torch.float16),
    #         torch.randn(6, 512, 16, 44, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(depthnet_sequential_mlir_model, "depthnet_sequential.mlirbc")

    # context_channels = 80
    # depthnet_context_conv = nn.Conv2d(mid_channels,
    #     context_channels,
    #     kernel_size=1,
    #     stride=1,
    #     padding=0,
    # )
    # depthnet_context_conv.eval()
    # depthnet_context_conv_mlir_model = torchscript.compile(
    #     depthnet_context_conv,
    #     [
    #         # torch.randn(6, 512, 16, 44, dtype=torch.float16),
    #         torch.randn(6, 512, 16, 44, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(depthnet_context_conv_mlir_model, "depthnet_context_conv.mlirbc")

    # depthnet_bn = nn.BatchNorm1d(27)
    # depthnet_bn.eval()
    # depthnet_bn_mlir_model = torchscript.compile(
    #     depthnet_bn,
    #     [
    #         torch.randn(6, 27, dtype=torch.float32),
    #     ],
    #     output_type="linalg-on-tensors",
    #     use_tracing=True,
    # )
    # save_model(depthnet_bn_mlir_model, "depthnet_bn.mlirbc")

    @MODELS.register_module()
    class ResNet(nn.Module):
        def __init__(self,
                in_channels=80,
                depth=18,
                num_stages=3,
                strides=(1,2,2),
                dilations=(1,1,1),
                out_indices=[0, 1, 2],
                norm_eval=False,
                base_channels=160,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                frozen_stages=0,
        ):
            super().__init__()
            self.model = models.resnet50()
            self.model.eval()
        def forward(self, x):
            return self.model(x)
    
    bev_backbone_conf = {
        'type': 'ResNet',
        'in_channels': 80,
        'depth': 18,
        'num_stages': 3,
        'strides': (1, 2, 2),
        'dilations': (1, 1, 1),
        'out_indices': [0, 1, 2],
        'norm_eval': False,
        'base_channels': 160
    }
    trunk = MODELS.build(bev_backbone_conf)
    print(trunk)
