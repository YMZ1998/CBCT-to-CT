from collections import OrderedDict
from typing import Dict

import timm
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import ops


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


activation_layer = nn.ReLU(inplace=True)
# activation_layer = nn.LeakyReLU(0.1, inplace=True)

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        middle_channels = in_channels // 2
        self.conv = nn.Sequential(
            # nn.ConvTranspose2d(in_channels, middle_channels, 4, 2, 1, 0, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            activation_layer,
            nn.Conv2d(middle_channels, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.conv(x)


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                      padding=((kernel_size + 2 * (dilation - 1)) - 1) // 2, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_layer
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        # print("in_channels:", in_channels, "out_channels:", out_channels)
        super(DecoderBlock, self).__init__()
        middle_channels = int(in_channels // 2)

        self.conv1 = Conv(in_channels, middle_channels, kernel_size=3, dilation=1)
        self.up = UpConv(middle_channels, middle_channels)
        self.conv2 = Conv(middle_channels, out_channels, kernel_size=3, dilation=1)

        self.drop = ops.DropBlock2d(p=0.2, block_size=3, inplace=True)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        x = torch.cat([y, x], dim=1)
        x = self.drop(x)
        return x


class EfficientUNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1, pretrain_backbone=True, model_name=None, act=True):
        super(EfficientUNet, self).__init__()
        backbone = timm.create_model(model_name, pretrained=pretrain_backbone, in_chans=in_chans)
        self.stage_out_channels = [16, 24, 40, 112, 320]

        # stage_indices = [1, 2, 3, 5, 7]
        stage_indices = [2, 3, 4, 6, 8]
        return_layers = dict([(str(j), f"stage{i}") for i, j in enumerate(stage_indices)])
        self.backbone = IntermediateLayerGetter(backbone.as_sequential(), return_layers=return_layers)
        # self.backbone = IntermediateLayerGetter(backbone.features, return_layers=return_layers)

        self.up1 = DecoderBlock(self.stage_out_channels[4], self.stage_out_channels[3])
        self.up2 = DecoderBlock(self.stage_out_channels[3] * 2, self.stage_out_channels[2])
        self.up3 = DecoderBlock(self.stage_out_channels[2] * 2, self.stage_out_channels[1])
        self.up4 = DecoderBlock(self.stage_out_channels[1] * 2, self.stage_out_channels[0])
        self.outconv = OutConv(self.stage_out_channels[0] * 2, num_classes=num_classes)

        self.act = act
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(x)
        # for i in range(5):
        #     print(i, backbone_out['stage{}'.format(str(i))].shape)
        # encoder
        e0 = backbone_out['stage0']
        e1 = backbone_out['stage1']
        e2 = backbone_out['stage2']
        e3 = backbone_out['stage3']
        e4 = backbone_out['stage4']

        # decoder
        d4 = self.up1(e4, e3)
        d3 = self.up2(d4, e2)
        d2 = self.up3(d3, e1)
        d1 = self.up4(d2, e0)
        out = self.outconv(d1)

        if self.act:
            out = self.tanh(out)
        return out


if __name__ == '__main__':
    from torchsummary import summary

    model = EfficientUNet(in_chans=5, num_classes=1, pretrain_backbone=False, model_name='efficientnet_b0').to("cuda")
    summary(model, (5, 320, 320))

    # model2 = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3).to("cuda")
    # summary(model2, (3, 320, 320))
    # import os
    # cache_dir = torch.hub.get_dir()
    # weights_dir = os.path.join(cache_dir, "checkpoints")
    # print(f"预训练权重存储路径: {weights_dir}")
