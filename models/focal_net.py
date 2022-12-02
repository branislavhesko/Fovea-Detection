import torch
import torch.nn as nn

from models.focal_net_backbone import FocalNet as FocalNetBackbone
from models.decoder import _CentralPart, _DecoderBlock, _FinalBlock


class FocalNet(nn.Module):
    CHANNELS = [96, 192, 384, 768]

    def __init__(self) -> None:
        super().__init__()
        self.backbone = FocalNetBackbone()
        self.central_part = _CentralPart(768, 512)
        self.decoder4 = _DecoderBlock(512, 256, 256)
        self.decoder3 = _DecoderBlock(640, 256, 256)
        self.decoder2 = _DecoderBlock(448, 256, 256)
        self.decoder1 = _DecoderBlock(352, 256, 256)
        self.final_block = _FinalBlock(256, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        features = self.backbone(x)
        layer4_out = self.central_part(features[3])
        layer3_out = self.decoder3(torch.cat([self.decoder4(layer4_out), features[2]], dim=1))
        layer2_out = self.decoder2(torch.cat([layer3_out, features[1]], dim=1))
        layer1_out = self.decoder1(torch.cat([layer2_out, features[0]], dim=1))

        return nn.functional.interpolate(self.final_block(layer1_out), size=(h, w), mode='bilinear', align_corners=True)


if __name__ == '__main__':
    model = FocalNet()
    x = torch.randn(8, 3, 512, 512)
    y = model(x)
    print(y.shape)