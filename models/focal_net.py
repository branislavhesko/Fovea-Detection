import torch
import torch.nn as nn

from models.focal_net_backbone import FocalNet as FocalNetBackbone
from models.decoder import _CentralPart, _DecoderBlock, _FinalBlock
from models.transformer import Transformer

class FocalNet(nn.Module):
    CHANNELS = [96, 192, 384, 768]

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = FocalNetBackbone()
        self.central_part = _CentralPart(768, 512)
        self.decoder4 = _DecoderBlock(512, 256, 256)
        self.decoder3 = _DecoderBlock(640, 256, 256)
        self.decoder2 = _DecoderBlock(448, 256, 256)
        self.decoder1 = _DecoderBlock(352, 256, 256)
        self.final_block = _FinalBlock(256, num_classes)
        self.transformer = Transformer(
            num_tokens=16 ** 2,
            num_encoder_layers=4,
            num_decoder_layers=4,
            d_model=512,
            num_output_queries=16 ** 2 + 2,
            nhead=8
        )

    def forward(self, x):
        b, c, h, w = x.shape
        features = self.backbone(x)
        layer4_out = self.central_part(features[3])
        # TODO: Use position queries ass aux loss.
        position_queries, mask_queries = self.transformer(layer4_out)
        layer3_out = self.decoder3(torch.cat([self.decoder4(mask_queries), features[2]], dim=1))
        layer2_out = self.decoder2(torch.cat([layer3_out, features[1]], dim=1))
        layer1_out = self.decoder1(torch.cat([layer2_out, features[0]], dim=1))

        return torch.clamp(
            nn.functional.interpolate(torch.sigmoid(self.final_block(layer1_out)), size=(h, w), mode='bilinear', align_corners=True),
            min=0,
            max=1.0 - 1e-4
        )


if __name__ == '__main__':
    model = FocalNet(1)
    x = torch.randn(8, 3, 512, 512)
    y = model(x)
    print(y.shape)