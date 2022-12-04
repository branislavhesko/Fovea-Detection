import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork

from models.focal_net_backbone import FocalNet as FocalNetBackbone
from models.decoder import _CentralPart, _DecoderBlock, _FinalBlock
from models.transformer import Transformer

class FocalNet(nn.Module):
    CHANNELS = [96, 192, 384, 768]

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = FocalNetBackbone()
        self.central_part = _CentralPart(768, 512)
        self.transformer = Transformer(
            num_tokens=16 ** 2,
            num_encoder_layers=4,
            num_decoder_layers=4,
            d_model=512,
            num_output_queries=16 ** 2 + 2,
            nhead=8
        )
        self.fpn = FeaturePyramidNetwork(in_channels_list=self.CHANNELS, out_channels=256)
        self.coarse_to_position = nn.Conv2d(512, 1, kernel_size=3)

    def _remap(self, features):
        features_dict = {}
        for idx in range(1, len(features) + 1):
            features_dict[f"layer{idx}"] = features[idx - 1]
        return features_dict

    def forward(self, x):
        b, c, h, w = x.shape
        features = self.backbone(x)
        layer4_out = self.central_part(features[3])
        # TODO: Use position queries ass aux loss.
        position_queries, mask_queries = self.transformer(layer4_out)
        fpn_out = self.fpn(self._remap(features))
        fine_features = fpn_out["layer1"]
        coarse_features = self.coarse_to_position(mask_queries)
        max_position = torch.argmax(coarse_features.view(coarse_features.shape[0], -1), dim=-1)
        windows = F.unfold(fine_features, kernel_size=17, padding=8, stride=8).view(fine_features.shape[0], fine_features.shape[1], 17, 17, -1)
        return windows[..., max_position]


if __name__ == '__main__':
    model = FocalNet(1)
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    print(y)