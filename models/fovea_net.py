import torch
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sequential


class FoveaNet(torch.nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.decoder4 = _DecoderBlock(*(1024 + 1024, 1024, 512))
        self.decoder3 = _DecoderBlock(*(1024, 768, 512))
        self.decoder2 = _DecoderBlock(*(256 + 512, 384, 256))
        self.decoder1 = _DecoderBlock(*(64 + 256, 192, 128))
        self.central_part = _CentralPart(2048, 1024)
        self.final_part = _FinalBlock(128, num_classes)

    def forward(self, input_):
        x = self.backbone.conv1(input_)
        x = self.backbone.bn1(x)
        enc1 = self.backbone.relu(x)
        x = self.backbone.maxpool(enc1)
        layer1_out = self.backbone.layer1(x)
        layer2_out = self.backbone.layer2(layer1_out)
        layer3_out = self.backbone.layer3(layer2_out)
        layer4_out = self.backbone.layer4(layer3_out)
        dec_4 = self.decoder4(torch.cat([self.central_part(layer4_out), layer3_out], dim=1))
        dec_3 = self.decoder3(torch.cat([layer2_out, dec_4], dim=1))
        dec_2 = self.decoder2(torch.cat([layer1_out, dec_3], dim=1))
        dec_1 = self.decoder1(torch.cat([enc1, dec_2], dim=1))
        return self.final_part(dec_1)


class _DecoderBlock(torch.nn.Module):

    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()

        layers = [
            Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            ConvTranspose2d(middle_channels, out_channels,
                            kernel_size=2, stride=2)
        ]

        self.decode = Sequential(*layers)

    def forward(self, x):
        return self.decode(x)


class _FinalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.final_block = Sequential(*[
            Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            BatchNorm2d(in_channels),
            ReLU(inplace=True),
            Conv2d(in_channels, out_channels, kernel_size=1)
        ])

    def forward(self, x):
        return self.final_block(x)


class _CentralPart(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        middle_channels = out_channels + (in_channels - out_channels) // 2
        self.central_part = Sequential(*[
            Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            BatchNorm2d(middle_channels),
            ReLU(inplace=True),
            ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        ])

    def forward(self, x):
        return self.central_part(x)


if __name__ == "__main__":
    input_ = torch.rand(2, 3, 128, 128).cuda()
    model = FoveaNet(2).cuda()
    print(model(input_).shape)