import torch
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, ReLU, Sequential


class FoveaNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decoder4 = _DecoderBlock(*(1024, 512, 256))

    def forward(self, input_):
        pass


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