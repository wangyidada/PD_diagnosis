import torch
import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inplanes, out_planes,  stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.conv3 = nn.Conv3d(
            out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes * self.expansion)

        self.relu = nn.PReLU()
        self.down_sample = nn.Conv3d(inplanes, out_planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm3d(out_planes * self.expansion)

    def forward(self, x):
        residual = self.down_sample(x)
        residual = self.bn(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,  in_channels, block, layers, out_planes=[64, 128, 256, 512], num_classes=2, expansion=4):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.expansion = expansion
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu1 = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, out_planes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, out_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_planes[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(out_planes[3] * self.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, input):
        x= input[:, 0, ...]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50(in_channels=1,  block=Bottleneck, num_classes=2, **kwargs):
    model = ResNet(in_channels=in_channels, block=block, layers=[3, 4, 6, 3], num_classes=num_classes,**kwargs)
    return model
