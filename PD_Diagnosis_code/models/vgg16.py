import torch
import torch.nn as nn


class VGG_module(nn.Module):
    def __init__(self, inplanes, out_planes, layer_num=2):
        super(VGG_module, self).__init__()
        self.layer_num=layer_num
        self.conv1 = nn.Conv3d(inplanes, out_planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)

        self.conv2 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)

        self.conv3 = nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.layer_num == 3:
            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)
            out = self.max_pool(out)
        else:
            out = self.max_pool(out)
        return out
class VGG16_3D(nn.Module):
    def __init__(self,  in_channels=1, out_planes=[64, 128, 256, 512], num_classes=2):
        super(VGG16_3D, self).__init__()
        self.layer1 = VGG_module(in_channels, out_planes[0], layer_num=2)
        self.layer2 = VGG_module(out_planes[0], out_planes[1], layer_num=2)
        self.layer3 = VGG_module(out_planes[1], out_planes[2], layer_num=3)
        self.layer4 = VGG_module(out_planes[2], out_planes[3], layer_num=3)
        self.layer5 = VGG_module(out_planes[3], out_planes[3], layer_num=3)

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_planes[3],  out_planes[3]),
            nn.Dropout(0.5),

            nn.Linear(out_planes[3], out_planes[1]),
            nn.Dropout(0.5),
            nn.Linear(out_planes[1], num_classes)
        )

    def forward(self, input):
        x = input[:, 0, ...]
        x = self.conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def vgg16(in_channels=1, num_classes=2):
    model = VGG16_3D(in_channels=in_channels, out_planes=[8, 16, 32, 64, 128, 256, 512], num_classes=num_classes)
    return model