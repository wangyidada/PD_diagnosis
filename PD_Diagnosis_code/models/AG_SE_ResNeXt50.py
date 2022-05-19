import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def show_feature(x, n=16, s=4, nr=4, nc=4, name=None, path=None):
    import os
    path = r'/homes/ydwang/projects/RJ_PD_dignosis/feature'
    plt.figure(figsize=(8, 8))

    for i in range(n):
        f = x.data.cpu().numpy()
        print(f.shape)
        ax = plt.subplot(nr, nc, i + 1)
        ax.axis('off')

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.imshow(f[0, i, ..., s], cmap='jet')
    plt.savefig(os.path.join(path, name), bbox_inches='tight', dpi=300)
    plt.show()


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.LeakyReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid())
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class SENetBottleneck(nn.Module):
    def __init__(self, inplanes, out_planes, cardinality, stride=1, expansion=2,
                 reduction=16):
        super(SENetBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.conv3 = nn.Conv3d(
            out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes * self.expansion)
        self.relu = nn.PReLU()
        self.se = SELayer3D(out_planes * self.expansion, reduction)
        self.stride = stride
        self.down_sample = nn.Conv3d(inplanes, out_planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.conv = nn.Conv3d(inplanes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(out_planes * self.expansion)

    def forward(self, x):
        if self.stride != 1:
            residual = self.down_sample(x)
            residual = self.bn(residual)
        else:
            residual = self.conv(x)
            residual = self.bn(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class SENetDilatedBottleneck(nn.Module):
    def __init__(self, inplanes, out_planes, cardinality, stride=1, expansion=2):
        super(SENetDilatedBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(inplanes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_planes)
        self.conv2 = nn.Conv3d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            dilation=2,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.conv3 = nn.Conv3d(
            out_planes, out_planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_planes * self.expansion)
        self.relu = nn.PReLU()
        self.down_sample = nn.Conv3d(inplanes, out_planes * self.expansion, kernel_size=1, stride=stride, bias=False)
        self.conv = nn.Conv3d(inplanes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm3d(out_planes * self.expansion)
        self.se = SELayer3D(out_planes * self.expansion, reduction=16)
        self.stride = stride

    def forward(self, x):
        if self.stride != 1:
            residual = self.down_sample(x)
            residual = self.bn(residual)
        else:
            residual = self.conv(x)
            residual = self.bn(residual)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class extract_feature(nn.Module):
    def __init__(self, in_channels=1, inplanes=32):
        super(extract_feature, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, inplanes, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inplanes)
        self.relu2 = nn.PReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        return x


class AG(nn.Module):
    def __init__(self, inplanes):
        super(AG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes*2, inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = nn.BatchNorm3d(inplanes)

        self.conv2 = nn.Conv3d(inplanes*2, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(inplanes)

    def forward(self, input1, input2):
        x = torch.cat((input1, input2), dim=1)
        att1 = self.conv1(x)
        att1 = self.bn1(att1)
        att1 = torch.sigmoid(att1)
        w1_map = att1*input1
        att2 = self.conv2(x)
        att2 = self.bn2(att2)
        att2 = torch.sigmoid(att2)
        w2_map = att2*input2
        w_map = w1_map + w2_map
        return w_map


class AG_SE_ResNeXt50(nn.Module):
    def __init__(self, block, layers, inplanes=32, out_planes=[32, 64, 128, 256, 512], cardinality=16, num_classes=2):
        self.inplanes = inplanes
        self.expansion = 2
        super(AG_SE_ResNeXt50, self).__init__()
        self.extract = extract_feature(in_channels=1, inplanes=inplanes)
        self.roi_extractor = extract_feature(in_channels=5, inplanes=inplanes)

        self.AG0 = AG(inplanes=inplanes)
        self.AG1 = AG(inplanes=out_planes[1])
        self.AG2 = AG(inplanes=out_planes[2])
        self.AG3 = AG(inplanes=out_planes[3])

        self.layer1 = self._make_layer(block, out_planes[0], layers[0],  cardinality)
        self.roi_layer1 = nn.Conv3d(inplanes, out_planes[1], kernel_size=3, stride=1, padding=1)

        self.layer2 = self._make_layer(block, out_planes[1], layers[1],  cardinality, stride=2)
        self.roi_layer2 = nn.Conv3d(out_planes[1], out_planes[2], kernel_size=3, stride=2, padding=1)

        self.layer3 = self._make_layer(block, out_planes[2], layers[2], cardinality, stride=2)
        self.roi_layer3 = nn.Conv3d(out_planes[2], out_planes[3], kernel_size=3, stride=2, padding=1)

        self.layer4 = self._make_layer(SENetDilatedBottleneck, out_planes[3], layers[3], cardinality, stride=1)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(out_planes[3] * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_planes, blocks, cardinality, stride=1):
        layers = []
        list_strdies = [stride] + [1]*(blocks-1)
        for s in list_strdies:
            layers.append(block(self.inplanes, out_planes, cardinality, stride=s, expansion=self.expansion))
            self.inplanes = out_planes * self.expansion
        return nn.Sequential(*layers)

    def forward(self, input, roi):
        x = input[:, 0, ...]
        x = self.extract(x)
        roi_feature = self.roi_extractor(roi[:, 1:, ...])
        x = self.AG0(x, roi_feature)
        # show_feature(x, n=16, s=4, nr=4, nc=4,  name='1.jpg')

        x = self.layer1(x)
        roi_1 = self.roi_layer1(roi_feature)
        x = self.AG1(x, roi_1)

        x = self.layer2(x)
        roi_2 = self.roi_layer2(roi_1)
        x = self.AG2(x, roi_2)
        # show_feature(x, n=16, s=2, nr=4, nc=4, name='2.jpg')

        x = self.layer3(x)
        roi_3 = self.roi_layer3(roi_2)
        x = self.AG3(x, roi_3)
        # show_feature(x, n=16, s=1, nr=4, nc=4,  name='3.jpg')

        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def AG_SE_ResNeXt50_model(**kwargs):
    """Constructs a SENet3D-50 model."""
    model = AG_SE_ResNeXt50(SENetBottleneck, [3, 4, 6, 3], inplanes=16, out_planes=[16, 32, 64, 128, 256, 512], cardinality=8, num_classes=2)
    return model


