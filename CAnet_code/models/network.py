import torch
import torch.nn as nn
from models.layers.modules import conv_block, UpCat, UnetDsv3, UnetDsv3D
from models.layers.grid_attention_layer import MultiAttentionBlock, MultiAttentionBlock_3D
from models.layers.channel_attention_layer import SE_Conv_Block_3D
from models.layers.scale_attention_layer import scale_atten_convblock, scale_atten_convblock_3D
from models.layers.nonlocal_layer import NONLocalBlock2D, NONLocalBlock3D


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups=8):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn1 = nn.GroupNorm(n_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.gn2 = nn.GroupNorm(n_groups, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.gn1(x))

        x = self.conv2(x)
        x = self.relu2(self.gn2(x))
        return x


class UpCat3D(nn.Module):
    def __init__(self, in_feat, out_feat, is_deconv=True):
        super(UpCat3D, self).__init__()

        if is_deconv:
            self.up = nn.ConvTranspose3d(in_feat, out_feat, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, inputs, outputs):
        # TODO: Upsampling required after deconv?
        outputs = self.up(outputs)
        out = torch.cat([inputs, outputs], dim=1)
        return out


class UNet3D(nn.Module):
    def __init__(self, input_shape, in_channels=2, out_channels=6, init_channels=16, p=0.2):
        super(UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()
        self.dropout = nn.Dropout(p=p)

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1 = BasicBlock(self.in_channels, init_channels)  # 16
        self.ds1 = nn.MaxPool3d(2, stride=2)

        self.conv2 = BasicBlock(init_channels, init_channels*2)
        self.ds2 = nn.MaxPool3d(2, stride=2)

        self.conv3 = BasicBlock(init_channels*2, init_channels*4)
        self.ds3 = nn.MaxPool3d(2, stride=2)

        self.conv4 = BasicBlock(init_channels*4, init_channels*8)
        self.ds4 = nn.MaxPool3d(2, stride=2)

        self.conv5 = BasicBlock(init_channels*8, init_channels*16)

    def make_decoder(self):
        init_channels = self.init_channels
        self.up4= UpCat3D(in_feat=init_channels*16, out_feat=init_channels*8, is_deconv=True)  # mode='bilinear'
        self.up4conv = BasicBlock(init_channels * 16, init_channels * 8)

        self.up3 = UpCat3D(in_feat=init_channels * 8, out_feat=init_channels * 4, is_deconv=True)  # mode='bilinear'
        self.up3conv = BasicBlock(init_channels * 8, init_channels * 4)

        self.up2 = UpCat3D(in_feat=init_channels * 4, out_feat=init_channels * 2, is_deconv=True)  # mode='bilinear'
        self.up2conv = BasicBlock(init_channels * 4, init_channels * 2)

        self.up1 = UpCat3D(in_feat=init_channels * 2, out_feat=init_channels , is_deconv=True)  # mode='bilinear'
        self.up1conv = BasicBlock(init_channels * 2, init_channels)

        self.out = nn.Conv3d(init_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        c1d = self.ds1(c1)

        c2 = self.conv2(c1d)
        c2d = self.ds2(c2)

        c3 = self.conv3(c2d)
        c3d = self.ds3(c3)

        c4 = self.conv4(c3d)
        c4d = self.ds4(c4)

        c5 = self.conv5(c4d)

        u4 = self.up4(c4,c5)
        u4c = self.up4conv(u4)

        u3 = self.up3(c3, u4c)
        u3c = self.up3conv(u3)

        u2 = self.up2(c2, u3c)
        u2c = self.up2conv(u2)

        u1 = self.up1(c1, u2c)
        u1c = self.up1conv(u1)

        uout = self.out(u1c)
        return uout


class CA_UNet3D(nn.Module):
    def __init__(self, input_shape, in_channels=2, out_channels=6, init_channels=16):
        super(CA_UNet3D, self).__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_channels = init_channels
        self.make_encoder()
        self.make_decoder()

    def make_encoder(self):
        init_channels = self.init_channels
        self.conv1 = BasicBlock(self.in_channels, init_channels)  # 16
        self.ds1 = nn.MaxPool3d(2, stride=2)

        self.conv2 = BasicBlock(init_channels, init_channels*2)
        self.ds2 = nn.MaxPool3d(2, stride=2)

        self.conv3 = BasicBlock(init_channels*2, init_channels*4)
        self.ds3 = nn.MaxPool3d(2, stride=2)

        self.conv4 = BasicBlock(init_channels*4, init_channels*8)
        self.ds4 = nn.MaxPool3d(2, stride=2)

        self.conv5 = BasicBlock(init_channels*8, init_channels*16)

    def make_decoder(self):
        init_channels = self.init_channels

        self.SA1 = NONLocalBlock3D(in_channels=init_channels*16, inter_channels=init_channels*16 // 4)

        self.SA2 = MultiAttentionBlock_3D(in_size=init_channels*4, gate_size=init_channels*8, inter_size=init_channels*4,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1, 1, 1))
        self.SA3 = MultiAttentionBlock_3D(in_size=init_channels*2, gate_size=init_channels*4, inter_size=init_channels*2,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1, 1, 1))
        self.SA4 = MultiAttentionBlock_3D(in_size=init_channels, gate_size=init_channels*2, inter_size=init_channels,
                                                   nonlocal_mode='concatenation', sub_sample_factor=(1, 1, 1))

        self.CA1 = SE_Conv_Block_3D(init_channels*16, init_channels*8, drop_out=True)
        self.CA2 = SE_Conv_Block_3D(init_channels*8, init_channels*4, drop_out=True)
        self.CA3 = SE_Conv_Block_3D(init_channels*4, init_channels*2, drop_out=True)
        self.CA4 = SE_Conv_Block_3D(init_channels*2, init_channels, drop_out=True)

        self.up4= UpCat3D(in_feat=init_channels*16, out_feat=init_channels*8, is_deconv=True)
        self.up3 = UpCat3D(in_feat=init_channels * 8, out_feat=init_channels * 4, is_deconv=True)
        self.up2 = UpCat3D(in_feat=init_channels * 4, out_feat=init_channels * 2, is_deconv=True)
        self.up1 = UpCat3D(in_feat=init_channels * 2, out_feat=init_channels, is_deconv=True)


        self.dsv4 = UnetDsv3D(in_size=init_channels * 8, out_size=self.out_channels, scale_factor=8)
        self.dsv3 = UnetDsv3D(in_size=init_channels * 4, out_size=self.out_channels, scale_factor=4)
        self.dsv2 = UnetDsv3D(in_size=init_channels * 2, out_size=self.out_channels, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=init_channels, out_channels=self.out_channels, kernel_size=1)

        self.scale_att = scale_atten_convblock(in_size= (4*self.out_channels), out_size=self.out_channels)
        self.final = nn.Conv3d(self.out_channels, self.out_channels, (1, 1, 1))

    def forward(self, x):
        c1 = self.conv1(x)
        c1d = self.ds1(c1)

        c2 = self.conv2(c1d)
        c2d = self.ds2(c2)

        c3 = self.conv3(c2d)
        c3d = self.ds3(c3)

        c4 = self.conv4(c3d)
        c4d = self.ds4(c4)

        center = self.conv5(c4d)

        up4 = self.up4(c4, center)
        s1 = self.SA1(up4)
        up4, c_weight1 = self.CA1(s1)

        g_conv3, att3 = self.SA2(c3, up4)
        up3 = self.up3(g_conv3, up4)
        up3, att_weight3 = self.CA2(up3)

        g_conv2, att2 = self.SA3(c2, up3)
        up2 = self.up2(g_conv2, up3)
        up2, att_weight2 = self.CA3(up2)

        g_conv1, att1 = self.SA4(c1, up2)
        up1 = self.up1(g_conv1, up2)
        up1, att_weight4 = self.CA4(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)
        out = self.final(out)
        return out