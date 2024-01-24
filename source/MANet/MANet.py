import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityMap(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1, BN=True):
        super(IdentityMap, self).__init__()

        if BN is True:
            self.IdenMap = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.IdenMap = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )
    def forward(self, x):
        idenMap = self.IdenMap(x)

        return idenMap


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels//reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_max =  F.max_pool2d(x, (x.shape[-2:][0], x.shape[-2:][1]))
        x_avg = F.avg_pool2d(x, (x.shape[-2:][0], x.shape[-2:][1]))

        x_max_att = self.mlp(x_max)
        x_avg_att = self.mlp(x_avg)

        channel_att = x_max_att + x_avg_att
        channel_att = self.sigmoid(channel_att).unsqueeze(-1).unsqueeze(-1)
        refined_features = x * channel_att

        return refined_features


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_maxPool = torch.max(x, dim=1, keepdim=True)[0]
        x_avgPool = torch.mean(x, dim=1, keepdim=True)

        features = torch.cat([x_maxPool, x_avgPool], dim=1)
        features = self.conv(features)
        feature_att = self.sigmoid(features)
        refined_features = x * feature_att

        return refined_features


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()

        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        ch_att = self.channel_att(x)
        sp_att = self.spatial_att(ch_att)

        return sp_att


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mid_channels=None):
        super(DoubleConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )


    def forward(self, x):
        return self.double_conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Upsample, self).__init__()

        if bilinear is True:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        out = self.up(x)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.BatchNorm2d
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=2, padding=0)
            # nn.BatchNorm2d
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

        self.resampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, g, x):
        w_g = self.W_g(g)
        w_x = self.W_x(x)

        s1 = self.relu(w_g + w_x)
        s2 = self.psi(s1)

        alpha = self.resampler(s2)

        return x * alpha



class InitialConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(InitialConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels // 2

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.identityMap = IdentityMap(in_channels, out_channels, stride=1, BN=True)

    def forward(self, x):
        out = self.double_conv(x) + self.identityMap(x)

        return out


class EncoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(EncoderConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = DoubleConv(in_channels, out_channels, stride=2, mid_channels=mid_channels)
        self.identityMap = IdentityMap(in_channels, out_channels, stride=2, BN=True)

    def forward(self, x):
        out = self.double_conv(x) + self.identityMap(x)

        return out


class BridgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(BridgeConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.encoder_conv = EncoderConv(in_channels, out_channels, mid_channels=mid_channels)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, stride=1, padding=0),
        # )

    def forward(self, x):
        out = self.encoder_conv(x)
        # out = self.conv(x1)

        return out


class DecoderConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DecoderConv, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        self.decoder_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # self.identityMap = IdentityMap(in_channels, out_channels, stride=1, BN=True)

    def forward(self, skip_att, x_up):
        concat = torch.cat([skip_att, x_up], dim=1)
        out = self.decoder_conv(concat)

        return out



class MANet_64(nn.Module):
    # Multi Attention Network (MANet)
    def __init__(self, in_channels, n_classes, init_features=64):
        super(MANet_64, self).__init__()
        features = init_features

        # Encoding
        self.initial_conv1 = InitialConv(in_channels, features)
        self.ch_att1 = ChannelAttention(features)

        self.encoder_conv2 = EncoderConv(features, features * 2)
        self.ch_att2 = ChannelAttention(features * 2)

        self.encoder_conv3 = EncoderConv(features * 2, features * 4)
        self.ch_att3 = ChannelAttention(features * 4)

        # Bridge
        self.bridge_conv =  BridgeConv(features * 4, features * 8)
        self.cbam = CBAM(features * 8)

        # Decoding
        self.skip_att1 = AttentionBlock(features * 8, features * 4, features * 2)
        self.sp_att1 = SpatialAttention()
        self.up1 = Upsample(features * 8, features * 4, bilinear=False)
        self.decoder_conv1 = DecoderConv(features * 8, features * 2)

        self.skip_att2 = AttentionBlock(features * 2, features * 2, features)
        self.sp_att2 = SpatialAttention()
        self.up2 = Upsample(features * 2, features * 2, bilinear=False)
        self.decoder_conv2 = DecoderConv(features * 4, features)

        self.skip_att3 = AttentionBlock(features, features, features // 2)
        self.sp_att3 = SpatialAttention()
        self.up3 = Upsample(features, features, bilinear=False)
        self.decoder_conv3 = DecoderConv(features * 2, features // 2, mid_channels=features)

        # Final Layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(features // 2, n_classes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        x1 = self.initial_conv1(x)
        ch_at1 = self.ch_att1(x1)

        x2 = self.encoder_conv2(ch_at1)
        ch_at2 =  self.ch_att2(x2)

        x3 = self.encoder_conv3(ch_at2)
        ch_at3 = self.ch_att3(x3)

        # Bridge
        bridge = self.bridge_conv(ch_at3)
        cbam = self.cbam(bridge)

        # Decoding
        skip_at = self.skip_att1(bridge, x3)
        sp_at =self.sp_att1(cbam)
        x_up = self.up1(sp_at)
        deco = self.decoder_conv1(skip_at, x_up)

        skip_at = self.skip_att2(deco, x2)
        sp_at = self.sp_att2(deco)
        x_up = self.up2(sp_at)
        deco = self.decoder_conv2(skip_at, x_up)

        skip_at = self.skip_att3(deco, x1)
        sp_at = self.sp_att3(deco)
        x_up = self.up3(sp_at)
        deco = self.decoder_conv3(skip_at, x_up)

        # Final Layer
        out = self.final_layer(deco)

        return out




class MANet(nn.Module):
    # Multi Attention Network (MANet)
    def __init__(self, in_channels, n_classes, init_features=68):
        super(MANet, self).__init__()
        features = init_features

        # Encoding
        self.initial_conv1 = InitialConv(in_channels, features)
        self.ch_att1 = ChannelAttention(features)

        self.encoder_conv2 = EncoderConv(features, features * 2)
        self.ch_att2 = ChannelAttention(features * 2)

        self.encoder_conv3 = EncoderConv(features * 2, features * 4)
        self.ch_att3 = ChannelAttention(features * 4)

        # Bridge
        self.bridge_conv =  BridgeConv(features * 4, features * 8)
        self.cbam = CBAM(features * 8)

        # Decoding
        self.skip_att1 = AttentionBlock(features * 8, features * 4, features * 2)
        self.sp_att1 = SpatialAttention()
        self.up1 = Upsample(features * 8, features * 4, bilinear=False)
        self.decoder_conv1 = DecoderConv(features * 8, features * 2)

        self.skip_att2 = AttentionBlock(features * 2, features * 2, features)
        self.sp_att2 = SpatialAttention()
        self.up2 = Upsample(features * 2, features * 2, bilinear=False)
        self.decoder_conv2 = DecoderConv(features * 4, features)

        self.skip_att3 = AttentionBlock(features, features, features // 2)
        self.sp_att3 = SpatialAttention()
        self.up3 = Upsample(features, features, bilinear=False)
        self.decoder_conv3 = DecoderConv(features * 2, features // 2, mid_channels=features)

        # Final Layer
        self.final_layer = nn.Sequential(
            nn.Conv2d(features // 2, n_classes, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoding
        x1 = self.initial_conv1(x)
        ch_at1 = self.ch_att1(x1)

        x2 = self.encoder_conv2(ch_at1)
        ch_at2 =  self.ch_att2(x2)

        x3 = self.encoder_conv3(ch_at2)
        ch_at3 = self.ch_att3(x3)

        # Bridge
        bridge = self.bridge_conv(ch_at3)
        cbam = self.cbam(bridge)

        # Decoding
        skip_at = self.skip_att1(bridge, x3)
        sp_at =self.sp_att1(cbam)
        x_up = self.up1(sp_at)
        deco = self.decoder_conv1(skip_at, x_up)

        skip_at = self.skip_att2(deco, x2)
        sp_at = self.sp_att2(deco)
        x_up = self.up2(sp_at)
        deco = self.decoder_conv2(skip_at, x_up)

        skip_at = self.skip_att3(deco, x1)
        sp_at = self.sp_att3(deco)
        x_up = self.up3(sp_at)
        deco = self.decoder_conv3(skip_at, x_up)

        # Final Layer
        out = self.final_layer(deco)

        return out


if __name__ == '__main__':
    # net = MANet(in_channels=3, n_classes=3)
    # print(net)

    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        net = MANet(in_channels=3, n_classes=3)
        macs, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True,
                                                 print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))