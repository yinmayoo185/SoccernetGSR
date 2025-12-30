import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import EfficientNet_V2_S_Weights

from model.non_local_embedded_gaussian import NONLocalBlock2D


def weights_init(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def weights_init2(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            # nn.GroupNorm(32, out_ch),
        )

    def forward(self, input):
        x1 = self.conv(input)
        x2 = self.channel_conv(input)
        x = x1 + x2
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi


class EfficientNet(nn.Module):
    def __init__(self, n_channels):
        super(EfficientNet, self).__init__()

        self.efficient_model = torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

        self.layer1 = self.efficient_model.features[0:2]
        self.layer2 = self.efficient_model.features[2:3]
        self.layer3 = self.efficient_model.features[3:4]
        self.layer4 = self.efficient_model.features[4:6]
        self.layer5 = self.efficient_model.features[6:9]

        self.nl_1 = NONLocalBlock2D(in_channels=64)
        self.nl_2 = NONLocalBlock2D(in_channels=160)

    def forward(self, x):
        # print(x.shape)
        x1 = self.layer1(x)
        # print("layer 1 : ", x1.shape)
        x2 = self.layer2(x1)
        # print("layer 2 : ", x2.shape)
        x3 = self.layer3(x2)
        x3 = self.nl_1(x3)
        # print("layer 3 : ", x3.shape)
        x4 = self.layer4(x3)
        x4 = self.nl_2(x4)
        x5 = self.layer5(x4)
        # print("layer 5 : ", x5.shape)

        return x1, x2, x3, x4, x5


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, size):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(size=size, mode="bilinear", align_corners=True),
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch_in, ch_out,
                      kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

        # self.up.apply(weights_init)

    def forward(self, x):
        x = self.up(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out,
                      kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out,
                      kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, out_ch, num_lines=21):
        super(Unet, self).__init__()
        # print("EfficientUnet_git_b6_res")
        self.pretrained_net = EfficientNet(3)
        # self.up6 = nn.ConvTranspose2d(1280, 512, 2, stride=2)
        self.up6 = UpConvBlock(ch_in=1280, ch_out=512, size=(24, 24))
        self.att1 = Attention_block(512, 160, 160)
        self.conv6 = DoubleConv(160 + 512, 512)
        # self.conv6 = ConvBlock(160 + 512, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up7 = UpConvBlock(ch_in=512, ch_out=256, size=(48, 48))
        self.att2 = Attention_block(256, 64, 64)
        self.conv7 = DoubleConv(256 + 64, 256)
        # self.conv7 = ConvBlock(256 + 64, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up8 = UpConvBlock(ch_in=256, ch_out=128, size=(96, 96))
        self.att3 = Attention_block(128, 48, 48)
        self.conv8 = DoubleConv(128 + 48, 128)
        # self.conv8 = ConvBlock(128 + 48, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up9 = UpConvBlock(ch_in=128, ch_out=64, size=(192, 192))
        self.att4 = Attention_block(64, 24, 24)
        self.conv9 = DoubleConv(64 + 24, 64)
        # self.conv9 = ConvBlock(64 + 24, 64)
        self.up10 = UpConvBlock(ch_in=64, ch_out=64, size=(384, 384))
        # self.up10 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.att5 = Attention_block(64, 3, 1)
        self.conv10 = DoubleConv(67, 64)
        # self.conv10 = ConvBlock(67, 64)
        self.conv11 = nn.Conv2d(64, out_ch, kernel_size=1)

        # New line heatmap head
        self.line_head = nn.Conv2d(64, num_lines, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Weights initialize
        self.conv6.apply(weights_init)
        self.conv7.apply(weights_init)
        self.conv8.apply(weights_init)
        self.conv9.apply(weights_init)
        self.conv10.apply(weights_init)
        self.conv11.apply(weights_init)
        self.line_head.apply(weights_init)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.pretrained_net(x)

        up_6 = self.up6(x5)
        # print("up_6 : ", up_6.shape)
        x4 = self.att1(up_6, x4)
        # print("x4 : ", x4.shape)
        merge6 = torch.cat([up_6, x4], dim=1)
        # print("merge6 : ", merge6.shape)
        c6 = self.conv6(merge6)
        # print("c6 : ", c6.shape)
        up_7 = self.up7(c6)
        # print("up_7 : ", up_7.shape)
        x3 = self.att2(up_7, x3)
        # print("x3 : ", x3.shape)
        merge7 = torch.cat([up_7, x3], dim=1)
        # print("merge7 : ", merge7.shape)
        c7 = self.conv7(merge7)
        # print("c7 : ", c7.shape)
        up_8 = self.up8(c7)
        # print("up_8 : ", up_8.shape)
        x2 = self.att3(up_8, x2)
        # print("x2 : ", x2.shape)
        merge8 = torch.cat([up_8, x2], dim=1)
        # print("merge8 : ", merge8.shape)
        c8 = self.conv8(merge8)
        # print("c8 : ", c8.shape)
        up_9 = self.up9(c8)
        # print("up_9 : ", up_9.shape)
        x1 = self.att4(up_9, x1)
        # print("x1 : ", x1.shape)
        merge9 = torch.cat([up_9, x1], dim=1)
        # print("merge9 : ", merge9.shape)
        c9 = self.conv9(merge9)
        # print("c9 : ", c9.shape)
        up_10 = self.up10(c9)
        # print("up_10 : ", up_10.shape)
        x = self.att5(up_10, x)
        # print("x : ", x.shape)
        merge10 = torch.cat([up_10, x], dim=1)
        # print("merge10 : ", merge10.shape)
        c10 = self.conv10(merge10)
        # print("c10 : ", c10.shape)
        c11 = self.conv11(c10)
        # distance map --------------
        # c11 = c11.squeeze(1)
        # c11 = c11.view(c11.size(0), -1, 2)
        # print("c11 : ", c11.shape)
        # distance map --------------
        out = self.sigmoid(c11)

        # Line heatmap output
        line_output = self.line_head(c10)
        line_out = self.sigmoid(line_output)

        # out = out.unsqueeze(2)  # Add T dimension at index 2
        # line_out = line_out.unsqueeze(2)  # Add T dimension at index 2

        return out, line_out


if __name__ == "__main__":
    model = Unet(out_ch=98, num_lines=21)
    # params = sum(param.numel() for param in model.parameters())condat
    x = torch.randn(1, 3, 384, 384)
    output, line_out = model(x)
    print("Keypoints : ", output.shape)
    print("Lines : ", line_out.shape)

    from ptflops import get_model_complexity_info, flops_counter

    flops, params = get_model_complexity_info(model, input_res=(3, 384, 384), as_strings=True,
                                              print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

    # efficient_model = torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    # # print("efficient_model : ", efficient_model)
    # layer1 = efficient_model.features[0:2]
    # print("layer1 : ", layer1)
    # layer2 = efficient_model.features[2:3]
    # print("layer2 : ", layer2)
    # layer3 = efficient_model.features[3:4]
    # print("layer3 : ", layer3)
    # layer4 = efficient_model.features[4:6]
    # print("layer4 : ", layer4)
    # layer5 = efficient_model.features[6:9]
    # print("layer5 : ", layer5)
