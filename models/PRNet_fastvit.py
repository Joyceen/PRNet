import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from thop import profile
from torch import Tensor
from typing import List

def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )


class PRNet(nn.Module):
    def __init__(self, backbone_name='fastvit_t8', pretrained=True):
        super(PRNet, self).__init__()

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        C = {'fastvit_t8': [48, 96, 192, 384],
             'fastvit_t12': [64, 128, 256, 512],
             'efficientformer_l1': [48, 96, 192, 384]}

        # detect output channels automatically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 384, 384)
            feats = self.backbone(dummy)
            c1, c2, c3, c4 = [f.shape[1] for f in feats]

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.MAM_1 = CoordAtt(c4, c4)
        self.MAM_2 = CoordAtt(c3, c3)
        self.MAM_3 = CoordAtt(c2, c2)
        self.MAM_4 = CoordAtt(c1, c1)

        self.PCM1 = MLPBlock(dim=64)
        self.PCM2 = MLPBlock(dim=64)
        self.PCM3 = MLPBlock(dim=64)
        self.PCM4 = MLPBlock(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 =  nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, bias=False),
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.dwc3 = conv3x3_bn_relu(c2, c2 // 2, k=1, s=1, p=0)
        self.dwc2 = conv3x3_bn_relu(c3, c3 // 2, k=1, s=1, p=0)
        self.dwc1 = conv3x3_bn_relu(c4, c4 // 2, k=1, s=1, p=0)
        self.dwcon_2 = conv3x3_bn_relu(c3, c3)
        self.dwcon_3 = conv3x3_bn_relu(c2, c2)
        self.dwcon_4 = conv3x3_bn_relu(c1, c1)

        self.xf_11 = nn.Conv2d(c4, 64, kernel_size=1)
        self.xf_12 = nn.Conv2d(c4, 64, kernel_size=1)
        self.xf_13 = nn.Conv2d(c4, 64, kernel_size=1)
        self.xf_14 = nn.Conv2d(c4, 64, kernel_size=1)
        self.xf_22 = nn.Conv2d(c3, 64, kernel_size=1)
        self.xf_23 = nn.Conv2d(c3, 64, kernel_size=1)
        self.xf_24 = nn.Conv2d(c3, 64, kernel_size=1)
        self.xf_33 = nn.Conv2d(c2, 64, kernel_size=1)
        self.xf_34 = nn.Conv2d(c2, 64, kernel_size=1)
        self.xf_44 = nn.Conv2d(c1, 64, kernel_size=1)

        # store for forward
        self._c = {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}


    def forward(self, x):
        rgb_list = self.backbone(x)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]

        c1, c2, c3, c4 = self._c['c1'], self._c['c2'], self._c['c3'], self._c['c4']

        _, _, h1, w1 = r1.shape
        _, _, h2, w2 = r2.shape
        _, _, h3, w3 = r3.shape
        _, _, h4, w4 = r4.shape

        xf_1 = self.MAM_1(r1)

        r1_up = F.interpolate(self.dwc1(xf_1), size=(h2, w2), mode='bilinear')
        r1_up, _ = torch.split(r1_up, [c3 // 2, c3 // 2], dim=1)
        r2, _ = torch.split(r2, [c3 // 2, c3 // 2], dim=1)
        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        xf_2 = self.MAM_2(r2_con)

        r2_up = F.interpolate(self.dwc2(xf_2), size=(h3, w3), mode='bilinear')
        r2_up, _ = torch.split(r2_up, [c2 // 2, c2 // 2], dim=1)
        r3, _ = torch.split(r3, [c2 // 2, c2 // 2], dim=1)
        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        xf_3 = self.MAM_3(r3_con)

        r3_up = F.interpolate(self.dwc3(xf_3), size=(h4, w4), mode='bilinear')
        r3_up, _ = torch.split(r3_up, [c1 // 2, c1 // 2], dim=1)
        r4, _ = torch.split(r4, [c1 // 2, c1 // 2], dim=1)
        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        xf_4 = self.MAM_4(r4_con)

        xf_11 = self.xf_11(xf_1)
        xf_12 = F.interpolate(self.xf_12(xf_1), size=(h2, w2), mode='bilinear')
        xf_13 = F.interpolate(self.xf_13(xf_1), size=(h3, w3), mode='bilinear')
        xf_14 = F.interpolate(self.xf_14(xf_1), size=(h4, w4), mode='bilinear')
        xf_22 = self.xf_22(xf_2)
        xf_23 = F.interpolate(self.xf_23(xf_2), size=(h3, w3), mode='bilinear')
        xf_24 = F.interpolate(self.xf_24(xf_2), size=(h4, w4), mode='bilinear')
        xf_33 = self.xf_33(xf_3)
        xf_34 = F.interpolate(self.xf_34(xf_3), size=(h4, w4), mode='bilinear')
        xf_44 = self.xf_44(xf_4)

        xf_4 = xf_44 + xf_14 + xf_24 + xf_34
        xf_3 = xf_33 + xf_23 + xf_13
        xf_2 = xf_22 + xf_12
        xf_1 = xf_11

        xf_1 = self.PCM1(xf_1)

        xc_1_2 = torch.cat((xf_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.PCM2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.PCM3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.PCM4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(xf_1), size=384, mode='bilinear')
        return y1, y2, y3, y4


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb):
        x = rgb

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)

        return out


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div=2,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat'
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        x = self.up(x)
        return x



if __name__ == '__main__':
    model = PRNet(backbone_name='fastvit_t8', pretrained=False)
    x = torch.randn(1, 3, 384, 384)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f'Output {i}: {o.shape}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total/1e6:.2f}M')
