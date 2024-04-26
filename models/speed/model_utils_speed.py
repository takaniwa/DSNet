# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = True


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation==1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation,  padding=dilation, bias=False)
#         self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out
    
class Conv1X1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=1, stride=1, dilation=1):
        super(Conv1X1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
#         self.bn = BatchNorm2d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return out

# 一个快速聚拢感受野的方法，改编自STDC
class MFACB(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2,2,2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
# #             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes *3, out_planes, kernel_size=1, padding=0, bias=False ),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        # out1 = self.conv_list[0](x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1    
    
class SPASPP(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[6,12,18,24]):
        super(SPASPP, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[3]))
        self.pooling = ASPPPooling(in_planes, inter_planes)
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes*5, out_planes, kernel_size=1, padding=0, bias=False ),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True)
        )
        self.process3 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
#             BatchNorm2d(out_planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
#             nn.Dropout(0.2)
        )
    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        out2 = self.pooling(x)
        # out1 = self.conv_list[0](x)
        for idx in range(4):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        out = torch.cat((out,out2), dim=1)
        return self.process3(self.process2(out) + out1)
    
    

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False,dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
#         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=dilation, dilation=dilation, bias=False, )
#         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
#         out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
#         out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
# #         self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
# #         self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
# #         self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
# #         out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
# #         out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
# #         out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmentheadCARAFE(nn.Module):

    def __init__(self, interplanes, outplanes, scale_factor=8, cp_rate=4):
        super(segmentheadCARAFE, self).__init__()
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.carafe = CARAFE(inC=interplanes, outC=outplanes, cp_rate=cp_rate, up_factor=scale_factor)
        self.scale_factor = scale_factor

    def forward(self, x):
        out = self.carafe(self.relu(self.bn2(x)))

        return out

class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
#         self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(x))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
    
class segmenthead_c(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_c, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(x))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out
class segmenthead_drop(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead_drop, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)

        self.drop = nn.Dropout(0.5)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        x = self.drop(self.relu(self.bn2(x)))
        out = self.conv2(x)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out

class segmenthead_l(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
        self.bn3 = BatchNorm2d(outplanes, momentum=bn_mom)

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        out = self.bn3(out)
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=algc)

        return out




class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''

    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class Muti_AFF(nn.Module):
    '''
    多特征融合 AFF, 一个像素级尺度，多个语义级尺度
    '''
    def __init__(self, channels=64, r=4):
        super(Muti_AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
        )

        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels)
        )

        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels)
        )

        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度

        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)

        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')

        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class iAFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)
        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

    # 整个 ASPP 架构


    

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18, 24], out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        ]
        super(DASPPConv, self).__init__(*modules)    
    
# 这是一个改变的ASPP，旨在增强特征表示，让各分支的语义交互

# 池化 -> 1*1 卷积 -> 上采样
class DASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
class DASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6, 12, 18, 24], out_channels=256):
        super(DASPP, self).__init__()
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
        ))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(DASPPConv(in_channels, out_channels, rate))

        # 池化
        # modules.append(ASPPPooling(in_channels, out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.convs = nn.ModuleList(modules)
        self.process1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),        
        )
        self.process2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),        
        )
        self.process3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),        
        )
        self.process4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels, momentum=bn_mom),        
        )
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        x1 = self.convs[0](x)
        x2 = self.convs[1](x)
        x3 = self.convs[2](x)
        x4 = self.convs[3](x)
        x5 = self.convs[4](x)
        x2 = self.process1(x1 + x2)
        x3 = self.process2(x2 + x3)
        x4 = self.process3(x3 + x4)
        x5 = self.process4(x4 + x5)

        x1 = torch.cat((x1,x2,x3,x4,x5), dim=1)
        return self.project(self.relu(x1))
    

    
    
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2, cp_rate=4, ifBN=False):
        super(CARAFE, self).__init__()
        self.ifBN = ifBN
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // cp_rate, 1)
        self.encoder = nn.Conv2d(inC // cp_rate, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)
        self.bn = nn.BatchNorm2d(outC)
    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor) # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W, self.up_factor ** 2) # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0) # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1) # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1) # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1) # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        if self.ifBN:
            out_tensor = self.bn(out_tensor)
        return out_tensor

if __name__ == '__main__':

    
    # x = torch.rand(4, 256, 64, 64).cuda()
    #
    #
    # aspp = ASPP(in_channels=256).cuda()
    #
    # x = aspp(x)
    # print(x.shape)
    data = torch.rand(4, 160, 10, 10)
    carafe = CARAFE(160, 19, up_factor=8)
    print(carafe(data).size())
