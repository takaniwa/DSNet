import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model_utils_speed import BasicBlock, Bottleneck, segmenthead, AFF, ASPP, CARAFE, segmentheadCARAFE, iAFF, segmenthead_drop, Muti_AFF, segmenthead_c, SPASPP, MFACB
import logging
import os
from thop import profile
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class DSNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(DSNet, self).__init__()
        self.augment = augment

        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
#             BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
#             BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer1_a = self._make_layer(BasicBlock, planes, planes, 2)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        
        self.layer3 = nn.Sequential(
                MFACB(planes * 2,planes * 2, planes * 4,dilation=[2,2,2]),
                MFACB(planes * 4,planes * 4, planes * 4,dilation=[2,2,2]),
                MFACB(planes * 4,planes * 4, planes * 4,dilation=[3,3,3]),
        )
        
        self.layer4 = nn.Sequential(
                MFACB(planes * 4,planes * 4,planes * 8,dilation=[3,3,3]),
                MFACB(planes * 8,planes * 8,planes * 8,dilation=[5,5,5]),
        )

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 4, 1, stride=1, dilation=5)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False),
#             BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
#             BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.compression5 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
#             BatchNorm2d(planes * 4, momentum=bn_mom),
        )


        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 4, m)
        self.layer4_ = self._make_layer(BasicBlock, planes * 4, planes * 4, m)
        self.layer5_ = self._make_layer(Bottleneck, planes * 4, planes * 2, 1)



        

        # 融合模块
        self.aff1 = Muti_AFF(channels=planes*4)
        self.aff2 = Muti_AFF(channels=planes*4)
        self.aff3 = Muti_AFF(channels=planes*4)
        self.spp = SPASPP(planes*4, planes*4, planes*4)

        self.lastlayer = segmenthead_c(planes*3, planes*2, num_classes)
        self.up8 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1),
#             BatchNorm2d(planes * 2, momentum=bn_mom),
        )

        if augment:
            self.seghead_p = segmenthead(planes * 4, head_planes, num_classes)
            self.seghead_d = segmenthead(planes * 4, planes, num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation =1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True, dilation=dilation))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

    def forward(self, x):

        width_output = x.shape[-1]
        height_output = x.shape[-2]
        x = self.conv1(x)
        x = self.layer1(x)
        x_a = self.layer1_a(x)
        x = self.relu(self.layer2(self.relu(x)))  


        x_ = self.layer3_(x)  
        x = self.layer3(x)   
        x_ = self.aff1(x_, self.compression3(x))  

        if self.augment:
            temp_1 = x_    

        x = self.layer4(x)  

        x_ = self.layer4_(self.relu(x_))  

        x_ = self.aff2(x_, self.compression4(x))  
        if self.augment:
            temp_2 = x_

        x_ = self.layer5_(self.relu(x_)) 
        x = self.layer5(x)
        x = self.relu(x)  

        x_ = self.aff3(x_, self.compression5(x))
        x_ = self.relu(x_)
        x_ = self.spp(x_)
        x_ = self.up8(x_)
        x_ = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=False)
        x_ = torch.cat((x_,x_a),dim=1)
        x_ = self.lastlayer(x_)
        x_ = F.interpolate(x_, size=[height_output, width_output], mode='bilinear', align_corners=False)


        if self.augment:
            x_extra_p = self.seghead_p(temp_1)
            x_extra_d = self.seghead_d(temp_2)
            x_extra_1 = F.interpolate(x_extra_p, size=[height_output, width_output], mode='bilinear', align_corners=False)
            x_extra_2 = F.interpolate(x_extra_d, size=[height_output, width_output], mode='bilinear', align_corners=False)


            return [x_extra_1, x_, x_extra_2]
        else:
            return x_


def get_pred_model(name, num_classes):
    if 's' in name:
        model = DSNet(m=2, n=2, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)

    return model


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    model = get_pred_model(name='ds_s', num_classes=19)
    model.eval()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    iterations = None
    # print(model)
    input = torch.randn(1, 3, 1024, 2048).cuda()
    # print(model)
    flops, params = profile(model.to(device), inputs=(input,))

    print("参数量：", params)
    print("FLOPS：", flops/(1024*1024*1024))

    # 在前向传播之前
    start_memory = torch.cuda.memory_allocated()

    # 前向传播
    output = model(input)

    # 在前向传播之后
    end_memory = torch.cuda.memory_allocated()
    memory_used = end_memory - start_memory

    print(f"GPU Memory Used: {memory_used / 1024 ** 2} MB")


    tt = []


    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()

        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_end = time.time()
        elapsed_time = t_end- t_start
        ms = elapsed_time/iterations
        latency = elapsed_time / iterations * 1000

        print(iterations)

    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)
    print(ms)




