import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, AFF, ASPP, CARAFE, segmentheadCARAFE, iAFF, segmenthead_drop, Muti_AFF, segmenthead_c, DUC, SPASPP, MFACB
import logging
import os
from thop import profile

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False


class DSNet(nn.Module):

    def __init__(self, m=2, n=3, num_classes=19, planes=64, name='s128', augment=True):
        super(DSNet, self).__init__()
        self.augment = augment
        self.name = name
        # I Branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        
        self.layer3 = nn.Sequential(
                MFACB(planes * 2,planes * 2, planes * 4,dilation=[2,2,2]),
                MFACB(planes * 4,planes * 4, planes * 4,dilation=[2,2,2]),
                MFACB(planes * 4,planes * 4, planes * 4,dilation=[3,3,3]),
        )
        
        if 's' in self.name:
            self.layer4 = nn.Sequential(
                MFACB(planes * 4,planes * 4,planes * 8,dilation=[3,3,3]),
                MFACB(planes * 8,planes * 8,planes * 8,dilation=[5,5,5]),
            )
        if 'm' in self.name:
            self.layer4 = nn.Sequential(
                    MFACB(planes * 4,planes * 4,planes * 8,dilation=[3,3,3]),
                    MFACB(planes * 8,planes * 8,planes * 8,dilation=[5,5,5]),
                    MFACB(planes * 8,planes * 8,planes * 8,dilation=[5,5,5]),
            )

        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 4, 1, stride=1, dilation=5)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )

        self.compression5 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 4, kernel_size=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )


        self.layer3_ = self._make_layer(BasicBlock, planes * 2, planes * 4, n)
        self.layer4_ = self._make_layer(BasicBlock, planes * 4, planes * 4, n)
        self.layer5_ = self._make_layer(Bottleneck, planes * 4, planes * 2, 1)



        

        # 融合模块
        self.aff1 = Muti_AFF(channels=planes*4)
        self.aff2 = Muti_AFF(channels=planes*4)
        self.aff3 = Muti_AFF(channels=planes*4)
        

        if self.name == 's128' or self.name == 'm':  
            self.spp = SPASPP(planes*4, planes*4, planes*4)
            self.layer1_a = self._make_layer(BasicBlock, planes, planes, 1)
            self.up8 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 4, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
            )
            self.lastlayer = segmenthead_c(planes*5, planes*4, num_classes)
        
        if self.name == 's64':  
            self.spp = SPASPP(planes*4, planes*4, planes*4)
            self.layer1_a = self._make_layer(BasicBlock, planes, planes, 1)
            self.up8 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(planes * 2, momentum=bn_mom),
            )
            self.lastlayer = segmenthead_c(planes*3, planes*2, num_classes)
            
        if self.name == 's256':  
            self.spp = SPASPP(planes*4, planes*8, planes*8)
            self.layer1_a = self._make_layer(BasicBlock, planes, planes*2, 1)
            self.up8 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 8, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=bn_mom),
            )
            self.lastlayer = segmenthead_c(planes*10, planes*8, num_classes)


        if augment:
            self.seghead_p = segmenthead(planes * 4, planes * 4, num_classes)
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
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
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
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
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


def get_seg_model(cfg, imgnet_pretrained):
    if 's' in cfg.MODEL.NAME:
        if cfg.MODEL.NAME == 'dsnet_head128':
            model = DSNet(m=2, n=2, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, name='s128', augment=True)
        if cfg.MODEL.NAME == 'dsnet_head64':
            model = DSNet(m=2, n=2, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, name='s64', augment=True)
        if cfg.MODEL.NAME == 'dsnet_head256':
            model = DSNet(m=2, n=2, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, name='s256', augment=True)            
    if 'm' in cfg.MODEL.NAME:
        model = DSNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, name='m',augment=True)

    print(model)
    if imgnet_pretrained:
        pretrained_path = '/root/autodl-tmp/DSNet/pretrained_models/imagenet/dhsnet_catnormal_wider_93.pth'
        if not os.path.exists(pretrained_path):
            print(f"Error: File not found at {pretrained_path}")
        pretrained_state = torch.load(pretrained_path, map_location='cpu')['state_dict']
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('使用imagenet预训练权重!!!')           
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        if 'module.' in list(pretrained_state.keys())[0]:
            # 如果包含 'module.' 前缀，去掉它
            pretrained_state = {k[7:]: v for k, v in pretrained_state.items()}
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if
                            (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))
        logging.info('使用imagenet预训练权重!!!')           
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model.load_state_dict(model_dict, strict=False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        print("11111")
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                           (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

    return model


def get_pred_model(name, num_classes):
    if name == 'dsnet_head128':
        model = DSNet(m=2, n=2, num_classes=num_classes, planes=32, name='s128', augment=False)
    if name == 'dsnet_head64':
        model = DSNet(m=2, n=2, num_classes=num_classes, planes=32, name='s64', augment=False)
    if name == 'dsnet_head256':
        model = DSNet(m=2, n=2, num_classes=num_classes, planes=32, name='s256', augment=False)            
    if name == 'm':
        model = DSNet(m=2, n=3, num_classes=num_classes, planes=64, name='m',augment=False)

    return model







