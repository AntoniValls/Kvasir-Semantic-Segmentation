# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# UNET++
class Squeeze_Excite(nn.Module):
    
    def __init__(self,channel,reduction):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class VGGBlock(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.SE = Squeeze_Excite(out_channels,8) # not sure for what this work? -> Stabilitzation?
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.SE(out)
        
        return(out)


class NestedUNet(nn.Module):
    
    def __init__(self, config, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        #nb_filter = [6, 12, 24, 48, 96]

        self.num_classes = config['num_classes']
        self.input_channels= config['input_channels']
        self.deep_supervision = config['deep_supervision']

        self.pool = nn.MaxPool2d(2, 2) # down-sampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # up-sampling

        self.conv0_0 = VGGBlock(self.input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        #self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        #self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        #self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        #self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        #self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
            #self.final4 = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.num_classes, kernel_size=1)
        
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        #x4_0 = self.conv4_0(self.pool(x3_0))
        #x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        #x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        #x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            #output4 = self.final4(x0_4)
            #return [output1, output2, output3, output4]
            return [output1, output2, output3]

        else:
            output = self.final(x0_3) #x0_4
            return output

def output_block():
    Layer = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1,1)),
                 nn.Sigmoid())
    return Layer


# Attention Net

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class AttU_Net(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        self.num_classes = config['num_classes']
        self.input_channels= config['input_channels']
        
        nb_filter = [32, 64, 128, 256, 512]
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=self.input_channels,ch_out=nb_filter[0])
        self.Conv2 = conv_block(nb_filter[0],nb_filter[1])
        self.Conv3 = conv_block(nb_filter[1],nb_filter[2])
        self.Conv4 = conv_block(nb_filter[2],nb_filter[3])
        # self.Conv5 = conv_block(nb_filter[3],nb_filter[4])

        # self.Up5 = up_conv(nb_filter[4],ch_out=nb_filter[3])
        # self.Att5 = Attention_block(F_g=nb_filter[3],F_l=nb_filter[3],F_int=nb_filter[2])
        # self.Up_conv5 = conv_block(ch_in=nb_filter[4], ch_out=nb_filter[3])

        self.Up4 = up_conv(ch_in=nb_filter[3],ch_out=nb_filter[2])
        self.Att4 = Attention_block(F_g=nb_filter[2],F_l=nb_filter[2],F_int=nb_filter[1])
        self.Up_conv4 = conv_block(ch_in=nb_filter[3], ch_out=nb_filter[2])
        
        self.Up3 = up_conv(ch_in=nb_filter[2],ch_out=nb_filter[1])
        self.Att3 = Attention_block(F_g=nb_filter[1],F_l=nb_filter[1],F_int=nb_filter[0])
        self.Up_conv3 = conv_block(ch_in=nb_filter[2], ch_out=nb_filter[1])
        
        self.Up2 = up_conv(ch_in=nb_filter[1],ch_out=nb_filter[0])
        self.Att2 = Attention_block(F_g=nb_filter[0],F_l=nb_filter[0],F_int=16)
        self.Up_conv2 = conv_block(ch_in=nb_filter[1], ch_out=nb_filter[0])

        self.Conv_1x1 = nn.Conv2d(nb_filter[0],self.num_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # #decoding + concat path
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5,x=x4)
        # d5 = torch.cat((x4,d5),dim=1)        
        # d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(x4)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

