import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

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

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

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

class Mymodel(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(Mymodel,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.pred1 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.pred2 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)
        self.pred3 = nn.Conv2d(32,output_ch,kernel_size=1,stride=1,padding=0)

        # self-attention
        self.pred_1 = single_conv(ch_in=64, ch_out=32)
        self.pred_2 = single_conv(ch_in=64, ch_out=32)
        self.pred_3 = single_conv(ch_in=64, ch_out=32)
        self.conv_Atten1 = Self_Attn(32,'relu')
        self.conv_Atten2 = Self_Attn(32,'relu')
        self.conv_Atten3 = Self_Attn(32,'relu')
        
        # fusion module
        # self.conv_fusion1 = conv_block(ch_in=160, ch_out=64)
        self.conv_fusion1 = conv_block(ch_in=192, ch_out=64)
        self.conv_fusion2 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)   # v1

        # Boundary-guided
        self.conv_boundary1 = conv_block(ch_in=320, ch_out=128)  # 256+64
        self.conv_boundary2 = conv_block(ch_in=192, ch_out=64)  # 128+64
        self.conv_boundary3 = conv_block(ch_in=64, ch_out=32)  # 128+64
        self.conv_boundary4 = nn.Conv2d(32,3,kernel_size=1,stride=1,padding=0)  # 128+64


    def forward(self,x):

        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)
        edge_fea1 = F.interpolate(d4, size=(64,64), mode='bilinear', align_corners=True)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        edge_fea2 = F.interpolate(d2, size=(64,64), mode='bilinear', align_corners=True)

        pred_1 = self.pred_1(d2)
        pred_2 = self.pred_2(d2)
        pred_3 = self.pred_3(d2)
        pred1 = self.pred1(pred_1)
        pred2 = self.pred2(pred_2)
        pred3 = self.pred3(pred_3)

        # Boundary-guided
        edge_map = torch.cat((edge_fea1, edge_fea2), dim=1)
        edge_map = self.conv_boundary1(edge_map)
        edge_map = EdgeConv(edge_map)
        edge_map = torch.cat((edge_map, edge_fea2), dim=1)
        edge_map = self.conv_boundary2(edge_map)
        edge_map = self.conv_boundary3(edge_map)
        edge_pred = self.conv_boundary4(edge_map)

        # self-attention
        attention_higher, _ = self.conv_Atten1(pred_1)
        attention_lower, _ = self.conv_Atten2(pred_2)
        attention_all, _ = self.conv_Atten3(pred_3)

        # fusion module
        # y = torch.cat((attention_higher, attention_all, attention_lower, d2), dim=1)
        y = torch.cat((attention_higher, attention_all, attention_lower, edge_map, d2), dim=1)
        y = self.conv_fusion1(y)
        pred = self.conv_fusion2(y)

        return [pred1, pred2, pred3, pred, edge_pred]


# 对图像进行了sobel操作（边缘计算）
def EdgeConv(im):
    in_channel = list(im.size())[1]
    out_channel = in_channel
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    '''定义sobel算子参数，所有值除以3----有人觉得出来的图更好些；但我感觉应该是概率问题，没啥用'''
    # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32') / 3
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出out_channle
    sobel_kernel = np.repeat(sobel_kernel, in_channel, axis=1)
    # 输入图的通道in_channle
    sobel_kernel = np.repeat(sobel_kernel, out_channel, axis=0)
    
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
    # print(conv_op.weight.size())
    # # print(conv_op, '\n')

    # print(im.shape)
    edge_detect = conv_op(im)
    # print(torch.max(edge_detect))
    # # 将输出转换为图片格式
    # edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect