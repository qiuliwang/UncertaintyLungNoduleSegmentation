# -*- coding: utf-8 -*-

import torch.nn.functional as F
from torch import nn
import torch
from .utils import *


class BCELoss(nn.Module):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        return self.bce_loss(output, target)

class CELoss(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.ce_loss(output, target)

class CELoss2d(nn.Module):
    def __init__(self, **kwargs):
        super(CELoss2d, self).__init__()

    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        n, c, h, w = input.size()

        input = input.transpose(1, 2).transpose(2, 3).contiguous()
        input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        input = input.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.cross_entropy(input, target, weight=weight, size_average=False)
        if size_average:
            loss /= mask.data.sum()
        return loss

    def forward(self, output, target, weight=None, size_average=True):
        return self.cross_entropy2d(output, target,weight, size_average)

class BinaryDiceLoss(nn.Module):
	def __init__(self):
		super(BinaryDiceLoss, self).__init__()
	
	def forward(self, input, targets):
		# 获取每个批次的大小 N
		N = targets.size()[0]
		# 平滑变量
		smooth = 1
		# 将宽高 reshape 到同一纬度
		input_flat = input.view(N, -1)
		targets_flat = targets.view(N, -1)
	
		# 计算交集
		intersection = input_flat * targets_flat 
		N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
		# 计算一个批次中平均每张图的损失
		loss = 1 - N_dice_eff.sum() / N
		return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, input, target):
        input = input > 0.5
        target = target == torch.max(target)

        input = to_float_and_cuda(input)
        target = to_float_and_cuda(target)

        num = input * target
        num = torch.sum(num, dim=2)  # 在dim维度上求和 维度减1 如果想要保留原始维度 使用keepdim=True
        num = torch.sum(num, dim=2)

        den1 = input * input
        den1 = torch.sum(den1, dim=2)
        den1 = torch.sum(den1, dim=2)

        den2 = target * target
        den2 = torch.sum(den2, dim=2)
        den2 = torch.sum(den2, dim=2)

        dice = 2 * (num / (den1 + den2)) + 1e-6
        dice_total = 1 - 1 * torch.sum(dice) / dice.size(0)  # divide by batchsize

        return dice_total

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2))
        return loss_contrastive

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))

class IMLoss(nn.Module):
    def __init__(self, **kwargs):
        super(IMLoss, self).__init__()

    # feature1: student
    def forward(self, feature1, feature2, exp=4):
        if feature1.shape[2] != feature2.shape[2]:
            feature1 = F.interpolate(feature1, feature2.size()[-2:], mode='bilinear')
        return torch.sum((at(feature1, exp) - at(feature2, exp)).pow(2), dim=1).mean()

class KDChannelLoss(nn.Module):
    def __init__(self, **kwargs):
        super(KDChannelLoss, self).__init__()

    def forward(self, source_logits, target_logits, gt, num_classes):
        kd_loss = 0.0

        source_prob = []
        target_prob = []

        temperature = 2.0

        for i in range(num_classes):

            eps = 1e-6

            s_mask = torch.unsqueeze(gt[:,i,:,:], 1).repeat([1,num_classes,1,1])
            s_logits_mask_out = source_logits * s_mask
            s_logits_avg = torch.sum(s_logits_mask_out, [0,2,3]) / (torch.sum(gt[:,i,:,:]) + eps)
            s_soft_prob = F.softmax(s_logits_avg/temperature, dim=0)

            source_prob.append(s_soft_prob)

            t_mask = torch.unsqueeze(gt[:,i,:,:], 1).repeat([1,num_classes,1,1])
            t_logits_mask_out = target_logits * t_mask
            t_logits_avg = torch.sum(t_logits_mask_out, [0,2,3]) / (torch.sum(gt[:,i,:,:]) + eps)
            t_soft_prob = F.softmax(t_logits_avg/temperature, dim=0)

            target_prob.append(t_soft_prob)

            ## KL divergence loss
            loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob/t_soft_prob)) + torch.sum(t_soft_prob * torch.log(t_soft_prob/s_soft_prob))) / 2.0

            kd_loss += loss

        kd_loss = kd_loss / num_classes

        return kd_loss


"""Create loss"""
__factory = {
    'cross_entropy': CELoss,
    'cross_entropy2d': CELoss2d,
    'BCE_logit': BCELoss,
    'ContrastiveLoss':ContrastiveLoss,
    'BinaryDiceLoss': BinaryDiceLoss,
    'DiceLoss': DiceLoss,
    'IMLoss': IMLoss,
    'KDChannel': KDChannelLoss,
    }


def get_names():
    return __factory.keys()


def init_loss(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown loss: {}".format(name))
    return __factory[name](**kwargs)


if __name__ == '__main__':
    pass


