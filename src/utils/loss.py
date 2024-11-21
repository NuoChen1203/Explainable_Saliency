import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from numpy import pi
import math

epsilon = 1e-7 #regularization value in Keras

def NSS(input,fixation):    
    input = input.view(input.size(0),-1)
    input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input))
    fixation = fixation.view(fixation.size(0),-1)
    input = torch.div(input-input.mean(-1,keepdim=True).expand_as(input),input.std(-1,keepdim=True).expand_as(input) + epsilon)
    loss = torch.div(torch.mul(input,fixation).sum(-1), fixation.sum(-1) + epsilon)

    return -torch.mean(loss)

def CC(input,fixmap): 
    input = input.view(input.size(0),-1)
    input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input))
    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap)+epsilon)
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input)+epsilon)

    sum_prod = torch.mul(input,fixmap).sum(-1,keepdim=True)
    sum_x = input.sum(-1,keepdim=True)
    sum_y = fixmap.sum(-1,keepdim=True)
    sum_x_square = (input**2).sum(-1,keepdim=True)
    sum_y_square = (fixmap**2).sum(-1,keepdim=True)
    num = sum_prod - torch.mul(sum_x,sum_y)/input.size(-1)
    den = torch.sqrt((sum_x_square-sum_x**2/input.size(-1))*(sum_y_square-sum_y**2/input.size(-1)))
    loss = torch.div(num,den+epsilon)

    return -2*torch.mean(loss)




def KLD(input,fixmap):
    input = input.view(input.size(0),-1)
    input = torch.div(input,input.max(-1,keepdim=True)[0].expand_as(input))
    fixmap = fixmap.view(fixmap.size(0),-1)
    fixmap = torch.div(fixmap,fixmap.sum(-1,keepdim=True).expand_as(fixmap))
    input = torch.div(input,input.sum(-1,keepdim=True).expand_as(input))
    loss = torch.mul(fixmap,torch.log(torch.div(fixmap,input+epsilon) + epsilon)).sum(-1)

    return 10*torch.mean(loss)


def LL(input,fixmap):
    input = input.view(input.size(0),-1)
    input = F.softmax(input,dim=-1)
    fixmap = fixmap.view(fixmap.size(0),-1)
    loss =  torch.mul(torch.log(input+epsilon),fixmap).sum(-1)

    return -torch.sum(loss)


def cross_entropy(input,target):
    input = input.view(input.size(0), -1)
    input = F.softmax(input,dim=-1)
    target = target.view(target.size(0),-1)
    loss = (-target*torch.log(torch.clamp(input,min=epsilon,max=1))).sum(-1)
    loss = torch.mean(loss)
    return loss.mean()
