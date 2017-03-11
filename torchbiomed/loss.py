import torch
from torch.autograd import Function
from itertools import repeat

# Intersection = dot(A, B)
# Union = dot(A, A) + dot(B, B)
# The Dice loss function is defined as
# 1/2 * intersection / union
#
# The derivative is 2[(union * target - 2 * intersect * input) / union^2]

class DiceLoss(Function):
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target):
        self.save_for_backward(input, target)
        target = target.view(1, target.numel())
        # target is a binary valued tensor - dot(target, target) => sum(target)
        target_neg = torch.cuda.FloatTensor(target.size()).fill_(1.)
        target_neg = target_neg - target
        target = torch.cat((target, target_neg), 0)
        intersect = 2*torch.dot(input, target)
        union = torch.dot(input, input) + torch.sum(target)
        out = torch.FloatTensor(1).zero_()
        out = torch.add(out, intersect / union)
        return out

    def backward(self, grad_output):
        input, target = self.saved_tensors
        target_neg = torch.cuda.FloatTensor(target.size()).fill_(1.)
        target_neg = target_neg - target
        target = torch.cat((target, target_neg), 0)

        union = torch.dot(input, input) +  torch.sum(target)
        intersect = torch.dot(input, target)
        num = (torch.mul(target, union) - 2*torch.mul(input, intersect))
        denom = (union*union)
        grad_input = torch.mul(torch.div(num,denom), 2)
        grad_input = torch.mul(grad_input, grad_output[0])
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)
