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
        input = input[0]
        intersect = torch.dot(input, target)
        union = torch.dot(input, input) + torch.sum(target)
        IoU = intersect/ union
#        print('union: {:.3f}\t intersect: {:.6f}\t IoU: {:.7f}'.format(
#            union, intersect, IoU))
        out = torch.FloatTensor(1).fill_(0.)
        out = torch.add(out, 2*IoU)
        return out

    def backward(self, grad_output):
        input, target = self.saved_tensors
        target = target.view(1, target.numel())
        union = torch.dot(input[0], input[0]) +  torch.sum(target)
        intersect = torch.dot(input[0], target)
        num = (torch.mul(target, union) - 2*torch.mul(input[0], intersect))
        denom = (union*union)

        grad_input = torch.cat((torch.mul(torch.div(num,denom), 2*grad_output[0]),
                                torch.mul(torch.div(num,denom), -2*grad_output[0])), 0)
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)
