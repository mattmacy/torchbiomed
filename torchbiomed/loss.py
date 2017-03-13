import torch
from torch.autograd import Function
from itertools import repeat
import numpy as np

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
        eps = 0.0000001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        result = torch.cuda.FloatTensor(result_.size())
        result.copy_(result_)
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
#        print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
#            union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, target = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = target.view(1, target.numel())
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, grad_output[0]),
                                torch.mul(dDice, -grad_output[0])), 0)
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)
