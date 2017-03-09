from torch.autograd import Function



# The Dice loss function is defined as
# 1/2 * intersection / union

class DiceLoss(Function):
    def __init__(self):
        pass

    def forward(self, input, target):
        self.save_for_backward(input, target)
        out = 2*torch.dot(input, target) / (torch.dot(input, input) +  torch.sum(target))
        return out

    def backward(self, grad_output):
        input, target = self.saved_tensors
        union = torch.dot(input, input) +  torch.sum(target)
        intersect = torch.dot(input, target)
        num = (torch.mul(target, union) - 2*torch.mul(input, intersect))
        denom = (union*union)
        out = 2*(num/denom)
