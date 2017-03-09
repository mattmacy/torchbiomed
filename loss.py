from torch.autograd import Function


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
        # target is a binary valued tensor - dot(target, target) => sum(target)
        out = 2*torch.dot(input, target) / (torch.dot(input, input) +  torch.sum(target))
        return out

    def backward(self, grad_output):
        input, target = self.saved_tensors
        union = torch.dot(input, input) +  torch.sum(target)
        intersect = torch.dot(input, target)
        num = (torch.mul(target, union) - 2*torch.mul(input, intersect))
        denom = (union*union)
        grad_input = 2*(num/denom)
        grad_output_expanded = grad_output.view(*repeat(1, grad_input.dim()))
        grad_input.mul_(grad_output_expanded.expand_as(grad_input))
        return grad_input , None

def dice_loss(input, target):
    return DiceLoss()(input, target)
