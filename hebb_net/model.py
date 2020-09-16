import math
import torch
import itertools

from tqdm import tqdm
from torch import nn, autograd
from torch.nn import init, functional as fn


def act_fn(h):
    k = torch.ones_like(h)
    return torch.max(-1 * k, torch.min(k, h))


class LinearHebbianFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, feedback_weight):
        ctx.save_for_backward(input, weight, feedback_weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, feedback_weight = ctx.saved_tensors
        grad_input = grad_weight = grad_feedback_weight = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
            # grad_input = torch.nn.functional.normalize(grad_input)
            grad_input = fn.normalize(grad_input, p=2, dim=0)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]:
            grad_feedback_weight = grad_weight
        else:
            grad_feedback_weight = None

        return grad_input, grad_weight, grad_feedback_weight

class HebbianLinear(nn.Module):
    def __init__(self, input_features, output_features, backward_type='FXFB'):
        super(HebbianLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.forward_weight = nn.Parameter(torch.Tensor(output_features, input_features))
        need_feedback_update = True if backward_type == 'URFB' else False
        self.feedback_weight = nn.Parameter(torch.Tensor(output_features, input_features), requires_grad=need_feedback_update)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.forward_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.feedback_weight, a=math.sqrt(5))

    def forward(self, input):
        return LinearHebbianFunction.apply(input, self.forward_weight, self.feedback_weight)

class HebbMLP(nn.Module):
    def __init__(self, in_features):
        super(HebbMLP, self).__init__()

        self.in_features = in_features
        self.lin1 = HebbianLinear(3072, 512, backward_type='URFB')
        self.lin2 = HebbianLinear(512, 128, backward_type='URFB')
        self.lin3 = HebbianLinear(128, 10, backward_type='URFB')

    def forward(self, x):
        x = x.flatten(1, -1)
        x = self.lin1(x)
        x = act_fn(x)
        x = self.lin2(x)
        x = act_fn(x)
        x = self.lin3(x)
        return x
