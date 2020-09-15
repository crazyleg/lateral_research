import math
import torch

from torch import nn, autograd
from torch.nn import init


class ExtremeLinearFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input, forward_weight, feedback_weight):
        ctx.save_for_backward(input, forward_weight, feedback_weight)
        output = input.mm(forward_weight.t())
        ctx.output = output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, forward_weight, feedback_weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[1]:
            # grad_weight = grad_output.t().mm(input)

            # TODO: get rid of second atanh
            # inv_x = torch.atanh(torch.pinverse(input))
            # print(torch.isnan(inv_x).any())
            inv_x = torch.pinverse(input)
            error_weight = inv_x.mm(ctx.output - grad_output)
            grad_weight = forward_weight - error_weight.t()
            # grad_weight = grad_weight + (forward_weight - error_weight.t())

            # WOW
            # weight = weight - grad_weight * 0.01

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(forward_weight)

        return grad_input, grad_weight, None


class ExtremeLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(ExtremeLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.forward_weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.feedback_weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.forward_weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.feedback_weight, a=math.sqrt(5))

    def forward(self, input):
        return ExtremeLinearFunction.apply(input, self.forward_weight, self.feedback_weight)


class DeepELM(nn.Module):
    def __init__(self, in_features):
        super(DeepELM, self).__init__()

        self.in_features = in_features
        self.l1 = nn.Linear(784, 32)
        self.l2 = ExtremeLinear(32, 32)
        self.l3 = ExtremeLinear(32, 10)

    def forward(self, x):
        x = x.flatten(1, -1)
        x = self.l1(x)
        x = torch.tanh(x)
        x = self.l2(x)
        x = torch.tanh(x)
        x = self.l3(x)
        return x
