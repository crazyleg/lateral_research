import math
import torch
import typing as tp

from torch import nn, autograd
from torch.nn import init, functional as fn


class DualLinearFunction(autograd.Function):

    @staticmethod
    def forward(ctx, input, forward_weight, feedback_weight):
        ctx.save_for_backward(input, forward_weight, feedback_weight)
        return input.mm(forward_weight.t())

    @staticmethod
    def backward(ctx, grad_output):
        input, forward_weight, feedback_weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if ctx.needs_input_grad[0]:
            # calculate gradient flow using feedback_weight
            grad_input = grad_output.mm(feedback_weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        # don't update feedback_weight
        return grad_input, grad_weight, None


class DualLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(DualLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.forward_weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.forward_weight, a=math.sqrt(5))

    def forward(self, input, feedback_weight):
        return DualLinearFunction.apply(input, self.forward_weight, feedback_weight)


class ReversedSiameseNet(nn.Module):
    def __init__(self):
        super(ReversedSiameseNet, self).__init__()

        self.weights = nn.ParameterList([
            nn.Parameter(torch.Tensor(32, 784)),
            nn.Parameter(torch.Tensor(32, 32)),
            nn.Parameter(torch.Tensor(10, 32)),
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weights:
            init.kaiming_uniform_(weight, a=math.sqrt(5))

    def forward(self, grad: torch.Tensor, inputs: tp.List[torch.Tensor]):
        grad_weights = []
        for input, weight in zip(reversed(inputs), reversed(self.weights)):
            grad_weights.append(grad.t().mm(input[0]))
            grad = grad.mm(weight)

            # relu backward
            if input[1] is not None:
                grad[input[1] < 0] = 0

        return list(reversed(grad_weights))


class DualClassifier(nn.Module):
    def __init__(self):
        super(DualClassifier, self).__init__()

        self.l1 = DualLinear(784, 32)
        self.l2 = DualLinear(32, 32)
        self.l3 = DualLinear(32, 10)

    def forward(self, x, net: ReversedSiameseNet):
        inputs = []
        x = x.flatten(1, -1)

        inputs.append((x.detach(), None))
        x1 = self.l1(x, net.weights[0])
        x2 = torch.relu(x1)

        inputs.append((x2.detach(), x1.detach()))
        x1 = self.l2(x2, net.weights[1])
        x2 = torch.relu(x1)

        inputs.append((x2.detach(), x1.detach()))
        x = self.l3(x2, net.weights[2])
        return x, inputs
