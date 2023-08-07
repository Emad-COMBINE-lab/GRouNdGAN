import math
import typing

import torch
import torch.nn as nn


class MaskedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks its weights by 'mask'.
    """

    @staticmethod
    def forward(
        ctx: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx: torch.Tensor, grad_output: torch.Tensor):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class MaskedLinear(nn.Module):
    def __init__(
        self,
        mask: torch.Tensor,
        bias: bool = True,
        device: typing.Optional[str] = "cuda" if torch.cuda.is_available() else "cpu",
    ):    
        """
        An extension of Pytorch's linear module based on the following thread:
        https://discuss.pytorch.org/t/custom-connections-in-neural-network-layers/3027/13

        Parameters
        ----------
        mask : torch.Tensor
            Mask Tensor with shape (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.

            Example: the following mask declares a 4-dim from-layer and 3-dim to-layer.
            Neurons 0, 2, and 3 of the from-layer are connected to neurons 0 and 2 of
            the to-layer. Neuron 1 of the from-layer is connected to neuron 1 of the
            to-layer.

            mask = torch.tensor(
            [[1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
            [1, 0, 1],]
            )

        bias : bool, optional
            By default True

        device : typing.Optional[str], optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        """
        super(MaskedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        self.device = device

        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        self.weight = nn.Parameter(
            torch.Tensor(self.output_features, self.input_features).to(self.device)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features).to(self.device))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reapply_mask(self):
        """Function to be called after weights have been initialized
        (e.g., using torch.nn.init) to reapply mask to weight."""
        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor):
        return MaskedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return "input_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None
        )


if __name__ == "check grad":
    from torch.autograd import gradcheck

    customlinear = MaskedLinearFunction.apply

    input = (
        torch.randn(20, 20, dtype=torch.double, requires_grad=True),
        torch.randn(30, 20, dtype=torch.double, requires_grad=True),
        None,
        None,
    )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)
