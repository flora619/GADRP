import math
import os

import torch
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        self.device=device
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))          # in_features × out_features
        self.a=Parameter(torch.zeros((1),**factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()



    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, H0,input, adj,a):

        H = (1 - a) * torch.matmul(adj, input) + a * H0
        W=self.weight
        output=torch.matmul(H,W)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
