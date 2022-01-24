import math
import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple


def neighborhood_convolution(
    x: Tensor, neighborhood: Tensor, weight: Tensor, bias: Optional[Tensor]
) -> Tensor:
    # x: [B, C_{in}, N]
    # neighborhood: [K]
    # weight: [C_{in}, K, C_{out}]
    # bias: [C_{out}]
    (batch_size, in_channels, number_spins) = x.size()
    (_, _, out_channels) = weight.size()
    relevant = x[..., neighborhood]
    # print(x.size(), relevant.size())
    output = torch.matmul(relevant.view((batch_size, -1)), weight.view((-1, out_channels)))
    if bias is not None:
        output = output + bias.unsqueeze(dim=0)
    return output


class LatticeConvolution(torch.nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    sublattices: int
    number_spins: int
    adj: List[Tuple[int, Tensor]]
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self, in_channels: int, out_channels: int, adj: List[Tuple[int, Tensor]]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.adj = adj
        self.number_spins = len(self.adj)
        self.kernel_size = adj[0][1].numel()
        self.sublattices = max(map(lambda t: t[0], adj)) + 1
        self.weight = Parameter(
            torch.empty(self.sublattices, self.in_channels, self.kernel_size, self.out_channels)
        )
        self.bias = Parameter(torch.empty(self.sublattices, self.out_channels))
        self.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.number_spins
        per_site = [
            neighborhood_convolution(
                x, neighborhood, self.weight[sublattice], self.bias[sublattice]
            )
            for (sublattice, neighborhood) in self.adj
        ]
        r = torch.stack(per_site, dim=2)
        assert r.size() == (x.size(0), self.out_channels, self.number_spins)
        return r

    def reset_parameters(self) -> None:
        k = 1 / (self.in_channels * self.kernel_size)
        torch.nn.init.uniform_(self.weight, -math.sqrt(k), math.sqrt(k))
        torch.nn.init.uniform_(self.bias, -math.sqrt(k), math.sqrt(k))

