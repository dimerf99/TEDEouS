import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from typing import *


# class SplineLinear(nn.Linear):
#     def __init__(self,
#                  in_features: int,
#                  out_features: int,
#                  init_scale: float = 0.1,
#                  **kw) -> None:
#
#         self.init_scale = init_scale
#         super().__init__(in_features, out_features, bias=False, **kw)
#
#     def reset_parameters(self) -> None:
#         nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
#
#
# class RadialBasisFunction(nn.Module):
#     def __init__(
#         self,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         denominator: float = None,  # larger denominators lead to smoother basis
#     ):
#         super().__init__()
#         grid = torch.linspace(grid_min, grid_max, num_grids)
#         self.grid = torch.nn.Parameter(grid, requires_grad=False)
#         self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
#
#     def forward(self, x):
#         return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
#
#
# class FastKANLayer(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         use_base_update: bool = True,
#         base_activation=F.silu,
#         spline_weight_init_scale: float = 0.1,
#     ):
#
#         super(FastKANLayer, self).__init__()
#         self.layernorm = nn.LayerNorm(input_dim)
#         self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
#         self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
#         self.use_base_update = use_base_update
#         if use_base_update:
#             self.base_activation = base_activation
#             self.base_linear = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x, time_benchmark=False):
#         if not time_benchmark:
#             spline_basis = self.rbf(self.layernorm(x))
#         else:
#             spline_basis = self.rbf(x)
#         ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
#         if self.use_base_update:
#             base = self.base_linear(self.base_activation(x))
#             ret = ret + base
#         return ret
#
#
# class FastKAN(nn.Module):
#     def __init__(
#         self,
#         layers_hidden: List[int],
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         use_base_update: bool = True,
#         base_activation=F.silu,
#         spline_weight_init_scale: float = 0.1,
#     ):
#
#         super(FastKAN, self).__init__()
#         self.model = nn.ModuleList([
#             FastKANLayer(
#                 in_dim, out_dim,
#                 grid_min=grid_min,
#                 grid_max=grid_max,
#                 num_grids=num_grids,
#                 use_base_update=use_base_update,
#                 base_activation=base_activation,
#                 spline_weight_init_scale=spline_weight_init_scale,
#             ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
#         ])
#
#     def forward(self, x):
#         for layer in self.model:
#             x = layer(x)
#         return x


# class SplineLinear(nn.Module):
#     def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
#         self.init_scale = init_scale
#         super().__init__(**kw)
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
#
#     def forward(self, x):
#         return F.linear(x, self.weight, bias=None)
#
#
# class RadialBasisFunction(nn.Module):
#     def __init__(
#         self,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         denominator: float = None,  # larger denominators lead to smoother basis
#     ):
#         super().__init__()
#         grid = torch.linspace(grid_min, grid_max, num_grids)
#         self.grid = torch.nn.Parameter(grid, requires_grad=False)
#         self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
#
#     def forward(self, x):
#         return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)
#
#
# class FastKANLayer(nn.Module):
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         grid_min: float = -2.,
#         grid_max: float = 2.,
#         num_grids: int = 8,
#         use_base_update: bool = True,
#         base_activation=F.silu,
#         spline_weight_init_scale: float = 0.1,
#     ):
#
#         super(FastKANLayer, self).__init__()
#         self.layernorm = nn.LayerNorm(input_dim)
#         self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
#         self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
#         self.use_base_update = use_base_update
#         if use_base_update:
#             self.base_activation = base_activation
#             self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
#             nn.init.xavier_normal_(self.base_weight)
#
#     def forward(self, x, time_benchmark=False):
#         if not time_benchmark:
#             spline_basis = self.rbf(self.layernorm(x))
#         else:
#             spline_basis = self.rbf(x)
#         ret = F.linear(spline_basis.view(*spline_basis.shape[:-2], -1), self.spline_linear.weight, bias=None)
#         if self.use_base_update:
#             base = F.linear(self.base_activation(x), self.base_weight, bias=None)
#             ret = ret + base
#         return ret
#
#
# class FastKAN(nn.Module):
#     def __init__(
#         self,
#         layers_hidden: List[int],
#         grid_min: float = -10.,
#         grid_max: float = 10.,
#         num_grids: int = 2,
#         use_base_update: bool = True,
#         base_activation=F.silu,
#         spline_weight_init_scale: float = 0.1,
#     ):
#
#         super(FastKAN, self).__init__()
#         self.model = nn.ModuleList([
#             FastKANLayer(
#                 in_dim, out_dim,
#                 grid_min=grid_min,
#                 grid_max=grid_max,
#                 num_grids=num_grids,
#                 use_base_update=use_base_update,
#                 base_activation=base_activation,
#                 spline_weight_init_scale=spline_weight_init_scale,
#             ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
#         ])
#
#     def forward(self, x):
#         for layer in self.model:
#             x = layer(x)
#         return x


class SplineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_scale = init_scale
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

    def forward(self, x):
        return F.linear(x, self.weight)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use layernorms on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
            self.reset_parameters()

    def reset_parameters(self):
        if self.use_base_update:
            nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            x = self.layernorm(x)
        spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(x.size(0), -1))
        if self.use_base_update:
            base = F.linear(self.base_activation(x), self.base_weight)
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 5,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:

        super(FastKAN, self).__init__()
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




