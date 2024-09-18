import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import uniform
from torch.distributions.normal import Normal
from torch.optim import Adam, optimizer
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
from torch.autograd import grad

import scipy.io
from scipy.interpolate import griddata, interp1d

import itertools
from functools import partial

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import trange


# # var 1
# def MLP(layers, activation=F.relu):
#     def xavier_init(d_in, d_out):
#         glorot_stddev = 1. / (d_in + d_out) / 2.
#         W = torch.randn(d_in, d_out) * glorot_stddev
#         b = torch.zeros(d_out)
#         return nn.Parameter(W), nn.Parameter(b)
#
#     def init(key):
#         U1, b1 = xavier_init(layers[0], layers[1])
#         U2, b2 = xavier_init(layers[0], layers[1])
#
#         def init_layer(d_in, d_out):
#             W, b = xavier_init(d_in, d_out)
#             return nn.Linear(d_in, d_out)
#
#         torch.manual_seed(key)
#         key, *keys = torch.randint(0, 10 ** 10, (len(layers),))
#         params = [init_layer(d_in, d_out) for d_in, d_out in zip(layers[:-1], layers[1:])]
#         return params, U1, b1, U2, b2
#
#     def apply(params, inputs):
#         a = len(*params)
#         U1, b1, U2, b2 = params[0][6:]
#         params = params[0][:6]
#         U = activation(F.linear(inputs, U1, b1))
#         V = activation(F.linear(inputs, U2, b2))
#         for layer in params[:-1]:
#             outputs = activation(layer(inputs))
#             inputs = torch.mul(outputs, U) + torch.mul(1 - outputs, V)
#         outputs = F.linear(inputs, *params[-1].parameters())
#         return outputs
#
#     return init, apply


# var 2
# class MLP(nn.Module):
#     def __init__(self, layers, activation=F.relu):
#         super(MLP, self).__init__()
#         self.layers = layers
#         self.activation = activation
#
#         # Random seed for reproducibility
#         torch.manual_seed(0)
#         keys = torch.randint(0, 10 ** 10, (len(layers),))
#
#         # Initialize parameters
#         self.params = nn.ModuleList()
#         self.U1, self.b1 = self.xavier_init(layers[0], layers[1], keys[0])
#         self.U2, self.b2 = self.xavier_init(layers[0], layers[1], keys[1])
#
#         for i in range(len(layers) - 1):
#             W, b = self.xavier_init(layers[i], layers[i + 1], keys[i])
#             self.params.append(nn.Parameter(W))
#             self.params.append(nn.Parameter(b))
#
#     def xavier_init(self, d_in, d_out, key):
#         torch.manual_seed(key.item())
#         glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
#         W = glorot_stddev * torch.randn(d_in, d_out)
#         b = torch.zeros(d_out)
#         return W, b
#
#     def forward(self, inputs):
#         U = self.activation(inputs @ self.U1 + self.b1)
#         V = self.activation(inputs @ self.U2 + self.b2)
#
#         for i in range(0, len(self.params) - 2, 2):
#             W = self.params[i]
#             b = self.params[i + 1]
#             outputs = self.activation(inputs @ W + b)
#             inputs = torch.mul(outputs, U) + torch.mul(1 - outputs, V)
#
#         W = self.params[-2]
#         b = self.params[-1]
#         outputs = inputs @ W + b
#
#         return outputs


# # var 3
# class MLP(nn.Module):
#     def __init__(self, layers, activation=F.relu):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#
#         for i in range(len(layers) - 1):
#             self.layers.append(nn.Linear(layers[i], layers[i+1]))
#             nn.init.xavier_uniform_(self.layers[-1].weight)
#             nn.init.zeros_(self.layers[-1].bias)
#
#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = self.activation(layer(x))
#         x = self.layers[-1](x)
#         return x


# var 4
class MLP(nn.Module):
    def __init__(self, layers, activation=F.relu):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

        # Additional networks U1, b1 and U2, b2
        self.U1 = nn.Linear(layers[0], layers[1])
        self.U2 = nn.Linear(layers[0], layers[1])
        nn.init.xavier_uniform_(self.U1.weight)
        nn.init.zeros_(self.U1.bias)
        nn.init.xavier_uniform_(self.U2.weight)
        nn.init.zeros_(self.U2.bias)

    def forward(self, x):
        U = self.activation(self.U1(x))
        V = self.activation(self.U2(x))
        inputs = x

        for layer in self.layers[:-1]:
            outputs = self.activation(layer(inputs))
            inputs = outputs * U + (1 - outputs) * V

        x = self.layers[-1](inputs)
        return x


# # var 1
# class PI_DeepONet:
#     def __init__(self, branch_layers, trunk_layers):
#         # Network initialization and evaluation functions
#         self.branch_init, self.branch_apply = MLP(branch_layers, activation=torch.tanh)
#         self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=torch.tanh)
#
#         # Initialize
#         torch.manual_seed(1234)
#         key_branch_init = torch.randint(0, 10 ** 10, (1,))
#         branch_params = self.branch_init(key_branch_init)[0]
#
#         torch.manual_seed(4321)
#         key_trunk_init = torch.randint(0, 10 ** 10, (1,))
#         trunk_params = self.trunk_init(key_trunk_init)[0]
#
#         # params = torch.tensor([branch_params, trunk_params])
#         # params = torch.stack(branch_params, trunk_params)
#         # params = branch_params + trunk_params
#
#         # Extract parameters from the Linear layers
#         branch_params_tensors = [param for layer in branch_params for param in layer.parameters()]  # Исправлено
#         trunk_params_tensors = [param for layer in trunk_params for param in layer.parameters()]  # Исправлено
#
#         self.params = branch_params_tensors + trunk_params_tensors
#
#         lr = torch.optim.lr_scheduler.ExponentialLR(torch.optim.Adam(self.params), gamma=0.9,
#                                                     last_epoch=-1)  # Исправлено
#         self.optimizer = torch.optim.Adam(self.params)
#
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#         # self.opt_init, self.opt_update, self.get_params = torch.optim.optimizers.adam(lr)
#         # self.opt_state = self.opt_init(params)
#         # _, self.unravel = torch.nn.utils.ravel_multiindex(params)
#
#     # Define the operator net
#     def operator_net(self, params, u, t, x):
#         a = params
#         s = len(params)
#         branch_params, trunk_params = [params[:len(params) // 2]], params[len(params) // 2:]
#         y = torch.stack([t, x])
#         B = self.branch_apply(branch_params, u)
#         T = self.trunk_apply(trunk_params, y)
#         outputs = torch.sum(B * T)
#         return outputs
#
#     def s_t_net(self, params, u, t, x):
#         s_t = grad(self.operator_net, inputs=(params, u, t, x), create_graph=True)[0][2]
#         return s_t
#
#     def residual_net(self, params, u, t, x):
#         s_tt = grad(self.s_t_net, inputs=(params, u, t, x), create_graph=True)[0][2]
#         s_xx = grad(self.s_t_net, inputs=(params, u, t, x), create_graph=True)[0][3]
#         a = 1 / 4
#         res = a * s_tt - s_xx
#         return res
#
#     def loss_ics(self, params, batch):
#         inputs, outputs = batch
#         (u, y) = inputs
#
#         # s_pred = torch.vmap(self.operator_net)(params, u, y[:,0], y[:,1])
#         # s_t_pred = torch.vmap(self.s_t_net)(params, u, y[:,0], y[:,1])
#         # loss_1 = torch.mean((outputs.flatten() - s_pred)**2)
#         # loss_2 = torch.mean(s_t_pred ** 2)
#
#         s_pred = torch.stack([self.operator_net(params, u[i], y[i, 0], y[i, 1]) for i in range(len(u[0]))])  # u.size(0)
#         s_t_pred = torch.stack([self.s_t_net(params, u[i], y[i, 0], y[i, 1]) for i in range(len(u[0]))])  # u.size(0)
#         loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
#         loss_2 = torch.mean(s_t_pred ** 2)
#
#         loss = loss_1 + loss_2
#
#         return loss
#
#     def loss_bcs(self, params, batch):
#         inputs, outputs = batch
#         u, y = inputs
#
#         # s_bc1_pred = torch.vmap(self.operator_net)(params, u, y[:,0], y[:,1])
#         # s_bc2_pred = torch.vmap(self.operator_net)(params, u, y[:,2], y[:,3])
#         # loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:,0]) ** 2)
#         # loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:,1]) ** 2)
#
#         s_bc1_pred = torch.stack(
#             [self.operator_net(params, u[i], y[i, 0], y[i, 1]) for i in range(len(u[0]))])  # u.size(0)
#         s_bc2_pred = torch.stack(
#             [self.operator_net(params, u[i], y[i, 2], y[i, 3]) for i in range(len(u[0]))])  # u.size(0)
#         loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
#         loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
#
#         loss_s_bc = loss_s_bc1 + loss_s_bc2
#         return loss_s_bc
#
#     def loss_res(self, params, batch):
#         inputs, _ = batch
#         u, y = inputs
#
#         # pred = torch.vmap(self.residual_net)(params, u, y[:,0], y[:,1])
#         pred = torch.stack([self.residual_net(params, u[i], y[i, 0], y[i, 1]) for i in range(len(u[0]))])  # u.size(0)
#         loss = torch.mean(pred ** 2)
#
#         return loss
#
#     def loss(self, params, ics_batch, bcs_batch, res_batch):
#         loss_ics = self.loss_ics(params, ics_batch)
#         loss_bcs = self.loss_bcs(params, bcs_batch)
#         loss_res = self.loss_res(params, res_batch)
#         loss = loss_ics + loss_bcs + loss_res
#         return loss
#
#     # Define a compiled update step
#     def step(self, i, opt_state, ics_batch, bcs_batch, res_batch):
#         params = [p for p in self.params]
#         opt_state.zero_grad()
#         loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
#         loss_value.backward()
#         opt_state.step()
#         return loss_value
#     # def step(self, i, ics_batch, bcs_batch, res_batch):
#     #     self.optimizer.zero_grad()
#     #     params = tuple(self.optimizer.param_groups[0]['params'])
#     #     loss = self.loss(params, ics_batch, bcs_batch, res_batch)
#     #     loss.backward()
#     #     self.optimizer.step()
#     #     return loss.item()
#
#     # def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
#     #     ics_data = iter(ics_dataset)
#     #     bcs_data = iter(bcs_dataset)
#     #     res_data = iter(res_dataset)
#
#     #     pbar = trange(nIter)
#     #     for it in pbar:
#     #         ics_batch = next(ics_data)
#     #         bcs_batch = next(bcs_data)
#     #         res_batch = next(res_data)
#
#     #         loss_value = self.step(it, ics_batch, bcs_batch, res_batch)
#     #         self.loss_log.append(loss_value)
#     #         pbar.set_postfix({'Loss': loss_value})
#
#     def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
#         opt_state = torch.optim.Adam(self.params)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(opt_state, gamma=0.9)
#
#         ics_data = iter(ics_dataset)
#         bcs_data = iter(bcs_dataset)
#         res_data = iter(res_dataset)
#
#         pbar = trange(nIter)
#
#         # Main training loop
#         for it in pbar:
#             ics_batch = next(ics_data)
#             bcs_batch = next(bcs_data)
#             res_batch = next(res_data)
#
#             loss_value = self.step(next(self.itercount), opt_state, ics_batch, bcs_batch, res_batch)
#
#             if it % 1000 == 0:
#                 params = [p for p in self.params]
#
#                 loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
#                 loss_ics_value = self.loss_ics(params, ics_batch)
#                 loss_bcs_value = self.loss_bcs(params, bcs_batch)
#                 loss_res_value = self.loss_res(params, res_batch)
#
#                 self.loss_log.append(loss_value.item())
#                 self.loss_ics_log.append(loss_ics_value.item())
#                 self.loss_bcs_log.append(loss_bcs_value.item())
#                 self.loss_res_log.append(loss_res_value.item())
#
#                 pbar.set_postfix({
#                     'Loss': loss_value.item(),
#                     'loss_ics': loss_ics_value.item(),
#                     'loss_bcs': loss_bcs_value.item(),
#                     'loss_res': loss_res_value.item()
#                 })
#             scheduler.step()
#
#     def predict_s(self, U_star, Y_star):
#         with torch.no_grad():
#             s_pred = torch.stack(
#                 [self.operator_net(U_star[i], Y_star[i, 0], Y_star[i, 1]) for i in range(U_star.size(0))])
#         return s_pred
#
#     def predict_res(self, U_star, Y_star):
#         with torch.no_grad():
#             r_pred = torch.stack(
#                 [self.residual_net(U_star[i], Y_star[i, 0], Y_star[i, 1]) for i in range(U_star.size(0))])
#         return r_pred


# # var 2
# class PI_DeepONet(nn.Module):
#     def __init__(self, branch_layers, trunk_layers):
#         super(PI_DeepONet, self).__init__()
#         self.branch_net = MLP(branch_layers, activation=torch.tanh)
#         self.trunk_net = MLP(trunk_layers, activation=torch.tanh)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#     def operator_net(self, params, u, t, x):
#         branch_params, trunk_params = params
#         y = torch.stack([t, x], dim=-1)
#         B = self.branch_net(branch_params, u)
#         T = self.trunk_net(trunk_params, y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
#     def s_t_net(self, params, u, t, x):
#         t.requires_grad_(True)
#         s = self.operator_net(params, u, t, x)
#         s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#         return s_t
#
#     def residual_net(self, params, u, t, x):
#         t.requires_grad_(True)
#         x.requires_grad_(True)
#         s = self.operator_net(params, u, t, x)
#         s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#         s_tt = torch.autograd.grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
#         s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
#         s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
#         a = 1 / 4
#         res = a * s_tt - s_xx
#         return res
#
#     def loss_ics(self, params, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         s_pred = self.operator_net(params, u, y[:, 0], y[:, 1])
#         s_t_pred = self.s_t_net(params, u, y[:, 0], y[:, 1])
#         loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
#         loss_2 = torch.mean(s_t_pred ** 2)
#         loss = loss_1 + loss_2
#         return loss
#
#     def loss_bcs(self, params, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         s_bc1_pred = self.operator_net(params, u, y[:, 0], y[:, 1])
#         s_bc2_pred = self.operator_net(params, u, y[:, 2], y[:, 3])
#         loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
#         loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
#         loss_s_bc = loss_s_bc1 + loss_s_bc2
#         return loss_s_bc
#
#     def loss_res(self, params, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         pred = self.residual_net(params, u, y[:, 0], y[:, 1])
#         loss = torch.mean(pred ** 2)
#         return loss
#
#     def loss(self, params, ics_batch, bcs_batch, res_batch):
#         loss_ics = self.loss_ics(params, ics_batch)
#         loss_bcs = self.loss_bcs(params, bcs_batch)
#         loss_res = self.loss_res(params, res_batch)
#         loss = loss_ics + loss_bcs + loss_res
#         return loss
#
#     def step(self, i, opt_state, ics_batch, bcs_batch, res_batch):
#         params = list(self.parameters())
#         self.optimizer.zero_grad()
#         loss = self.loss(params, ics_batch, bcs_batch, res_batch)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()
#
#     def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
#         ics_data = iter(ics_dataset)
#         bcs_data = iter(bcs_dataset)
#         res_data = iter(res_dataset)
#
#         pbar = trange(nIter)
#         for it in pbar:
#             ics_batch = next(ics_data)
#             bcs_batch = next(bcs_data)
#             res_batch = next(res_data)
#
#             loss_value = self.step(next(self.itercount), self.optimizer, ics_batch, bcs_batch, res_batch)
#             # self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch)
#
#             if it % 1000 == 0:
#                 params = list(self.parameters())
#
#                 loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
#                 loss_ics_value = self.loss_ics(params, ics_batch)
#                 loss_bcs_value = self.loss_bcs(params, bcs_batch)
#                 loss_res_value = self.loss_res(params, res_batch)
#
#                 self.loss_log.append(loss_value)
#                 self.loss_ics_log.append(loss_ics_value.item())
#                 self.loss_bcs_log.append(loss_bcs_value.item())
#                 self.loss_res_log.append(loss_res_value.item())
#
#                 pbar.set_postfix({'Loss': loss_value,
#                                   'loss_ics': loss_ics_value.item(),
#                                   'loss_bcs': loss_bcs_value.item(),
#                                   'loss_res': loss_res_value.item()})
#
#     def predict_s(self, params, U_star, Y_star):
#         val = params
#         s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return s_pred
#
#     def predict_res(self, params, U_star, Y_star):
#         val = params
#         r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return r_pred


# # var 3
# class PI_DeepONet(nn.Module):
#     def __init__(self, branch_layers, trunk_layers):
#         super(PI_DeepONet, self).__init__()
#         self.branch_net = MLP(branch_layers, activation=torch.tanh)
#         self.trunk_net = MLP(trunk_layers, activation=torch.tanh)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#     def operator_net(self, u, t, x):
#         y = torch.stack([t, x], dim=-1)
#         B = self.branch_net(u)
#         T = self.trunk_net(y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
#     def s_t_net(self, u, t, x):
#         t.requires_grad_(True)
#         s = self.operator_net(u, t, x)
#         s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#         return s_t
#
#     def residual_net(self, u, t, x):
#         t.requires_grad_(True)
#         x.requires_grad_(True)
#         s = self.operator_net(u, t, x)
#         s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#         s_tt = torch.autograd.grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
#         s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
#         s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
#         a = 1 / 4
#         res = a * s_tt - s_xx
#         return res
#
#     def loss_ics(self, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         s_pred = self.operator_net(u, y[:, 0], y[:, 1])
#         s_t_pred = self.s_t_net(u, y[:, 0], y[:, 1])
#         loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
#         loss_2 = torch.mean(s_t_pred ** 2)
#         loss = loss_1 + loss_2
#         return loss
#
#     def loss_bcs(self, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
#         s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])
#         loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
#         loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
#         loss_s_bc = loss_s_bc1 + loss_s_bc2
#         return loss_s_bc
#
#     def loss_res(self, batch):
#         inputs, outputs = batch
#         u, y = inputs
#         pred = self.residual_net(u, y[:, 0], y[:, 1])
#         loss = torch.mean(pred ** 2)
#         return loss
#
#     def loss(self, ics_batch, bcs_batch, res_batch):
#         loss_ics = self.loss_ics(ics_batch)
#         loss_bcs = self.loss_bcs(bcs_batch)
#         loss_res = self.loss_res(res_batch)
#         loss = loss_ics + loss_bcs + loss_res
#         return loss
#
#     def step(self, i, opt_state, ics_batch, bcs_batch, res_batch):
#         self.optimizer.zero_grad()
#         loss = self.loss(ics_batch, bcs_batch, res_batch)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()
#
#     def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
#         ics_data = iter(ics_dataset)
#         bcs_data = iter(bcs_dataset)
#         res_data = iter(res_dataset)
#
#         pbar = trange(nIter)
#         for it in pbar:
#             ics_batch = next(ics_data)
#             bcs_batch = next(bcs_data)
#             res_batch = next(res_data)
#
#             loss_value = self.step(next(self.itercount), self.optimizer, ics_batch, bcs_batch, res_batch)
#
#             if it % 1000 == 0:
#                 loss_value = self.loss(ics_batch, bcs_batch, res_batch)
#                 loss_ics_value = self.loss_ics(ics_batch)
#                 loss_bcs_value = self.loss_bcs(bcs_batch)
#                 loss_res_value = self.loss_res(res_batch)
#
#                 self.loss_log.append(loss_value.item())
#                 self.loss_ics_log.append(loss_ics_value.item())
#                 self.loss_bcs_log.append(loss_bcs_value.item())
#                 self.loss_res_log.append(loss_res_value.item())
#
#                 pbar.set_postfix({'Loss': loss_value.item(),
#                                   'loss_ics': loss_ics_value.item(),
#                                   'loss_bcs': loss_bcs_value.item(),
#                                   'loss_res': loss_res_value.item()})
#
#     def predict_s(self, U_star, Y_star):
#         s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return s_pred
#
#     def predict_res(self, U_star, Y_star):
#         r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return r_pred


# var 4
class PI_DeepONet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super(PI_DeepONet, self).__init__()
        self.branch_net = MLP(branch_layers, activation=torch.tanh)
        self.trunk_net = MLP(trunk_layers, activation=torch.tanh)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    def operator_net(self, u, t, x):
        y = torch.stack([t, x], dim=-1)
        B = self.branch_net(u)
        T = self.trunk_net(y)
        outputs = torch.sum(B * T, dim=-1)
        return outputs

    def s_t_net(self, u, t, x):
        t.requires_grad_(True)
        s = self.operator_net(u, t, x)
        s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
        return s_t

    def residual_net(self, u, t, x):
        t.requires_grad_(True)
        x.requires_grad_(True)
        s = self.operator_net(u, t, x)
        s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
        s_tt = torch.autograd.grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
        s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
        s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
        a = 1 / 4
        res = a * s_tt - s_xx
        return res

    def loss_ics(self, u, y, outputs):
        s_pred = self.operator_net(u, y[:, 0], y[:, 1])
        s_t_pred = self.s_t_net(u, y[:, 0], y[:, 1])
        loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
        loss_2 = torch.mean(s_t_pred ** 2)
        loss = loss_1 + loss_2
        return loss

    def loss_bcs(self, u, y, outputs):
        s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
        s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])
        loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
        loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
        loss_s_bc = loss_s_bc1 + loss_s_bc2
        return loss_s_bc

    def loss_res(self, u, y):
        pred = self.residual_net(u, y[:, 0], y[:, 1])
        loss = torch.mean(pred ** 2)
        return loss

    def loss(self, ics_batch, bcs_batch, res_batch):
        ics_inputs, ics_outputs = ics_batch
        bcs_inputs, bcs_outputs = bcs_batch
        res_inputs, _ = res_batch

        u_ics, y_ics = ics_inputs
        u_bcs, y_bcs = bcs_inputs
        u_res, y_res = res_inputs

        loss_ics = self.loss_ics(u_ics, y_ics, ics_outputs)
        loss_bcs = self.loss_bcs(u_bcs, y_bcs, bcs_outputs)
        loss_res = self.loss_res(u_res, y_res)

        lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 10.0
        loss = lambda_ics * loss_ics + lambda_bcs * loss_bcs + lambda_res * loss_res

        return loss

    def step(self, ics_batch, bcs_batch, res_batch):
        self.optimizer.zero_grad()
        loss = self.loss(ics_batch, bcs_batch, res_batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        for _ in pbar:
            ics_batch = next(ics_data)
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            loss_value = self.step(ics_batch, bcs_batch, res_batch)
            loss_ics_value = self.loss_ics(ics_batch[0][0], ics_batch[0][1], ics_batch[1])
            loss_bcs_value = self.loss_bcs(bcs_batch[0][0], bcs_batch[0][1], bcs_batch[1])
            loss_res_value = self.loss_res(res_batch[0][0], res_batch[0][1])

            # loss_value = self.loss(ics_batch, bcs_batch, res_batch)
            # loss_ics_value = self.loss_ics(ics_batch)
            # loss_bcs_value = self.loss_bcs(bcs_batch)
            # loss_res_value = self.loss_res(res_batch)

            self.loss_log.append(loss_value)
            self.loss_ics_log.append(loss_ics_value.item())
            self.loss_bcs_log.append(loss_bcs_value.item())
            self.loss_res_log.append(loss_res_value.item())

            pbar.set_postfix({'Loss': loss_value,
                              'loss_ics': loss_ics_value.item(),
                              'loss_bcs': loss_bcs_value.item(),
                              'loss_res': loss_res_value.item()})

    def predict_s(self, U_star, Y_star):
        s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred

    def predict_res(self, U_star, Y_star):
        r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, batch_size=64, rng_seed=1234):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.rng_seed = rng_seed

    def __getitem__(self, index):
        'Generate one batch of data'
        torch.manual_seed(self.rng_seed)  # Используем rng_seed в качестве seed для нового подключа
        subkey = torch.randint(0, 100000, (1,))
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    def __len__(self):
        'Denotes the total number of samples'
        return self.N // self.batch_size

    def __data_generation(self, subkey):
        'Generates data containing batch_size samples'
        idx = torch.randperm(self.N)[:self.batch_size]
        s = self.s[idx]
        y = self.y[idx]
        u = self.u[idx]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


# length_scale = 0.5
# output_scale = 10.0
# gp_params = (length_scale, output_scale)
# c = 1.0
#
#
# # var 1
# def RBF(X1, X2, gp_params):
#     length_scale, output_scale = gp_params
#     sqdist = torch.sum(X1 ** 2, 1).reshape(-1, 1) + torch.sum(X2 ** 2, 1) - 2 * torch.matmul(X1, X2.T)
#     return output_scale ** 2 * torch.exp(-0.5 / length_scale ** 2 * sqdist + 1e-6)
#
#
# def generate_one_gaussain_sample(key, gp_params, N):
#     length_scale, output_scale = gp_params  # по желанию можно использовать или здесь, или в функции RBF непосредственно
#     jitter = 1e-10
#     X = torch.linspace(0.0, 1.0, N).reshape(-1, 1)
#     K = RBF(X, X, gp_params)
#
#     # var 1
#     # L = torch.linalg.cholesky(K + jitter * torch.eye(N))  # Здесь была ошибка
#
#     # var 2
#     # try:
#     #     L = torch.linalg.cholesky(K)
#     # except torch.linalg.LinAlgError:
#     #     # Попробуем еще раз с большим jitter
#     #     jitter *= 10
#     #     K += jitter * torch.eye(N)
#     #     L = torch.linalg.cholesky(K)
#
#     # var 3
#     L = torch.linalg.lu(K + jitter * torch.eye(N))[0]  # LU decomposition
#
#     torch.manual_seed(key)
#     normal_dist = Normal(torch.zeros(N), torch.ones(N))
#     gp_sample = L @ normal_dist.sample()
#
#     return gp_sample
#
#
# # # var 2
# # def RBF(x1, x2, gp_params):
# #     length_scale, output_scale = gp_params
# #     diffs = torch.unsqueeze(x1 / length_scale, 1) - \
# #             torch.unsqueeze(x2 / length_scale, 0)
# #     r2 = torch.sum(diffs ** 2, dim=2)
# #     return output_scale * torch.exp(-0.5 * r2)
# #
# #
# # def generate_one_gaussain_sample(key, gp_params, N):
# #     length_scale, output_scale = gp_params
# #     jitter = 1e-10
# #     X = torch.linspace(0.0, 1.0, N)[:, None]
# #     K = RBF(X, X, gp_params)
# #     L, info = torch.linalg.cholesky_ex(K + jitter * torch.eye(N))
# #     torch.manual_seed(key)
# #
# #     # gp_sample = torch.matmul(L, torch.randn(N))
# #
# #     # # 1st variant
# #     # gp_sample = torch.mm(L, torch.randn((1, N)).t()).squeeze()
# #
# #     # # 2nd variant
# #     # gp_sample = torch.matmul(L, torch.randn(N))
# #     # gp_sample_size = gp_sample.shape
# #
# #     # # 3rd variant
# #     # gp_sample = torch.mm(L, torch.distributions.normal.Normal(
# #     #     torch.zeros(1, N),
# #     #     torch.ones(1, N)).sample())
# #     # gp_sample_size = gp_sample.shape
# #
# #     # # 4th variant
# #     # gp_sample = torch.einsum('ij,jk->ik', L, Normal(torch.zeros(1, N), torch.ones(1, N)).sample())
# #     # gp_sample_size = gp_sample.shape
# #
# #     # 5th variant
# #     gp_sample = torch.einsum('ij->i', L @ Normal(torch.zeros(N, 1), torch.ones(N, 1)).sample())
# #     gp_sample_size = gp_sample.shape
# #
# #     # # 6th variant
# #     # gp_sample = torch.mm(L, torch.randn(N))
# #     # gp_sample_size = gp_sample.shape
# #
# #     return gp_sample
#
#
# # # var 3
# # def generate_one_gaussain_sample(seed, gp_params, N):
# #     np.random.seed(seed)
# #     return np.random.normal(loc=gp_params['mean'], scale=gp_params['std'], size=N)
#
#
# def generate_one_training_data(key, m=100, P=100, Q=100):
#     torch.manual_seed(key)
#     subkeys = torch.randint(0, 10 ** 10, (5,))
#
#     # Генерация одного образца ввода
#     N = 512
#     # gp_params = {'mean': 0, 'std': 1}
#
#     gp_sample = generate_one_gaussain_sample(int(subkeys[0].item()), gp_params, N)
#     x = torch.linspace(0, 1, m)
#     X = torch.linspace(0, 1, N).unsqueeze(1)
#
#     # var 1
#     def u_fn(x):
#         # g_samp = gp_sample.numpy()
#         interp_gp = interp1d(X.flatten(), gp_sample.numpy())
#         # val1 = x
#         # val2 = 1 - x
#         # val3 = torch.tensor(interp_gp(x))
#         return x * (1 - x) * torch.tensor(interp_gp(x))
#
#     # # var 2
#     # def u_fn(x):
#     #     interp_gp = interp1d(X.flatten().numpy(), gp_sample)
#     #     return x * (1 - x) * torch.tensor(interp_gp(x.numpy()), dtype=torch.float32)
#
#     # # var 3 - ERROR
#     # if len(gp_sample.shape) == 0:
#     #     gp_sample = gp_sample.unsqueeze(0)  # Add a new dimension if scalar
#     #
#     # x = torch.linspace(0, 1, m)
#     # X = torch.linspace(0, 1, N).unsqueeze(1)
#     #
#     # def u_fn(x):
#     #     interp_gp = torch.nn.functional.interpolate(gp_sample.unsqueeze(0), size=(x.size(0),), mode='linear')
#     #     return x * (1 - x) * interp_gp.squeeze()
#
#     # Входные датчики и измерения
#     u = torch.stack([u_fn(x_i) for x_i in x])
#
#     # IC обучающие данные
#     u_ic = torch.tile(u, (P, 1))
#     t_0 = torch.zeros((P, 1))
#     torch.manual_seed(int(subkeys[1].item()))
#     x_0 = torch.rand((P, 1))
#
#     y_ic = torch.hstack([t_0, x_0])
#     s_ic = torch.stack([u_fn(x_0_i) for x_0_i in x_0])
#
#     # BC обучающие данные
#     u_bc = torch.tile(u, (P, 1))
#
#     torch.manual_seed(int(subkeys[2].item()))
#     t_bc1 = torch.rand((P, 1))
#     x_bc1 = torch.zeros((P, 1))
#     torch.manual_seed(int(subkeys[3].item()))
#     t_bc2 = torch.rand((P, 1))
#     x_bc2 = torch.ones((P, 1))
#     y_bc1 = torch.hstack([t_bc1, x_bc1])
#     y_bc2 = torch.hstack([t_bc2, x_bc2])
#     y_bc = torch.hstack([y_bc1, y_bc2])
#     s_bc = torch.zeros((Q, 2))
#
#     # Residual обучающие данные
#     u_r = torch.tile(u, (Q, 1))
#     torch.manual_seed(int(subkeys[4].item()))
#     y_r = torch.rand((Q, 2))
#     s_r = torch.zeros((Q, 1))
#
#     return u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r
#
#
# def generate_training_data(key, N, m, P, Q):
#     torch.manual_seed(key)
#     keys = torch.randint(0, 10 ** 10, (N,))
#     # print(keys.shape)
#
#     # # NEED CORRECT
#     # u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r = \
#     #     torch.vmap(generate_one_training_data)(keys, torch.tensor([m]), torch.tensor([P]), torch.tensor([Q]))
#
#     u_ic, y_ic, s_ic = [], [], []
#     u_bc, y_bc, s_bc = [], [], []
#     u_r, y_r, s_r = [], [], []
#
#     for key in keys:
#         u_ic_i, y_ic_i, s_ic_i, u_bc_i, y_bc_i, s_bc_i, u_r_i, y_r_i, s_r_i\
#             = generate_one_training_data(key, m, P, Q)
#         u_ic.append(u_ic_i)
#         y_ic.append(y_ic_i)
#         s_ic.append(s_ic_i)
#
#         u_bc.append(u_bc_i)
#         y_bc.append(y_bc_i)
#         s_bc.append(s_bc_i)
#
#         u_r.append(u_r_i)
#         y_r.append(y_r_i)
#         s_r.append(s_r_i)
#
#     u_ic, y_ic, s_ic = torch.stack(u_ic), torch.stack(y_ic), torch.stack(s_ic)
#     u_bc, y_bc, s_bc = torch.stack(u_bc), torch.stack(y_bc), torch.stack(s_bc)
#     u_r, y_r, s_r = torch.stack(u_r), torch.stack(y_r), torch.stack(s_r)
#
#     # print(u_ic.shape, y_ic.shape, s_ic.shape)
#     # print(u_bc.shape, y_bc.shape, s_bc.shape)
#     # print(u_r.shape, y_r.shape, s_r.shape)
#
#     u_ic_train = u_ic.view(N * P, -1).float()
#     y_ic_train = y_ic.view(N * P, -1).float()
#     s_ic_train = s_ic.view(N * P, -1).float()
#
#     u_bc_train = u_bc.view(N * P, -1).float()
#     y_bc_train = y_bc.view(N * P, -1).float()
#     s_bc_train = s_bc.view(N * P, -1).float()
#
#     u_r_train = u_r.view(N * Q, -1).float()
#     y_r_train = y_r.view(N * Q, -1).float()
#     s_r_train = s_r.view(N * Q, -1).float()
#
#     return u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train


# # Параметры для гауссовского процесса
# length_scale = 0.5
# output_scale = 10.0
# gp_params = (length_scale, output_scale)
#
#
# def RBF(x1, x2, gp_params):
#     length_scale, output_scale = gp_params
#     diffs = torch.unsqueeze(x1 / length_scale, 1) - torch.unsqueeze(x2 / length_scale, 0)
#     r2 = torch.sum(diffs ** 2, dim=2)
#     return output_scale * torch.exp(-0.5 * r2)
#
#
# def generate_one_gaussian_sample(seed, gp_params, N):
#     jitter = 1e-10
#     X = torch.linspace(0.0, 1.0, N).unsqueeze(1)
#     K = RBF(X, X, gp_params)
#     # L = torch.linalg.cholesky(K + jitter * torch.eye(N))
#     L = torch.linalg.lu(K + jitter * torch.eye(N))[0]  # LU decomposition
#     gp_sample = torch.matmul(L, torch.normal(mean=0.0, std=1.0, size=(N,)).float())
#
#     return gp_sample
#
#
# # def generate_one_gaussian_sample(seed, gp_params, N):
# #     np.random.seed(seed)
# #     return np.random.normal(loc=gp_params['mean'], scale=gp_params['std'], size=N)
#
#
# def generate_one_training_data(seed, m=100, P=100, Q=100):
#     torch.manual_seed(seed)
#     subkeys = torch.randint(0, 10000, (5,), dtype=torch.int64)
#
#     # Генерация одного образца ввода
#     N = 512
#     gp_params = {'mean': 0, 'std': 1}
#
#     gp_sample = generate_one_gaussian_sample(int(subkeys[0].item()), gp_params, N)
#     x = torch.linspace(0, 1, m)
#     X = torch.linspace(0, 1, N).unsqueeze(1)
#
#     def u_fn(x):
#         interp_gp = interp1d(X.flatten().numpy(), gp_sample)
#         return x * (1 - x) * torch.tensor(interp_gp(x.numpy()), dtype=torch.float32)
#
#     # Входные датчики и измерения
#     u = torch.stack([u_fn(x_i) for x_i in x])
#
#     # IC обучающие данные
#     u_ic = u.repeat((P, 1))
#     t_0 = torch.zeros((P, 1), dtype=torch.float32)
#     torch.manual_seed(int(subkeys[1].item()))
#     x_0 = torch.rand((P, 1), dtype=torch.float32)
#     y_ic = torch.cat([t_0, x_0], dim=1)
#     s_ic = torch.stack([u_fn(x_0_i) for x_0_i in x_0])
#
#     # BC обучающие данные
#     u_bc = u.repeat((P, 1))
#     torch.manual_seed(int(subkeys[2].item()))
#     t_bc1 = torch.rand((P, 1), dtype=torch.float32)
#     x_bc1 = torch.zeros((P, 1), dtype=torch.float32)
#     torch.manual_seed(int(subkeys[3].item()))
#     t_bc2 = torch.rand((P, 1), dtype=torch.float32)
#     x_bc2 = torch.ones((P, 1), dtype=torch.float32)
#     y_bc1 = torch.cat([t_bc1, x_bc1], dim=1)
#     y_bc2 = torch.cat([t_bc2, x_bc2], dim=1)
#     y_bc = torch.cat([y_bc1, y_bc2], dim=1)
#     s_bc = torch.zeros((P, 2), dtype=torch.float32)
#
#     # Residual обучающие данные
#     u_r = u.repeat((Q, 1))
#     torch.manual_seed(int(subkeys[4].item()))
#     y_r = torch.rand((Q, 2), dtype=torch.float32)
#     s_r = torch.zeros((Q, 1), dtype=torch.float32)
#
#     return u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r
#
#
# def generate_training_data(seed, N, m, P, Q):
#     torch.manual_seed(seed)
#     keys = torch.randint(0, 10000, (N,), dtype=torch.int64)
#
#     u_ic, y_ic, s_ic = [], [], []
#     u_bc, y_bc, s_bc = [], [], []
#     u_r, y_r, s_r = [], [], []
#
#     for key in keys:
#         u_ic_i, y_ic_i, s_ic_i, u_bc_i, y_bc_i, s_bc_i, u_r_i, y_r_i, s_r_i = generate_one_training_data(key.item(), m, P, Q)
#         u_ic.append(u_ic_i)
#         y_ic.append(y_ic_i)
#         s_ic.append(s_ic_i)
#
#         u_bc.append(u_bc_i)
#         y_bc.append(y_bc_i)
#         s_bc.append(s_bc_i)
#
#         u_r.append(u_r_i)
#         y_r.append(y_r_i)
#         s_r.append(s_r_i)
#
#     u_ic = torch.cat(u_ic)
#     y_ic = torch.cat(y_ic)
#     s_ic = torch.cat(s_ic)
#     u_bc = torch.cat(u_bc)
#     y_bc = torch.cat(y_bc)
#     s_bc = torch.cat(s_bc)
#     u_r = torch.cat(u_r)
#     y_r = torch.cat(y_r)
#     s_r = torch.cat(s_r)
#
#     u_ic_train = u_ic.view(N * P, -1).float()
#     y_ic_train = y_ic.view(N * P, -1).float()
#     s_ic_train = s_ic.view(N * P, -1).float()
#
#     u_bc_train = u_bc.view(N * P, -1).float()
#     y_bc_train = y_bc.view(N * P, -1).float()
#     s_bc_train = s_bc.view(N * P, -1).float()
#
#     u_r_train = u_r.view(N * Q, -1).float()
#     y_r_train = y_r.view(N * Q, -1).float()
#     s_r_train = s_r.view(N * Q, -1).float()
#
#     return u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train


# Hyperparameters
length_scale = 0.1
output_scale = 10.0
gp_params = (length_scale, output_scale)
# c = 1.0


def RBF(x1, x2, gp_params):
    length_scale, output_scale = gp_params
    diffs = torch.unsqueeze(x1 / length_scale, 1) - torch.unsqueeze(x2 / length_scale, 0)
    r2 = torch.sum(diffs ** 2, dim=2)
    return output_scale * torch.exp(-0.5 * r2)


def generate_one_gaussain_sample(key, gp_params, N):
    torch.manual_seed(key)
    jitter = 1e-10
    X = torch.linspace(0.0, 1.0, N).view(-1, 1)
    K = RBF(X, X, gp_params)
    L = torch.linalg.lu(K + jitter * torch.eye(N))[0]
    gp_sample = torch.matmul(L, torch.normal(mean=0.0, std=1.0, size=(N,)).float())
    return gp_sample


def generate_one_training_data(key, m=100, P=100, Q=100):
    subkeys = torch.Generator().manual_seed(key).manual_seed(key).seed()

    # Generate one input sample
    N = 512
    gp_sample = generate_one_gaussain_sample(subkeys, gp_params, N)
    x = torch.linspace(0, 1, m)
    X = torch.linspace(0, 1, N).view(-1, 1)

    def u_fn(x):
        interp_gp = np.interp(x, X.numpy().flatten(), gp_sample.numpy())
        return x * (1 - x) * torch.from_numpy(interp_gp)

    # Input sensor locations and measurements
    u = u_fn(x)

    # IC training data
    u_ic = u.repeat(P, 1)

    t_0 = torch.zeros((P, 1))
    x_0 = torch.rand((P, 1))
    y_ic = torch.hstack([t_0, x_0])
    s_ic = u_fn(x_0)

    # BC training data
    u_bc = u.repeat(P, 1)

    t_bc1 = torch.rand((P, 1))
    x_bc1 = torch.zeros((P, 1))

    t_bc2 = torch.rand((P, 1))
    x_bc2 = torch.ones((P, 1))

    y_bc1 = torch.hstack([t_bc1, x_bc1])
    y_bc2 = torch.hstack([t_bc2, x_bc2])
    y_bc = torch.hstack([y_bc1, y_bc2])

    s_bc = torch.zeros((Q, 2))

    # Residual training data
    u_r = u.repeat(Q, 1)
    y_r = torch.rand((Q, 2))
    s_r = torch.zeros((Q, 1))

    return u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r


def generate_training_data(key, N, m, P, Q):
    torch.manual_seed(key)
    keys = torch.randint(0, 10 ** 10, (N,), dtype=torch.int64)

    results = [generate_one_training_data(key.item(), m, P, Q) for key in keys]

    u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r = zip(*results)

    u_ic_train = torch.cat(u_ic).float()
    y_ic_train = torch.cat(y_ic).float()
    s_ic_train = torch.cat(s_ic).float()

    u_bc_train = torch.cat(u_bc).float()
    y_bc_train = torch.cat(y_bc).float()
    s_bc_train = torch.cat(s_bc).float()

    u_r_train = torch.cat(u_r).float()
    y_r_train = torch.cat(y_r).float()
    s_r_train = torch.cat(s_r).float()

    return u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train


N_train = 441
N_test = 21
m = 21
P_train = 21
Q_train = 21

# Generate training data
torch.manual_seed(0)  # use different key for generating training data
key_train = torch.randint(0, 10 ** 10, (1, ))
u_ics_train, y_ics_train, s_ics_train, u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = \
    generate_training_data(key_train, N_train, m, P_train, Q_train)

# print('u_ics_train =', u_ics_train, '\ny_ics_train =', y_ics_train, '\ns_ics_train =', s_ics_train)
# print('u_bcs_train =', u_bcs_train, '\ny_bcs_train =', y_bcs_train, '\ns_bcs_train =', s_res_train)
# print('u_res_train =', u_res_train, '\ny_res_train =', y_res_train, '\ns_res_train =', s_res_train)

# Initialize model
neurons = 200
branch_layers = [m, neurons, neurons, neurons, neurons, neurons]
trunk_layers = [2, neurons, neurons, neurons, neurons, neurons]
model = PI_DeepONet(branch_layers, trunk_layers)

# Create data set
batch_size = 512
ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

# Train
model.train(ics_dataset, bcs_dataset, res_dataset, nIter=20000)

# Получение параметров модели
params = model.state_dict()

# Сохранение параметров модели
torch.save(params, 'weights/512/wave_params_512_0.pth')

# Сохранение логов потерь
np.save('weights/512/wave_loss_res_512_0.npy', np.array(model.loss_res_log))
np.save('weights/512/wave_loss_ics_512_0.npy', np.array(model.loss_ics_log))
np.save('weights/512/wave_loss_bcs_512_0.npy', np.array(model.loss_bcs_log))

# Загрузка модели и логов потерь
flat_params = torch.load('weights/512/wave_params_512_0.pth')
loss_ics = np.load('weights/512/wave_loss_ics_512_0.npy')
loss_bcs = np.load('weights/512/wave_loss_bcs_512_0.npy')
loss_res = np.load('weights/512/wave_loss_res_512_0.npy')

# # Example
# loss_res = np.load('weights/0/wave_loss_res_0.npy')
# loss_res = np.load('wave_loss_res_0.npy')

# Загрузка модели
params = model.parameters(flat_params)

# Настройка графиков
plt.rc('font', family='serif')
plt.rcParams.update(plt.rcParamsDefault)

# График потерь
fig = plt.figure(figsize=(16, 12))

iters = 1000 * np.arange(len(loss_ics))
plt.plot(iters, loss_bcs, lw=2, label='$\mathcal{L}_{bc}$')
plt.plot(iters, loss_ics, lw=2, label='$\mathcal{L}_{ic}$')
plt.plot(iters, loss_res, lw=2, label='$\mathcal{L}_{r}$')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.show()

# Предсказанное решение
T = 21
P = 21
P_test = 21

s_fn = lambda t, x: np.sin(np.pi * x) * np.cos(np.pi * 2 * t)

x = np.linspace(0, 1, P)
u = np.sin(np.pi * x)

u_test = np.tile(u, (P ** 2, 1))

x = np.linspace(0, 1, P_test)
t = np.linspace(0, 1, P_test)
TT, XX = np.meshgrid(t, x)

y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])

# Предсказание модели
s_pred = model.predict_s(torch.tensor(u_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')

for k in range(T - 1):
    u_k = S_pred[:, -1]
    u_test_k = np.tile(u_k, (P_test ** 2, 1))
    s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
    S_pred = np.hstack([S_pred, S_pred_k])

# Точное решение
Nx = 21
Nt = 21 * T

x = np.linspace(0, 1, Nx)
t = np.linspace(0, T, Nt)
TT, XX = np.meshgrid(t, x)

S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)

# Относительная ошибка
error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
print('Relative l2 error: {:.3e}'.format(error))

# Визуализация решений
fig = plt.figure(figsize=(16, 12))

plt.subplot(3, 1, 1)
plt.pcolor(TT, XX, S_test, cmap='jet')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Exact')
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 2)
plt.pcolor(TT, XX, S_pred, cmap='jet')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title('Predicted')
plt.colorbar()
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.pcolor(TT, XX, np.abs(S_pred - S_test), cmap='jet')
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.colorbar()
plt.title('Absolute error')
plt.tight_layout()
plt.show()

# 3D Визуализация решений
Nx = 21
Nt = 21

x = torch.linspace(0, 1, Nx)
t = torch.linspace(0, 1, Nt)

grid = torch.cartesian_prod(x, t)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

xs = grid[:, 0].detach().numpy().reshape(-1)
ys = grid[:, 1].detach().numpy().reshape(-1)
zs = s_pred.detach().numpy()

ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
ax.set_title('PI DeepONet solution (3D)')
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

plt.show()

########################################################################################################################

# # key = random.PRNGKey(54321)
# key_train = random.PRNGKey(7)
# print('key_train = ', key_train)
# # print('type of key_train = ', type(key_train))
# # print('key_train.shape = ', key_train.shape)

# keys = random.split(key_train, 5)
# print('keys after random.split = ', keys)
# print('keys[3] = ', keys[3])
# print('keys[4] = ', keys[-1])
# print('keys.shape = ', keys.shape)

# keys = [[j for j in i] for i in keys]
# print('keys.shape after random.split = ', keys.shape)

# print(torch.randint(0, 10 ** 10, (5, 2))[0][0])
#
# torch.manual_seed(0)
# t1 = torch.randint(0, 10 ** 10, (5, 2))
# print(t1)
#
# torch.manual_seed(1)
# t2 = torch.randint(0, 10 ** 10, (5, 2))
# print(t2)
#
# torch.manual_seed(2)
# t3 = torch.randint(0, 10 ** 10, (5, 2))
# print(t3)
#
# torch.manual_seed(3)
# t4 = torch.randint(0, 10 ** 10, (5, 2))
# print(t4)
#
# torch.manual_seed(0)
# t = torch.randint(0, 10 ** 10, (5, 2))
# print(t)
#
# # print(t.reshape(1, -1))
#
# a = torch.arange(10).reshape(2, 5)
# # print(a)
# # print(torch.split(a, 1))
#
# k = random.normal(keys[0], (512, ))
# print(k)
# print(k.shape)

# arr = jax.random.normal(jax.random.split(jax.random.PRNGKey(0), 5)[0], (5,))
# print(arr)
# print(jax.random.split(jax.random.PRNGKey(0), 5).reshape(10, 1))
#
# torch.manual_seed(0)
# a = torch.randint(0, 10 ** 10, (5,))
# print(a)
# print(int(a[0].item()))
# print(torch.randn(5))
#
# print(jax.random.uniform(jax.random.split(jax.random.PRNGKey(0), 5)[0], (10, 1)))
# print(torch.rand(5, 1))
#
# N = 10
# t = torch.randint(0, 10 ** 10, (N,))
# print(t)
# print(jax.random.split(jax.random.PRNGKey(0), N))
#
# [print(int(i)) for i in t]
#
# A = torch.randn(2, 2, dtype=torch.complex128)
# print(A)
# print(torch.linalg.cholesky_ex(A))














