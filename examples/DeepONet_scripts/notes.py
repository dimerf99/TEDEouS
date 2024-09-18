# import torch
# import jax
# import numpy as np
#
# import jax.numpy as jnp
# from jax import random, vmap, grad, jit
# import scipy.io
# from scipy.interpolate import griddata, interp1d
#
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.distributions as dist
# from torch.distributions import uniform
# from torch.optim import Adam, optimizer
# from torch.utils.data import Dataset, DataLoader
# from torch.utils import data
# from torch.autograd import grad
#
# import itertools
# from functools import partial
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# def func(x):
#     return x ** 2
#
#
# a, b, c, *d = torch.vmap(func)(torch.arange(5))
# print(a, b, c, d)
#
# # a, b, c, d = torch.vmap(func)(torch.arange(4), torch.tensor([4]), torch.tensor([5]), torch.tensor([6]))
# # print(a, b, c, d)
#
#
# a, b, c, d = jax.vmap(func)(jnp.arange(4))
# print(a, b, c, d)
# print(type(a))
#
#
# u = torch.stack([func(num) for num in torch.arange(4)])
# print()


# a = [torch.arange(3) for _ in range(5)]
# print(a)
#
# res = torch.stack(a)
# print(res)


# from scipy.interpolate import interp1d

# # Создать массив точек данных
# x = np.array([0, 1, 2, 3, 4])
# y = np.array([0, 1, 4, 9, 16])
#
# # Создать интерполятор
# interpolator = interp1d(x, y)
#
# # Оценить значение функции в точке x=1.5
# y_interp = interpolator(1.5)
#
# print(y_interp)  # Выведет: 2.25


# t = torch.tensor([[[1, 2],
#                    [3, 4]],
#                   [[5, 6],
#                    [7, 8]]])
#
# print(t.flatten())
#
# x = torch.tensor([[[[1, 2]], [[3, 4]], [[5, 6]]]])
#
# print(x.flatten())


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.fc1 = nn.Linear(2, 16)
#         self.fc2 = nn.Linear(16, 32)
#         self.fc3 = nn.Linear(32, 1)
#
#     def forward(self, x):
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         x = nn.functional.relu(self.fc3(x))
#         return x
#
#
# class PINN():
#     def __init__(self, X, u, lb, ub, physics):
#         self.lb = torch.tensor(lb).float()
#         self.ub = torch.tensor(ub).float()
#         self.physics = physics
#
#         self.x = torch.tensor(X[:, 0:1], requires_grad=True).float()
#         self.t = torch.tensor(X[:, 1:2], requires_grad=True).float()
#         self.u = torch.tensor(u).float()
#
#         self.network = Network()
#
#         self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.001)
#
#     def makeNetwork(self, x, t):
#         X = torch.cat([x, t], 1)
#         return self.network(X)
#
#     def residual(self, x, t):
#         u = self.makeNetwork(x, t)
#         u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#         u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#         u_xx = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#         u_tt = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#         # return u_tt - 4 * u_xx
#         return u_t + u * u_x - (0.01 / np.pi) * u_xx
#
#     def lossResidual(self):
#         u_pred = self.makeNetwork(self.x, self.t)
#         residual_pred = self.residual(self.x, self.t)
#         loss = torch.mean((self.u - u_pred))
#         if self.physics is True:
#             loss += torch.mean(residual_pred ** 2)
#         self.optimizer.zero_grad()
#         loss.backward()
#         return loss
#
#
# a = torch.arange(6, dtype=torch.float64)
# print(a)
# print(a.reshape(-1, 1))
# print(a.reshape(1, -1))
# print(torch.reshape(a, (-1,)))

# class PINN(nn.Module):
#     def __init__(self):
#         super(PINN, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(2, 100),
#             nn.Tanh(),
#             nn.Linear(100, 100),
#             nn.Tanh(),
#             nn.Linear(100, 100),
#             nn.Tanh(),
#             nn.Linear(100, 1)
#         )
#
#     def forward(self, x, t):
#         x_t_tensor = torch.cat((x, t), dim=1)
#         return self.layers(x_t_tensor)
#
#     def loss_function(self, x, t):
#         f = self.forward(x, t)
#         f_x = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
#         f_xx = torch.autograd.grad(f_x.sum(), x, create_graph=True)[0]
#         f_t = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
#         f_tt = torch.autograd.grad(f_t.sum(), t, create_graph=True)[0]
#
#         wave_pde = f_tt - 4 * f_xx
#         laplace_pde = f_tt + f_xx + 2 * f
#
#         wave_pde_loss = torch.mean()
#         laplace_pde_loss = torch.mean()
#
#         return laplace_pde_loss


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import itertools
# from tqdm import trange
#
#
# class MLP(nn.Module):
#     def __init__(self, layers, activation=F.relu):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#
#         # Initialize the layers with Xavier initialization
#         for i in range(len(layers) - 1):
#             layer = nn.Linear(layers[i], layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.layers.append(layer)
#
#         # Additional networks U1, b1 and U2, b2
#         self.U1 = nn.Linear(layers[0], layers[1])
#         self.U2 = nn.Linear(layers[0], layers[1])
#
#         nn.init.xavier_normal_(self.U1.weight)
#         nn.init.zeros_(self.U1.bias)
#         nn.init.xavier_normal_(self.U2.weight)
#         nn.init.zeros_(self.U2.bias)
#
#     def forward(self, x):
#         U = self.activation(self.U1(x))
#         V = self.activation(self.U2(x))
#
#         for layer in self.layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#
#         x = self.layers[-1](x)
#         return x
#
#
# # MLP and PI DeepONet should be
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
#     # eval.py in solver (???)
#     def operator_net(self, u, t, x):
#         y = torch.stack([t, x], dim=-1)
#         B = self.branch_net(u)
#         T = self.trunk_net(y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
#     # eval.py in solver
#     def residual_net(self, u, t, x):
#         t.requires_grad_(True)
#         x.requires_grad_(True)
#
#         s = self.operator_net(u, t, x)
#         s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#         s_tt = torch.autograd.grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
#         s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
#         s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]
#
#         a = 1 / 4
#         res = a * s_tt - s_xx
#         return res
#
#     # losses ###############################################################################
#     # losses.py in solver
#     def loss_ics(self, u, y, outputs):
#         t = y[:, 0]
#         x = y[:, 1]
#
#         s_pred = self.operator_net(u, t, x)
#         t.requires_grad_(True)
#         s = self.operator_net(u, t, x)
#         s_t_pred = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
#
#         loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
#         loss_2 = torch.mean(s_t_pred ** 2)
#         loss = loss_1 + loss_2
#
#         return loss
#
#     # losses.py in solver
#     def loss_bcs(self, u, y, outputs):
#         s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
#         s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])
#
#         loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
#         loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
#         loss_s_bc = loss_s_bc1 + loss_s_bc2
#         return loss_s_bc
#
#     # losses.py in solver
#     def loss_res(self, u, y):
#         pred = self.residual_net(u, y[:, 0], y[:, 1])
#         loss = torch.mean(pred ** 2)
#         return loss
#
#     # losses.py in solver
#     def loss(self, ics_batch, bcs_batch, res_batch):
#         ics_inputs, ics_outputs = ics_batch
#         bcs_inputs, bcs_outputs = bcs_batch
#         res_inputs, _ = res_batch
#
#         u_ics, y_ics = ics_inputs
#         u_bcs, y_bcs = bcs_inputs
#         u_res, y_res = res_inputs
#
#         loss_ics = self.loss_ics(u_ics, y_ics, ics_outputs)
#         loss_bcs = self.loss_bcs(u_bcs, y_bcs, bcs_outputs)
#         loss_res = self.loss_res(u_res, y_res)
#
#         lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 4.0  # мало искажений, медленная сходимость (лучший)
#
#         loss = lambda_ics * loss_ics + lambda_bcs * loss_bcs + lambda_res * loss_res
#         return loss
#
#     ########################################################################################
#
#     # closure.py in solver
#     def step(self, ics_batch, bcs_batch, res_batch):
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
#         for i in pbar:
#             ics_batch = next(ics_data)
#             bcs_batch = next(bcs_data)
#             res_batch = next(res_data)
#
#             loss_value = self.step(ics_batch, bcs_batch, res_batch)
#             loss_ics_value = self.loss_ics(ics_batch[0][0], ics_batch[0][1], ics_batch[1])
#             loss_bcs_value = self.loss_bcs(bcs_batch[0][0], bcs_batch[0][1], bcs_batch[1])
#             loss_res_value = self.loss_res(res_batch[0][0], res_batch[0][1])
#
#             self.loss_log.append(loss_value)
#             self.loss_ics_log.append(loss_ics_value.item())
#             self.loss_bcs_log.append(loss_bcs_value.item())
#             self.loss_res_log.append(loss_res_value.item())
#
#             pbar.set_postfix({'Loss': loss_value,
#                               'loss_ics': loss_ics_value.item(),
#                               'loss_bcs': loss_bcs_value.item(),
#                               'loss_res': loss_res_value.item()})
#
#             # if i % 250 == 0:
#             #     # Предсказанное решение
#             #     T = 11
#             #     P = 11
#             #     P_test = 11
#             #
#             #     x = np.linspace(0, 1, P)
#             #     u = np.sin(np.pi * x)
#             #
#             #     u_test = np.tile(u, (P ** 2, 1))
#             #
#             #     x = np.linspace(0, 1, P_test)
#             #     t = np.linspace(0, 1, P_test)
#             #     TT, XX = np.meshgrid(t, x)
#             #
#             #     y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])
#             #
#             #     # Предсказание модели
#             #     s_pred = self.operator_net(torch.tensor(u_test, dtype=torch.float32),
#             #                                torch.tensor(y_test, dtype=torch.float32)[:, 0],
#             #                                torch.tensor(y_test, dtype=torch.float32)[:, 1])
#             #     S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')
#             #
#             #     for k in range(T - 1):
#             #         u_k = S_pred[:, -1]
#             #         u_test_k = np.tile(u_k, (P_test ** 2, 1))
#             #         s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32),
#             #                                    torch.tensor(y_test, dtype=torch.float32))
#             #         S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
#             #         S_pred = np.hstack([S_pred, S_pred_k])
#             #
#             #     # Точное решение
#             #     Nx = 11
#             #     Nt = 11 * T
#             #
#             #     x = np.linspace(0, 1, Nx)
#             #     t = np.linspace(0, T, Nt)
#             #     TT, XX = np.meshgrid(t, x)
#             #
#             #     S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)
#             #
#             #     # Относительная ошибка
#             #     error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
#             #     print('\nRelative l2 error: {:.3e}\n'.format(error))
#             #
#             #     # 3D Визуализация решений
#             #     Nx = 11
#             #     Nt = 11
#             #
#             #     x = torch.linspace(0, 1, Nx)
#             #     t = torch.linspace(0, 1, Nt)
#             #
#             #     grid = torch.cartesian_prod(x, t)
#             #
#             #     fig = plt.figure(figsize=(16, 12))
#             #     ax = fig.add_subplot(111, projection='3d')
#             #
#             #     xs = grid[:, 0].detach().numpy().reshape(-1)
#             #     ys = grid[:, 1].detach().numpy().reshape(-1)
#             #     zs = s_pred.detach().numpy()
#             #
#             #     ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
#             #     ax.set_title('PI DeepONet solution (3D)')
#             #     ax.set_xlabel("$x$")
#             #     ax.set_ylabel("$t$")
#             #
#             #     plt.show()
#
#     def predict_s(self, U_star, Y_star):
#         s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return s_pred
#
#     def predict_res(self, U_star, Y_star):
#         r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return r_pred
#
#
# # Пример использования
# branch_layers = [21, 200, 200]
# trunk_layers = [2, 200, 200]
#
# deepONet = PI_DeepONet(branch_layers, trunk_layers)
#
# # Dummy input для тестирования
# u_dummy = torch.randn(32, 21)
# t_dummy = torch.randn(32)
# x_dummy = torch.randn(32)
#
# output = deepONet.operator_net(u_dummy, t_dummy, x_dummy)
# print(output)
# print(output.shape)


# import torch
# import torch.nn as nn
# import itertools
#
#
# class UnifiedDeepONet(nn.Module):
#     def __init__(self, branch_layers, trunk_layers, activation_branch=torch.tanh, activation_trunk=torch.tanh):
#         super(UnifiedDeepONet, self).__init__()
#
#         self.branch_net = self._build_mlp(branch_layers, activation_branch)
#         self.trunk_net = self._build_mlp(trunk_layers, activation_trunk)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#     def _build_mlp(self, layers, activation):
#         mlp = nn.ModuleList()
#         activations = []
#
#         # Initialize the layers with Xavier initialization
#         for i in range(len(layers) - 1):
#             layer = nn.Linear(layers[i], layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             mlp.append(layer)
#             activations.append(activation)
#
#         # Additional networks U1 and U2
#         U1 = nn.Linear(layers[0], layers[1])
#         U2 = nn.Linear(layers[0], layers[1])
#
#         nn.init.xavier_normal_(U1.weight)
#         nn.init.zeros_(U1.bias)
#         nn.init.xavier_normal_(U2.weight)
#         nn.init.zeros_(U2.bias)
#
#         return mlp, activations, U1, U2
#
#     def forward_mlp(self, mlp, activations, U1, U2, x):
#         U = activations[0](U1(x))
#         V = activations[0](U2(x))
#
#         for i in range(len(mlp) - 1):
#             outputs = activations[i](mlp[i](x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#
#         x = mlp[-1](x)
#         return x
#
#     def forward(self, u, t, x):
#         y = torch.stack([t, x], dim=-1)
#         B = self.forward_mlp(self.branch_net[0], self.branch_net[1], self.branch_net[2], self.branch_net[3], u)
#         T = self.forward_mlp(self.trunk_net[0], self.trunk_net[1], self.trunk_net[2], self.trunk_net[3], y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
#
# # Пример использования объединённого класса
# net = UnifiedDeepONet(
#     branch_layers=[2, 100, 100, 100, 1],
#     trunk_layers=[2, 100, 100, 100, 1]
# )
#
# # Пример прохода данных через сеть
# u = torch.randn(10, 2)  # пример входа для ветвления сети
# t = torch.randn(10)  # пример входа для стволовой сети
# x = torch.randn(10)  # пример входа для стволовой сети
#
# output = net(u, t, x)
# print(output)


# # PINN wave full
#
# import torch
# import torch.nn as nn
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from numpy import linalg as LA
# from math import pi
#
#
# def nn_autograd(model, points, var=0, axis=[0]):
#     points.requires_grad = True
#     fi = model(points)[:, var].sum(0)
#     for ax in axis:
#         grads, = torch.autograd.grad(fi, points, create_graph=True)
#         fi = grads[:, ax].sum()
#     gradient_full = grads[:, axis[-1]].reshape(-1, 1)
#     return gradient_full
#
#
# def wave_equation(grid, a=1/4):
#     u_xx = nn_autograd(model, grid, axis=[0, 0])
#     u_tt = nn_autograd(model, grid, axis=[1, 1])
#     f = a * u_tt - u_xx
#     return f
#
#
# def norm_loss(operator):
#     return torch.sum(torch.mean((operator) ** 2, 0))
#
#
# def MSE_loss(u_true, u_pred):
#     return torch.mean((u_true - u_pred) ** 2)
#
#
# # Dirichlet BC
#
# def Dir_BC_loss(lst_bnd):
#     #     b_loss_1 = torch.mean((model(lst_bnd[0][0]) - lst_bnd[0][1]) ** 2)
#     #     b_loss_2 = torch.mean((nn_autograd(model, lst_bnd[1][0], axis=[1]) - lst_bnd[1][1]) ** 2)
#     #     b_loss_3 = torch.mean((model(lst_bnd[2][0]) - lst_bnd[2][1]) ** 2)
#     #     b_loss_4 = torch.mean((model(lst_bnd[3][0]) - lst_bnd[3][1]) ** 2)
#     #     b_loss = b_loss_1 + b_loss_2 + b_loss_3 + b_loss_4
#
#     b_loss_1 = MSE_loss(model(lst_bnd[0][0]), lst_bnd[0][1])
#     b_loss_2 = MSE_loss(nn_autograd(model, lst_bnd[1][0], axis=[1]), lst_bnd[1][1])
#
#     b_loss_3 = MSE_loss(model(lst_bnd[2][0]), lst_bnd[2][1])
#     b_loss_4 = MSE_loss(model(lst_bnd[3][0]), lst_bnd[3][1])
#     b_loss = b_loss_1 + b_loss_2 + b_loss_3 + b_loss_4
#
#     return b_loss
#
#
# x_grid = np.linspace(0, 1, 11)
# t_grid = np.linspace(0, 1, 11)
#
# x = torch.from_numpy(x_grid)
# t = torch.from_numpy(t_grid)
#
# grid = torch.cartesian_prod(x, t).float()
#
# # 1. u(x, 0) = sin(pi*x) |==> t = 0
# b1 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
# u_b1 = torch.sin(torch.pi * b1[:, 0])
#
# # 2. u_t(x, 0) = 0 |==> t = 0
# b2 = torch.cartesian_prod(x, torch.from_numpy(np.array([0], dtype=np.float64))).float()
# u_b2 = torch.from_numpy(np.zeros(len(b2), dtype=np.float64))
#
# # 3. u(0, t) = 0 |==> x = 0
# b3 = torch.cartesian_prod(torch.from_numpy(np.array([0], dtype=np.float64)), t).float()
# u_b3 = torch.from_numpy(np.zeros(len(b3), dtype=np.float64))
#
# # 4. u(1, t) = 0 |==> x = 1
# b4 = torch.cartesian_prod(torch.from_numpy(np.array([1], dtype=np.float64)), t).float()
# u_b4 = torch.from_numpy(np.zeros(len(b4), dtype=np.float64))
#
# u_b_all = [[b1, u_b1], [b2, u_b2], [b3, u_b3], [b4, u_b4]]
#
# print('b1 =', b1, '\nb1.shape =', b1.shape,
#       '\nb2 =', b2, '\nb2.shape =', b2.shape,
#       '\nb3 =', b3, '\nb3.shape =', b3.shape,
#       '\nb4 =', b4, '\nb4.shape =', b4.shape)
#
# print('u_b1 =', u_b1, '\nu_b1.shape =', u_b1.shape,
#       '\nu_b2 =', u_b2, '\nu_b2.shape =', u_b2.shape,
#       '\nu_b3 =', u_b3, '\nu_b3.shape =', u_b3.shape,
#       '\nu_b4 =', u_b4, '\nu_b4.shape =', u_b4.shape)
#
# neurons = 100
#
# model = torch.nn.Sequential(
#             torch.nn.Linear(2, neurons),
#             torch.nn.Tanh(),
#             torch.nn.Linear(neurons, neurons),
#             torch.nn.Tanh(),
#             torch.nn.Linear(neurons, 1)
#             )
#
# optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
#
# lu, lb = [100, 1]
#
# epochs = 10000
# for i in range(epochs + 1):
#     optimizer.zero_grad()
#     loss = lu * norm_loss(wave_equation(grid)) + lb * Dir_BC_loss(u_b_all)
#     # loss = lu * norm_loss(wave_equation(grid)) + lb * Neum_BC_loss(u_b_all)
#     loss.backward()
#     optimizer.step()
#     if i % 500 == 0:
#         print(f"Epoch {i}/{epochs}: Loss = {loss.item()}")
#
# # Dirichlet BC
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# xs = grid[:, 0].detach().numpy().reshape(-1)
# ys = grid[:, 1].detach().numpy().reshape(-1)
# zs = model(grid).detach().numpy().reshape(-1)
#
# ax.plot_trisurf(xs, ys, zs, cmap=cm.jet, linewidth=0.2, alpha=1)
#
# ax.set_title("Dirichlet BC")
# ax.set_xlabel("$x$")
# ax.set_ylabel("$t$")
#
# plt.show()


# import torch
# import torch.nn as nn
# import numpy as np
#
#
# # Определяем инвариантные функции
# class UniVariateFunction(nn.Module):
#     def __init__(self, output_size):
#         super(UniVariateFunction, self).__init__()
#         self.linear = nn.Linear(1, output_size)
#
#     def forward(self, x):
#         x = self.linear(x)
#         return torch.sin(x)  # Используем синусоиду как функцию активации
#
#
# # Определяем модель KAN
# class KAN(nn.Module):
#     def __init__(self):
#         super(KAN, self).__init__()
#         self.phi = nn.ModuleList([UniVariateFunction(1) for _ in range(2)])  # Phi функции для переменных x и y
#         self.Phi = nn.Linear(2, 1)  # Phi функция для комбинации вывода
#
#     def forward(self, x):
#         x1, x2 = x[:, 0], x[:, 1]
#         x1 = self.phi[0](x1.view(-1, 1))
#         x2 = self.phi[1](x2.view(-1, 1))
#         out = torch.cat((x1, x2), dim=1)
#         out = self.Phi(out)
#         return out
#
#
# # Генерируем простой набор данных
# x = torch.linspace(-np.pi, np.pi, 200)
# y = torch.linspace(-np.pi, np.pi, 200)
# X, Y = torch.meshgrid(x, y)
# Z = torch.sin(X) + torch.cos(Y)
#
# # Достаем "вход" модели
# inputs = torch.stack([X.flatten(), Y.flatten()], dim=1)
# model = KAN()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# # Тренируем
# for epoch in range(1000):
#     optimizer.zero_grad()
#     outputs = model(inputs)
#     loss = criterion(outputs, Z.flatten())
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 20 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')


# import torch
#
# # a = {'3': [4, 5, 1], '2': [3, 5], '1': [6, 8, 1, 6,7, 9, 0]}
# # print(a)
# # l_max_var = len(max(a.values(), key=lambda x: len(x)))
# # print(l_max_var)
# #
# # for i in a:
# #     new_tensor = torch.full((l_max_var,), int(i))
# #     print(new_tensor)
#
# torch.manual_seed(0)
# a = torch.rand(10)
# b = torch.randn(10)
# a1 = torch.rand(5)
# b1 = torch.randn(5)
# a2 = torch.rand(2)
# b2 = torch.randn(2)
# print(a, b, a1, b1, a2, b2, sep='\n')


# a = None
# b = 2
#
# if a is not None:
#     print(b)
# else:
#     print(b + 10)


# class Plots(Callback):
#     """Class for ploting solutions."""
#     def __init__(self,
#                  print_every: Union[int, None] = 500,
#                  save_every: Union[int, None] = 500,
#                  title: str = None,
#                  img_dir: str = None,
#                  method: str = 'PINN',
#                  u: torch.tensor = None):
#         """
#         Args:
#             print_every (Union[int, None], optional): print plots after every *print_every* steps. Defaults to 500.
#             save_every (Union[int, None], optional): save plots after every *print_every* steps. Defaults to 500.
#             title (str, optional): plots title. Defaults to None.
#             img_dir (str, optional): directory title where plots are being saved. Defaults to None.
#         """
#         super().__init__()
#         self.print_every = print_every if print_every is not None else 0.1
#         self.save_every =  save_every if save_every is not None else 0.1
#         self.title = title
#         self.img_dir = img_dir
#         self.method = method
#         self.u = u
#
#     def _print_nn(self):
#         """
#         Solution plot for *NN, autograd* mode.
#
#         """
#
#         # Original
#         # try:
#         #     nvars_model = self.net[-1].out_features
#         # except:
#         #     nvars_model = self.net.model[-1].out_features
#
#         # Changes for KANs
#         # try:
#         #     nvars_model = self.net[-1].out_features
#         # except:
#         #     try:
#         #         nvars_model = self.net.model[-1].out_features
#         #     except:
#         #         try:
#         #             nvars_model = self.net.layers[-1].out_features
#         #         except:
#         #             nvars_model = self.net.layers[-1].output_dim
#
#         attributes = [['model', 'out_features'],
#                       ['layers', 'out_features'],
#                       ['layers', 'output_dim']]
#
#         nvars_model = None
#
#         for attribute in attributes:
#             try:
#                 nvars_model = getattr(getattr(self.net, attribute[0])[-1], attribute[-1])
#                 break
#             except AttributeError:
#                 pass
#
#         if nvars_model is None and self.method == 'PINN':
#             nvars_model = self.net[-1].out_features
#         else:
#             if hasattr(self.net, 'get_out_features'):
#                 nvars_model = self.net.get_out_features()
#
#         nparams = self.grid.shape[1]
#         fig = plt.figure(figsize=(15, 8))
#         for i in range(nvars_model):
#             if nparams == 1:
#                 ax1 = fig.add_subplot(1, nvars_model, i + 1)
#                 if self.title is not None:
#                     ax1.set_title(self.title + ' variable {}'.format(i))
#                 if self.method == 'PINN':
#                     ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
#                                 self.net(self.grid)[:, i].detach().cpu().numpy())
#                 elif self.method == 'PI_DeepONet':
#                     Nx = 10
#                     Nt = 10
#
#                     x = torch.linspace(0, 1, Nx)
#                     t = torch.linspace(0, 1, Nt)
#
#                     self.grid = torch.cartesian_prod(x, t)
#
#                     self.u = torch.from_numpy(np.tile(self.u, (Nx ** 2, 1)))
#
#                     ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
#                                 self.net(self.u, self.grid).detach().cpu().numpy())
#
#             else:
#                 ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
#                 if self.title is not None:
#                     ax1.set_title(self.title + ' variable {}'.format(i))
#                 if self.method == 'PINN':
#                     # a = self.grid[:, 0]
#                     # b = self.grid[:, 1]
#                     # c = self.net(self.grid)[:, i].detach().cpu().numpy()
#                     ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(),
#                                      self.grid[:, 1].detach().cpu().numpy(),
#                                      self.net(self.grid)[:, i].detach().cpu().numpy(),
#                                      cmap=cm.jet, linewidth=0.2, alpha=1)
#                 elif self.method == 'PI_DeepONet':
#                     # a = self.grid[:, 0]
#                     # b = self.grid[:, 1]
#                     # c = self.net(self.u, self.grid).detach().cpu().numpy()
#                     # er = self.grid[:, 1].detach().cpu().numpy().reshape(-1)
#
#                     Nx = 10
#                     Nt = 10
#
#                     x = torch.linspace(0, 1, Nx)
#                     t = torch.linspace(0, 1, Nt)
#
#                     self.grid = torch.cartesian_prod(x, t)
#
#                     self.u = torch.from_numpy(np.tile(self.u, (Nx ** 2, 1)))
#
#                     ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy().reshape(-1),
#                                      self.grid[:, 1].detach().cpu().numpy().reshape(-1),
#                                      self.net(self.u, self.grid).detach().cpu().numpy(),
#                                      cmap=cm.jet, linewidth=0.2, alpha=1)
#
#                 ax1.set_xlabel("x1")
#                 ax1.set_ylabel("x2")
#
#     def _print_mat(self):
#         """
#         Solution plot for mat mode.
#         """
#
#         nparams = self.grid.shape[0]
#         nvars_model = self.net.shape[0]
#         fig = plt.figure(figsize=(15, 8))
#         for i in range(nvars_model):
#             if nparams == 1:
#                 ax1 = fig.add_subplot(1, nvars_model, i+1)
#                 if self.title is not None:
#                     ax1.set_title(self.title+' variable {}'.format(i))
#                 ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
#                             self.net[i].detach().cpu().numpy().reshape(-1))
#             else:
#                 ax1 = fig.add_subplot(1, nvars_model, i+1, projection='3d')
#
#                 if self.title is not None:
#                     ax1.set_title(self.title+' variable {}'.format(i))
#                 ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
#                             self.grid[1].detach().cpu().numpy().reshape(-1),
#                             self.net[i].detach().cpu().numpy().reshape(-1),
#                             cmap=cm.jet, linewidth=0.2, alpha=1)
#             ax1.set_xlabel("x1")
#             ax1.set_ylabel("x2")
#
#     def _dir_path(self, save_dir: str) -> str:
#         """ Path for save figures.
#
#         Args:
#             save_dir (str): directory where saves in
#
#         Returns:
#             str: directory where saves in
#         """
#
#         if save_dir is None:
#             try:
#                 img_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'img')
#             except:
#                 current_dir = globals()['_dh'][0]
#                 img_dir = os.path.join(os.path.dirname(current_dir), 'img')
#
#             if not os.path.isdir(img_dir):
#                 os.mkdir(img_dir)
#             directory = os.path.abspath(os.path.join(img_dir,
#                                         str(datetime.datetime.now().timestamp()) + '.png'))
#         else:
#             if not os.path.isdir(save_dir):
#                 os.mkdir(save_dir)
#             directory = os.path.join(save_dir,
#                                      str(datetime.datetime.now().timestamp()) + '.png')
#         return directory
#
#     def solution_print(
#         self):
#         """ printing or saving figures.
#         """
#         print_flag = self.model.t % self.print_every == 0
#         save_flag = self.model.t % self.save_every == 0
#
#         if print_flag or save_flag:
#             self.net = self.model.net
#             self.grid = self.model.solution_cls.grid
#             if self.model.mode == 'mat':
#                 self._print_mat()
#             else:
#                 self._print_nn()
#             if save_flag:
#                 directory = self._dir_path(self.img_dir)
#                 plt.savefig(directory)
#             if print_flag:
#                 plt.show()
#             plt.close()
#
#     def on_epoch_end(self, logs=None):
#         self.solution_print()


# from mpl_toolkits.mplot3d import Axes3D
#
# import matplotlib.pyplot as plt
# import numpy as np
# from Equation import Expression
#
#
# x = np.arange(0,100,0.01)
# y = np.arange(0,100,0.01)
# x2 = np.append(0,x.flatten())
# y2 = np.append(0,y.flatten())
# z = x2 + y2
# print(z)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
# plt.show()


from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np

import torch

torch.manual_seed(1)

# Создаем два тензора размерностью (50, 1)
tensor1 = torch.rand((5, 1))  # Например, случайный тензор
print(tensor1)
tensor1 = torch.rand((5, 1))  # Еще один случайный тензор
print(tensor1)
tensor1 = torch.rand((5, 1))  # Еще один случайный тензор

print(tensor1)
# print(tensor2)
# print(tensor3)

# Объединяем два тензора по второму измерению (dim=1)
merged_tensor = torch.cat((tensor1, tensor1), dim=1)

# # Выводим размеры и сам новый тензор
# print(merged_tensor.shape)  # Ожидаемый вывод: torch.Size([50, 2])
# print(merged_tensor)


# x = np.arange(0,100,1)
# y = np.arange(0,100,1)
# x2 = np.append(0,x.flatten())
# y2 = np.append(0,y.flatten())
#
# x2,y2 = np.meshgrid(x2,y2) #This is what you were missing
#
# z = x2 + y2
#
# fig = plt.figure(figsize=(12,12))
# ax = fig.add_subplot(projection='3d')
# ax.plot_trisurf(x2.flatten(), y2.flatten(), z.flatten(), linewidth=0.2, antialiased=True) #flatten all the arrays here
#
#
# plt.show()


# # Дипонет
# class MLP(nn.Module):
#     def __init__(self, layers, activation=F.relu):
#         super(MLP, self).__init__()
#         self.layers = nn.ModuleList()
#         self.activation = activation
#
#         # Initialize the layers with Xavier initialization
#         for i in range(len(layers) - 1):
#             layer = nn.Linear(layers[i], layers[i + 1])
#
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.layers.append(layer)
#
#         # Additional networks U1, b1 and U2, b2
#         self.U1 = nn.Linear(layers[0], layers[1])
#         self.U2 = nn.Linear(layers[0], layers[1])
#
#         nn.init.xavier_normal_(self.U1.weight)
#         nn.init.zeros_(self.U1.bias)
#
#         nn.init.xavier_normal_(self.U2.weight)
#         nn.init.zeros_(self.U2.bias)
#
#     def forward(self, x):
#         U = self.activation(self.U1(x))
#         V = self.activation(self.U2(x))
#
#         for layer in self.layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#
#         x = self.layers[-1](x)
#         return x
#
#
# class PI_DeepONet(nn.Module):
#     def __init__(self,
#                  branch_net=None,
#                  branch_layers=[10, 100, 100, 100, 100],
#                  inputs_branch=10,
#                  trunk_net=None,
#                  trunk_layers=[2, 100, 100, 100, 1],
#                  inputs_trunk=2,
#                  neurons=100):
#         super(PI_DeepONet, self).__init__()
#
#         # Using custom MLP model
#         # self.branch_net = MLP(branch_layers, activation=torch.tanh)
#         # self.trunk_net = MLP(trunk_layers, activation=torch.tanh)
#         #
#         # self.out_features_branch = self.branch_net.layers[-1].out_features
#         # self.out_features_trunk = self.trunk_net.layers[-1].out_features
#         # self.out_features = min(self.out_features_branch, self.out_features_trunk)
#
#         self.branch_net = branch_net
#         self.trunk_net = trunk_net
#
#         self.out_features_branch = self.branch_net[-1].out_features
#         self.out_features_trunk = self.trunk_net[-1].out_features
#         self.out_features = min(self.out_features_branch, self.out_features_trunk)
#
#     def forward(self, u, grid):
#         outputs = self.branch_net(u) * self.trunk_net(grid)
#         return torch.sum(outputs, dim=-1)
#
#     def get_out_features(self):
#         return self.out_features
#
#     # def operator_net(self, u, grid):
#     #     # g_sh = grid.shape
#     #     # a = torch.transpose(grid, 0, 1)
#     #     # x, t = a
#     #     # y = torch.stack([x, t], dim=-1)
#     #     B = self.branch_net(u)
#     #     T = self.trunk_net(grid)
#     #     outputs = torch.sum(B * T, dim=-1)
#     #     return outputs
#
#     # def forward(self, x: torch.Tensor, update_grid=False):
#     #     for layer in self.model:
#     #         if update_grid:
#     #             layer.update_grid(x)
#     #         x = layer(x)
#     #     return x
#
#
# # class PI_DeepONet(nn.Module):
# #     def __init__(self, branch_layers, trunk_layers):
# #         super(PI_DeepONet, self).__init__()
# #         self.branch_net = MLP(branch_layers, activation=torch.tanh)
# #         self.trunk_net = MLP(trunk_layers, activation=torch.tanh)
# #
# #         t, x = grid
# #         y = torch.stack([t, x], dim=-1)
# #         B = self.branch_net(u)
# #         T = self.trunk_net(y)
# #         outputs = torch.sum(B * T, dim=-1)
# #         return outputs
# #
# #     def forward(self, u, grid):


# ПО ПОВОДУ СОЛВЕРА
#
# по сути в солвере задана генерация данных для подсети trunk модели PI_DeepoNet
# то есть реализована сетка grid аналогичная сетке y для модели trunk: trunk_net(y)
# получается что основная задача это реализация генерации данных для подсети branch
# таким образом чтобы итоговая реализация модели (которая приближает оператор или другими словами диффур)
# выглядела так: G(u)(y) ~ branch_net(u) * trunk_net(y)
#
# на данный момент в модель подаётся только сетка grid (далее будет обозначаться через игрек: y)
# но не подаются данные с функциями u
# сборка всех частей (решателя и данных) происходит на этапе написания экспериментов для отдельных уравнений
# то есть реализацию данных с функциями можно сделать отдельно в модуле data.py
# и далее подавать эти данные в модель решателя в модуле model.py
#
# но в model.py всё сделано так чтобы в модель можно было подавать только сетку но не данные с функциями
# основной вопрос: как реализовать модель без изменения внутренней структуры солвера
# ведь есть две основные проблемы которые надо решить
# первая: сделать так чтобы модель учитывала данные для branch сети (матрицы функций u)
# вторая: на данный момент лоссы не совпадают с тем как они задаются для PI_DeepONet
# нужно адаптировать лоссы для дипонета
#
# как решить эти проблемы без изменения структуры солвера я не понимаю
# но ведь можно изменить структуру солвера самым простым тупым способом
# а именно добавлять части необходимые для дипонета с помощью простого условного оператора
# причём можно сделать PINN дефолтным методом и тогда не придётся менять вообще никакие эксперименты
# по крайней мере я не вижу причин по которым этот вариант не должен сработать
#
# что подлежит изменению:
# 1) генерация данных (в модуле data.py)
# 2) ошибки (точнее не сам класс с ошибками а скорее применение этих ошибок в модуле solution.py)
# все остальные необходимые части присутствуют и нужно ими пользоваться
#
# какие модули следует изменить в соответствии с пунктом выше:
# 1) data.py (генерация значений функций для branch_net)
# 2) solution.py --> eval.py (лоссы)
# 3) возможно losses.py (лучше не менять а использовать как есть в eval.py с помощью разных аргументов)
# вообще говоря все изменения требуются ИСКЛЮЧИТЕЛЬНО по причине особой организации данных u, y (t, x)
# в модели PI_DeepONet
# то есть вместо команды model(grid) которая сейчас используется должно быть что-то вроде
# model(u, y) или при желании model(u)(y)


# f = lambda x: x ** 2
# print([f(i) for i in range(10)])












































