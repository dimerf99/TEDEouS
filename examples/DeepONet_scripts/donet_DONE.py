import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import kan
import efficient_kan
import fastkan

from torch.utils import data
from scipy.interpolate import griddata

import itertools
import matplotlib.pyplot as plt

from tqdm import trange


# Equation dict variant

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
#             # normal initialization
#             # nn.init.xavier_normal_(layer.weight)  # best for PI DeepOnet
#             # nn.init.kaiming_normal_(layer.weight)
#             # nn.init.trunc_normal_(layer.weight)
#
#             #  uniform initialization
#             # nn.init.xavier_uniform_(layer.weight)
#             # nn.init.kaiming_uniform_(layer.weight)
#
#             nn.init.zeros_(layer.bias)
#             self.layers.append(layer)
#
#         # Additional networks U1, b1 and U2, b2
#         self.U1 = nn.Linear(layers[0], layers[1])
#         self.U2 = nn.Linear(layers[0], layers[1])
#
#         # normal initialization
#         nn.init.xavier_normal_(self.U1.weight)  # best for PI DeepOnet
#         # nn.init.kaiming_normal_(self.U1.weight)
#         # nn.init.trunc_normal_(self.U1.weight)
#
#         # uniform initialization
#         # nn.init.xavier_uniform_(self.U1.weight)
#         # nn.init.kaiming_uniform_(self.U1.weight)
#         nn.init.zeros_(self.U1.bias)
#
#         # normal initialization
#         nn.init.xavier_normal_(self.U2.weight)  # best for PI DeepOnet
#         # nn.init.kaiming_normal_(self.U2.weight)
#         # nn.init.trunc_normal_(self.U2.weight)
#
#         # uniform initialization
#         # nn.init.xavier_uniform_(self.U2.weight)
#         # nn.init.kaiming_uniform_(self.U2.weig
#         # t)
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
#     def compute_deivative(self, u, t, x, orders):
#         u.requires_grad_(True)
#         t.requires_grad_(True)
#         x.requires_grad_(True)
#
#         s = self.operator_net(u, t, x)
#         deriv = s
#
#         for order in orders:
#             if order == 0:
#                 deriv = torch.autograd.grad(deriv, x, torch.ones_like(deriv), create_graph=True)[0]
#             elif order == 1:
#                 deriv = torch.autograd.grad(deriv, t, torch.ones_like(deriv), create_graph=True)[0]
#
#         return deriv
#
#     def residual_net(self, u, t, x, equation_dict):
#         residual = 0
#         for key, value in equation_dict.items():
#             coeff, orders, power = [value[k] for k in value]
#             term = coeff * self.compute_deivative(u, t, x, orders) ** power
#             residual += term
#         return residual
#
#     # losses ###############################################################################
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
#     def loss_bcs(self, u, y, outputs):
#         s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
#         s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])
#
#         loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
#         loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
#         loss_s_bc = loss_s_bc1 + loss_s_bc2
#         return loss_s_bc
#
#     def loss_res(self, u, y, equation_dict):
#         pred = self.residual_net(u, y[:, 0], y[:, 1], equation_dict)
#         loss = torch.mean(pred ** 2)
#         return loss
#
#     def loss(self, ics_batch, bcs_batch, res_batch, equation_dict):
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
#         loss_res = self.loss_res(u_res, y_res, equation_dict)
#
#         # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 1.0  # есть искажения, относительно быстрая сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 1.0  # есть искажения, относительно быстрая сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 10.0, 10.0, 2.0  # есть искажения, медленная сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 5.0  # к 1500 мало искажений, средняя сходимость (лучший)
#         # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 10.0  # есть искажения, медленная сходимость
#         lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 4.0  # мало искажений, медленная сходимость (лучший)
#         # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 1.0  # есть искажения, медленная сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 5.0  # мало искажений, медленная сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость
#         # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость
#
#         loss = lambda_ics * loss_ics + lambda_bcs * loss_bcs + lambda_res * loss_res
#         return loss
#
#     ########################################################################################
#
#     def step(self, ics_batch, bcs_batch, res_batch, equation_dict):
#         self.optimizer.zero_grad()
#         loss = self.loss(ics_batch, bcs_batch, res_batch, equation_dict)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()
#
#     def train(self, ics_dataset, bcs_dataset, res_dataset, equation_dict, nIter=10000):
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
#             loss_value = self.step(ics_batch, bcs_batch, res_batch, equation_dict)
#             loss_ics_value = self.loss_ics(ics_batch[0][0], ics_batch[0][1], ics_batch[1])
#             loss_bcs_value = self.loss_bcs(bcs_batch[0][0], bcs_batch[0][1], bcs_batch[1])
#             loss_res_value = self.loss_res(res_batch[0][0], res_batch[0][1], equation_dict)
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
#             if i % 250 == 0:
#                 # Предсказанное решение
#                 T = 11
#                 P = 11
#                 P_test = 11
#
#                 x = np.linspace(0, 1, P)
#                 u = np.sin(np.pi * x)
#
#                 u_test = np.tile(u, (P ** 2, 1))
#
#                 x = np.linspace(0, 1, P_test)
#                 t = np.linspace(0, 1, P_test)
#                 TT, XX = np.meshgrid(t, x)
#
#                 y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])
#
#                 # Предсказание модели
#                 s_pred = self.operator_net(torch.tensor(u_test, dtype=torch.float32),
#                                            torch.tensor(y_test, dtype=torch.float32)[:, 0],
#                                            torch.tensor(y_test, dtype=torch.float32)[:, 1])
#                 S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')
#
#                 for k in range(T - 1):
#                     u_k = S_pred[:, -1]
#                     u_test_k = np.tile(u_k, (P_test ** 2, 1))
#                     s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32),
#                                                torch.tensor(y_test, dtype=torch.float32))
#                     S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
#                     S_pred = np.hstack([S_pred, S_pred_k])
#
#                 # Точное решение
#                 Nx = 11
#                 Nt = 11 * T
#
#                 x = np.linspace(0, 1, Nx)
#                 t = np.linspace(0, T, Nt)
#                 TT, XX = np.meshgrid(t, x)
#
#                 S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)
#
#                 # Относительная ошибка
#                 error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
#                 print('\nRelative l2 error: {:.3e}\n'.format(error))
#
#                 # 3D Визуализация решений
#                 Nx = 11
#                 Nt = 11
#
#                 x = torch.linspace(0, 1, Nx)
#                 t = torch.linspace(0, 1, Nt)
#
#                 grid = torch.cartesian_prod(x, t)
#
#                 fig = plt.figure(figsize=(16, 12))
#                 ax = fig.add_subplot(111, projection='3d')
#
#                 xs = grid[:, 0].detach().numpy().reshape(-1)
#                 ys = grid[:, 1].detach().numpy().reshape(-1)
#                 zs = s_pred.detach().numpy()
#
#                 ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
#                 ax.set_title('PI DeepONet solution (3D)')
#                 ax.set_xlabel("$x$")
#                 ax.set_ylabel("$t$")
#
#                 plt.show()
#
#     def predict_s(self, U_star, Y_star):
#         s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return s_pred
#
#     def predict_res(self, U_star, Y_star):
#         r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
#         return r_pred


# Equation in residual variant

class MLP(nn.Module):
    def __init__(self, layers, activation=F.relu):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        # Initialize the layers with Xavier initialization
        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)

        # Additional networks U1, b1 and U2, b2
        self.U1 = nn.Linear(layers[0], layers[1])
        self.U2 = nn.Linear(layers[0], layers[1])

        nn.init.xavier_normal_(self.U1.weight)
        nn.init.zeros_(self.U1.bias)
        nn.init.xavier_normal_(self.U2.weight)
        nn.init.zeros_(self.U2.bias)

    def forward(self, x):
        a = x
        U = self.activation(self.U1(x))
        V = self.activation(self.U2(x))

        for layer in self.layers[:-1]:
            outputs = self.activation(layer(x))
            x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)

        x = self.layers[-1](x)
        return x


m = 10
neurons = m
# MLP and PI DeepONet should be
class PI_DeepONet(nn.Module):
    def __init__(self, inputs_branch=11, inputs_trunk=2, neurons=200):
        super(PI_DeepONet, self).__init__()

        branch_layers = [inputs_branch, neurons, neurons, neurons, neurons]
        trunk_layers = [inputs_trunk, neurons, neurons, neurons, neurons]

        # neurons = [50, 75, 100], your choose

        # Лучше использовать модели типа torch.nn.Sequential, чем кастомный MLP
        # 1. Они лучше работают
        # 2. Они используются в солвере

        self.branch_net = MLP(branch_layers, activation=torch.tanh)
        # self.branch_net = torch.nn.Sequential(
        #     torch.nn.Linear(inputs_branch, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons)
        # )
        # self.branch_net = efficient_kan.KAN(
        #     [inputs_branch, neurons, neurons],
        #     grid_size=3,
        #     spline_order=3,
        #     scale_noise=0.1,
        #     scale_base=1.0,
        #     scale_spline=1.0,
        #     base_activation=torch.nn.SiLU,
        #     grid_eps=0.02,
        #     grid_range=[-1, 1]
        # )

        self.trunk_net = MLP(trunk_layers, activation=torch.tanh)
        # self.trunk_net = torch.nn.Sequential(
        #     torch.nn.Linear(inputs_trunk, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(neurons, neurons)
        # )
        # self.trunk_net = efficient_kan.KAN(
        #     [inputs_trunk, neurons, neurons, 1],
        #     grid_size=10,
        #     spline_order=3,
        #     scale_noise=0.1,
        #     scale_base=1.0,
        #     scale_spline=1.0,
        #     base_activation=torch.nn.Tanh,
        #     grid_eps=0.02,
        #     grid_range=[-1, 1]
        # )
        # self.trunk_net = fastkan.FastKAN(
        #     [2, neurons, neurons, neurons, 1],
        #     grid_min=-4.,
        #     grid_max=4.,
        #     num_grids=2,
        #     use_base_update=True,
        #     base_activation=F.tanh,
        #     spline_weight_init_scale=0.05
        # )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # eval.py in solver (???)
    def operator_net(self, u, t, x):
        y = torch.stack([t, x], dim=-1)
        B = self.branch_net(u)
        T = self.trunk_net(y)
        outputs = torch.sum(B * T, dim=-1)
        return outputs

    # eval.py in solver
    def residual_net(self, u, t, x):
        t.requires_grad_(True)
        x.requires_grad_(True)

        s = self.operator_net(u, t, x)
        s_t = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
        s_tt = torch.autograd.grad(s_t, t, torch.ones_like(s_t), create_graph=True)[0]
        s_x = torch.autograd.grad(s, x, torch.ones_like(s), create_graph=True)[0]
        s_xx = torch.autograd.grad(s_x, x, torch.ones_like(s_x), create_graph=True)[0]

        a = 1 / 4
        res = a * s_tt - 1 * s_xx
        return res

    # losses ###############################################################################
    # losses.py in solver
    def loss_ics(self, u, y, outputs):
        # Выделенный ниже кусок был вставлен в losses.py в солвере
        # нужно адаптировать его таким образом чтобы в функции с ошибками не было

        ##############################################################################
        t = y[:, 0]
        x = y[:, 1]

        s_pred = self.operator_net(u, t, x)
        t.requires_grad_(True)
        s = self.operator_net(u, t, x)
        s_t_pred = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
        ##############################################################################

        loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2, 0)  # _loss_bcs
        loss_2 = torch.mean(s_t_pred ** 2)  # _loss_op
        loss = loss_1 + loss_2  # ics coeff will add in general loss-function

        return loss

    # losses.py in solver
    def loss_bcs(self, u, y, outputs):
        s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
        s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])

        loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
        loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
        loss_s_bc = loss_s_bc1 + loss_s_bc2  # bcs coeff will add in general loss-function
        return loss_s_bc

    # losses.py in solver
    def loss_res(self, u, y):
        pred = self.residual_net(u, y[:, 0], y[:, 1])
        loss = torch.mean(pred ** 2)  # res coeff will add in general loss-function
        return loss

    # losses.py in solver
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

        lambda_ics, lambda_bcs, lambda_res = 100.0, 100.0, 1.0  # мало искажений, медленная сходимость (лучший)

        a0 = u_ics == u_bcs

        loss = lambda_ics * loss_ics + lambda_bcs * loss_bcs + lambda_res * loss_res
        return loss

    ########################################################################################

    # closure.py in solver
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
        for i in pbar:
            ics_batch = next(ics_data)
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            loss_value = self.step(ics_batch, bcs_batch, res_batch)
            loss_ics_value = self.loss_ics(ics_batch[0][0], ics_batch[0][1], ics_batch[1])
            loss_bcs_value = self.loss_bcs(bcs_batch[0][0], bcs_batch[0][1], bcs_batch[1])
            loss_res_value = self.loss_res(res_batch[0][0], res_batch[0][1])

            self.loss_log.append(loss_value)
            self.loss_ics_log.append(loss_ics_value.item())
            self.loss_bcs_log.append(loss_bcs_value.item())
            self.loss_res_log.append(loss_res_value.item())

            pbar.set_postfix({'Loss': loss_value,
                              'loss_ics': loss_ics_value.item(),
                              'loss_bcs': loss_bcs_value.item(),
                              'loss_res': loss_res_value.item()})

            if i % 500 == 0:
                # Предсказанное решение
                T = m
                P = m
                P_test = m

                x = np.linspace(0, 1, P)
                u = np.sin(np.pi * x)

                u_test = np.tile(u, (P ** 2, 1))

                x = np.linspace(0, 1, P_test)
                t = np.linspace(0, 1, P_test)
                TT, XX = np.meshgrid(t, x)

                y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])

                # Предсказание модели
                s_pred = self.operator_net(torch.tensor(u_test, dtype=torch.float32),
                                           torch.tensor(y_test, dtype=torch.float32)[:, 0],
                                           torch.tensor(y_test, dtype=torch.float32)[:, 1])
                S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')

                for k in range(T - 1):
                    u_k = S_pred[:, -1]
                    u_test_k = np.tile(u_k, (P_test ** 2, 1))
                    s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32),
                                               torch.tensor(y_test, dtype=torch.float32))
                    S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
                    S_pred = np.hstack([S_pred, S_pred_k])

                # Точное решение
                Nx = m
                Nt = m * T

                x = np.linspace(0, 1, Nx)
                t = np.linspace(0, T, Nt)
                TT, XX = np.meshgrid(t, x)

                S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)

                # Относительная ошибка
                error_l2 = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
                print('\nRelative l2 error: {:.3e}'.format(error_l2))

                error_rmse = np.sqrt((np.mean(S_pred - S_test)) ** 2)
                print('Relative RMSE error: {:.3e}\n'.format(error_rmse))

                # 3D Визуализация решений
                Nx = m
                Nt = m

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

    def predict_s(self, U_star, Y_star):
        s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred

    def predict_res(self, U_star, Y_star):
        r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, batch_size=128, rng_seed=1234):
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
        torch.manual_seed(subkey)
        idx = torch.randperm(self.N)[:self.batch_size]
        s = self.s[idx]
        y = self.y[idx]
        u = self.u[idx]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


# Hyperparameters
length_scale = 0.5
output_scale = 10.0
gp_params = (length_scale, output_scale)
# c = 1.0


def RBF(x1, x2, gp_params):
    length_scale, output_scale = gp_params
    diffs = torch.unsqueeze(x1 / length_scale, 1) - torch.unsqueeze(x2 / length_scale, 0)
    r2 = torch.sum(diffs ** 2, dim=2)
    return output_scale * torch.exp(-0.5 * r2)


def generate_one_gaussian_sample(key, gp_params, N):
    torch.manual_seed(key)
    jitter = 1e-10
    X = torch.linspace(0.0, 1.0, N, dtype=torch.float64).view(-1, 1)
    K = RBF(X, X, gp_params)
    L = torch.linalg.cholesky(K + jitter * torch.eye(N))
    gp_sample = torch.matmul(L, torch.randn(N, dtype=torch.float64))
    return gp_sample


def generate_one_training_data(key, m=100, P=100, Q=100):
    torch.manual_seed(key)
    subkey = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

    # Generate one input sample
    N = 512
    gp_sample = generate_one_gaussian_sample(subkey, gp_params, N)
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
    # x_0 = torch.linspace(0, 1, P).reshape(P, 1)  # change version
    y_ic = torch.hstack([t_0, x_0])

    s_ic = u_fn(x_0)

    # BC training data
    u_bc = u.repeat(P, 1)

    t_bc1 = torch.rand((P, 1))
    # t_bc1 = torch.linspace(0, 1, P).reshape(P, 1)  # change version
    x_bc1 = torch.zeros((P, 1))

    t_bc2 = torch.rand((P, 1))
    # t_bc2 = torch.linspace(0, 1, P).reshape(P, 1)  # change version
    x_bc2 = torch.ones((P, 1))

    y_bc1 = torch.hstack([t_bc1, x_bc1])
    y_bc2 = torch.hstack([t_bc2, x_bc2])
    y_bc = torch.hstack([y_bc1, y_bc2])

    s_bc = torch.zeros((Q, 2))

    # Residual training data
    u_r = u.repeat(Q, 1)
    y_r = torch.rand((Q, 2))
    # y_r = torch.linspace(0, 1, Q).reshape(Q, 1).repeat(1, 2)
    s_r = torch.zeros((Q, 1))

    return u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r


def generate_training_data(key, N, m, P, Q):
    torch.manual_seed(key)
    keys = torch.randint(0, 10 ** 10, (N,), dtype=torch.int64)

    results = [generate_one_training_data(key, m, P, Q) for key in keys]

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


N_test = m
m = m
P_train = m
Q_train = m
N_train = m * 50

# Generate training data
torch.manual_seed(0)  # use different key for generating training data
key_train = torch.randint(0, 10 ** 10, (1, ))
u_ics_train, y_ics_train, s_ics_train, u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = \
    generate_training_data(key_train, N_train, m, P_train, Q_train)

print('u_ics_train =', u_ics_train, '\nu_ics_train.shape =', u_ics_train.shape,
      '\ny_ics_train =', y_ics_train, '\ny_ics_train.shape =', y_ics_train.shape,
      '\ns_ics_train =', s_ics_train, '\ns_ics_train.shape =', s_ics_train.shape)
print('u_bcs_train =', u_bcs_train, '\nu_bcs_train.shape =', u_bcs_train.shape,
      '\ny_bcs_train =', y_bcs_train, '\ny_bcs_train.shape =', y_bcs_train.shape,
      '\ns_bcs_train =', s_bcs_train, '\ns_bcs_train.shape =', s_bcs_train.shape)
print('u_res_train =', u_res_train, '\nu_res_train.shape =', u_res_train.shape,
      '\ny_res_train =', y_res_train, '\ny_res_train.shape =', y_res_train.shape,
      '\ns_res_train =', s_res_train, '\ns_res_train.shape =', s_res_train.shape)

print('\ny_ics_train =', y_ics_train.shape)
print('y_ics_train[0] =', y_ics_train[0].shape)

print('\ny_bcs_train =', y_bcs_train.shape)
print('y_bcs_train[0] =', y_bcs_train[0].shape)

tmp = torch.chunk(y_bcs_train, 2, dim=1)
print(tmp)

print('\ny_res_train =', y_res_train.shape)
print('y_res_train[0] =', y_res_train[0].shape)

# Initialize model

branch_layers = [m, neurons, neurons, neurons, neurons]
trunk_layers = [2, neurons, neurons, neurons, neurons]
model = PI_DeepONet(m, 2, neurons)

print(model, end='\n\n')

# Create data set
batch_size = 2048
ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

print('\nics_dataset =', u_ics_train)
print('ics_dataset.shape =', u_ics_train.shape)
print('\nbcs_dataset =', u_bcs_train)
print('bcs_dataset.shape =', u_bcs_train.shape)
print('\nres_dataset =', u_res_train)
print('res_dataset.shape =', u_res_train.shape)

epochs = 20000

equation = {
    'd2u/dt2**1':
        {
            'coeff': -1/4,
            'd2u/dt2': [1, 1],
            'pow': 1
        },
    '-C*d2u/dx2**1':
        {
            'coeff': 1,
            'd2u/dx2': [0, 0],
            'pow': 1
        }
}

# Train
model.train(ics_dataset, bcs_dataset, res_dataset, nIter=epochs)

# Получение параметров модели
params = model.state_dict()

# Сохранение параметров модели
torch.save(params, 'wave_params_512_done.pth')

# Сохранение логов потерь
np.save('wave_loss_res_512.npy', np.array(model.loss_res_log))
np.save('wave_loss_ics_512.npy', np.array(model.loss_ics_log))
np.save('wave_loss_bcs_512.npy', np.array(model.loss_bcs_log))

# Загрузка модели и логов потерь
flat_params = torch.load('wave_params_512_done.pth')
loss_ics = np.load('wave_loss_ics_512_done.npy')
loss_bcs = np.load('wave_loss_bcs_512_done.npy')
loss_res = np.load('wave_loss_res_512_done.npy')

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
T = 10
P = 10
P_test = 10

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
Nx = 10
Nt = 10 * T

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
Nx = 10
Nt = 10

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











