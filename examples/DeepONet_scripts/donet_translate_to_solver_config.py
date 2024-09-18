# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import pandas as pd
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F

from scipy.interpolate import griddata
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

# Custom models
from tedeous.models import PI_DeepONet, FourierNN, FeedForward, Fourier_embedding
from tedeous.data import DataGenerator, generate_training_data

from tedeous.model import Model
from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

from torch.utils import data
import itertools
from tqdm import trange


# class PI_DeepONet(nn.Module):
#     def __init__(self, branch_layers, trunk_layers, activation=F.relu):
#         super(PI_DeepONet, self).__init__()
#         self.activation = activation
#
#         # Initialize branch network
#         self.branch_layers = nn.ModuleList()
#         for i in range(len(branch_layers) - 1):
#             layer = nn.Linear(branch_layers[i], branch_layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.branch_layers.append(layer)
#         self.branch_U1 = nn.Linear(branch_layers[0], branch_layers[1])
#         self.branch_U2 = nn.Linear(branch_layers[0], branch_layers[1])
#         nn.init.xavier_normal_(self.branch_U1.weight)
#         nn.init.zeros_(self.branch_U1.bias)
#         nn.init.xavier_normal_(self.branch_U2.weight)
#         nn.init.zeros_(self.branch_U2.bias)
#
#         # Initialize trunk network
#         self.trunk_layers = nn.ModuleList()
#         for i in range(len(trunk_layers) - 1):
#             layer = nn.Linear(trunk_layers[i], trunk_layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.trunk_layers.append(layer)
#         self.trunk_U1 = nn.Linear(trunk_layers[0], trunk_layers[1])
#         self.trunk_U2 = nn.Linear(trunk_layers[0], trunk_layers[1])
#         nn.init.xavier_normal_(self.trunk_U1.weight)
#         nn.init.zeros_(self.trunk_U1.bias)
#         nn.init.xavier_normal_(self.trunk_U2.weight)
#         nn.init.zeros_(self.trunk_U2.bias)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#     def _forward_branch(self, x):
#         U = self.activation(self.branch_U1(x))
#         V = self.activation(self.branch_U2(x))
#         for layer in self.branch_layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#         x = self.branch_layers[-1](x)
#         return x
#
#     def _forward_trunk(self, x):
#         U = self.activation(self.trunk_U1(x))
#         V = self.activation(self.trunk_U2(x))
#         for layer in self.trunk_layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#         x = self.trunk_layers[-1](x)
#         return x
#
#     def operator_net(self, u, t, x):
#         y = torch.stack([t, x], dim=-1)
#         B = self._forward_branch(u)
#         T = self._forward_trunk(y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
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
#     def forward(self, u, t, x):
#         res = self.residual_net(u, t, x)
#         return res


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
        U = self.activation(self.U1(x))
        V = self.activation(self.U2(x))

        for layer in self.layers[:-1]:
            outputs = self.activation(layer(x))
            x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)

        x = self.layers[-1](x)
        return x


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
        t = y[:, 0]
        x = y[:, 1]

        s_pred = self.operator_net(u, t, x)
        t.requires_grad_(True)
        s = self.operator_net(u, t, x)
        s_t_pred = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]

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

        # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 1.0  # есть искажения, относительно быстрая сходимость
        # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 1.0  # есть искажения, относительно быстрая сходимость
        # lambda_ics, lambda_bcs, lambda_res = 10.0, 10.0, 2.0  # есть искажения, медленная сходимость
        # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 5.0  # к 1500 мало искажений, средняя сходимость (лучший)
        # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 10.0  # есть искажения, медленная сходимость
        lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 4.0  # мало искажений, медленная сходимость (лучший)
        # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 1.0  # есть искажения, медленная сходимость
        # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 5.0  # мало искажений, медленная сходимость
        # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость
        # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость

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

            if i % 250 == 0:
                # Предсказанное решение
                T = 11
                P = 11
                P_test = 11

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
                Nx = 11
                Nt = 11 * T

                x = np.linspace(0, 1, Nx)
                t = np.linspace(0, T, Nt)
                TT, XX = np.meshgrid(t, x)

                S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)

                # Относительная ошибка
                error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
                print('\nRelative l2 error: {:.3e}\n'.format(error))

                # 3D Визуализация решений
                Nx = 11
                Nt = 11

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


# class PI_DeepONet(nn.Module):
#     def __init__(self, branch_layers, trunk_layers, activation=F.relu):
#         super(PI_DeepONet, self).__init__()
#         self.activation = activation
#
#         # Initialize branch network
#         self.branch_layers = nn.ModuleList()
#         for i in range(len(branch_layers) - 1):
#             layer = nn.Linear(branch_layers[i], branch_layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.branch_layers.append(layer)
#         self.branch_U1 = nn.Linear(branch_layers[0], branch_layers[1])
#         self.branch_U2 = nn.Linear(branch_layers[0], branch_layers[1])
#         nn.init.xavier_normal_(self.branch_U1.weight)
#         nn.init.zeros_(self.branch_U1.bias)
#         nn.init.xavier_normal_(self.branch_U2.weight)
#         nn.init.zeros_(self.branch_U2.bias)
#
#         # Initialize trunk network
#         self.trunk_layers = nn.ModuleList()
#         for i in range(len(trunk_layers) - 1):
#             layer = nn.Linear(trunk_layers[i], trunk_layers[i + 1])
#             nn.init.xavier_normal_(layer.weight)
#             nn.init.zeros_(layer.bias)
#             self.trunk_layers.append(layer)
#         self.trunk_U1 = nn.Linear(trunk_layers[0], trunk_layers[1])
#         self.trunk_U2 = nn.Linear(trunk_layers[0], trunk_layers[1])
#         nn.init.xavier_normal_(self.trunk_U1.weight)
#         nn.init.zeros_(self.trunk_U1.bias)
#         nn.init.xavier_normal_(self.trunk_U2.weight)
#         nn.init.zeros_(self.trunk_U2.bias)
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#
#         self.itercount = itertools.count()
#         self.loss_log = []
#         self.loss_ics_log = []
#         self.loss_bcs_log = []
#         self.loss_res_log = []
#
#     def _forward_branch(self, x):
#         U = self.activation(self.branch_U1(x))
#         V = self.activation(self.branch_U2(x))
#         for layer in self.branch_layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#         x = self.branch_layers[-1](x)
#         return x
#
#     def _forward_trunk(self, x):
#         U = self.activation(self.trunk_U1(x))
#         V = self.activation(self.trunk_U2(x))
#         for layer in self.trunk_layers[:-1]:
#             outputs = self.activation(layer(x))
#             x = torch.mul(outputs, U) + torch.mul((1 - outputs), V)
#         x = self.trunk_layers[-1](x)
#         return x
#
#     def operator_net(self, u, t, x):
#         y = torch.stack([t, x], dim=-1)
#         B = self._forward_branch(u)
#         T = self._forward_trunk(y)
#         outputs = torch.sum(B * T, dim=-1)
#         return outputs
#
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
#     def forward(self, u, t, x):
#         res = self.residual_net(u, t, x)
#         return res

    # def loss_ics(self, u, y, outputs):
    #     t = y[:, 0]
    #     x = y[:, 1]
    #
    #     s_pred = self.operator_net(u, t, x)
    #     t.requires_grad_(True)
    #     s = self.operator_net(u, t, x)
    #     s_t_pred = torch.autograd.grad(s, t, torch.ones_like(s), create_graph=True)[0]
    #
    #     loss_1 = torch.mean((outputs.flatten() - s_pred) ** 2)
    #     loss_2 = torch.mean(s_t_pred ** 2)
    #     loss = loss_1 + loss_2
    #     return loss
    #
    # def loss_bcs(self, u, y, outputs):
    #     s_bc1_pred = self.operator_net(u, y[:, 0], y[:, 1])
    #     s_bc2_pred = self.operator_net(u, y[:, 2], y[:, 3])
    #
    #     loss_s_bc1 = torch.mean((s_bc1_pred - outputs[:, 0]) ** 2)
    #     loss_s_bc2 = torch.mean((s_bc2_pred - outputs[:, 1]) ** 2)
    #     loss_s_bc = loss_s_bc1 + loss_s_bc2
    #     return loss_s_bc
    #
    # def loss_res(self, u, y):
    #     pred = self.residual_net(u, y[:, 0], y[:, 1])
    #     loss = torch.mean(pred ** 2)
    #     return loss
    #
    # def loss(self, ics_batch, bcs_batch, res_batch):
    #     ics_inputs, ics_outputs = ics_batch
    #     bcs_inputs, bcs_outputs = bcs_batch
    #     res_inputs, _ = res_batch
    #
    #     u_ics, y_ics = ics_inputs
    #     u_bcs, y_bcs = bcs_inputs
    #     u_res, y_res = res_inputs
    #
    #     loss_ics = self.loss_ics(u_ics, y_ics, ics_outputs)
    #     loss_bcs = self.loss_bcs(u_bcs, y_bcs, bcs_outputs)
    #     loss_res = self.loss_res(u_res, y_res)
    #
    #     # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 1.0  # есть искажения, относительно быстрая сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 1.0  # есть искажения, относительно быстрая сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 10.0, 10.0, 2.0  # есть искажения, медленная сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 5.0  # к 1500 мало искажений, средняя сходимость (лучший)
    #     # lambda_ics, lambda_bcs, lambda_res = 5.0, 5.0, 10.0  # есть искажения, медленная сходимость
    #     lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 4.0  # мало искажений, медленная сходимость (лучший)
    #     # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 1.0  # есть искажения, медленная сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 5.0  # мало искажений, медленная сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 1.0, 1.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость
    #     # lambda_ics, lambda_bcs, lambda_res = 2.0, 2.0, 10.0  # мало искажений, ОЧЕНЬ медленная сходимость
    #
    #     loss = lambda_ics * loss_ics + lambda_bcs * loss_bcs + lambda_res * loss_res
    #     return loss
    #
    # def step(self, ics_batch, bcs_batch, res_batch):
    #     self.optimizer.zero_grad()
    #     loss = self.loss(ics_batch, bcs_batch, res_batch)
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item()
    #
    # def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
    #     ics_data = iter(ics_dataset)
    #     bcs_data = iter(bcs_dataset)
    #     res_data = iter(res_dataset)
    #
    #     pbar = trange(nIter)
    #     for i in pbar:
    #         ics_batch = next(ics_data)
    #         bcs_batch = next(bcs_data)
    #         res_batch = next(res_data)
    #
    #         loss_value = self.step(ics_batch, bcs_batch, res_batch)
    #         loss_ics_value = self.loss_ics(ics_batch[0][0], ics_batch[0][1], ics_batch[1])
    #         loss_bcs_value = self.loss_bcs(bcs_batch[0][0], bcs_batch[0][1], bcs_batch[1])
    #         loss_res_value = self.loss_res(res_batch[0][0], res_batch[0][1])
    #
    #         self.loss_log.append(loss_value)
    #         self.loss_ics_log.append(loss_ics_value.item())
    #         self.loss_bcs_log.append(loss_bcs_value.item())
    #         self.loss_res_log.append(loss_res_value.item())
    #
    #         pbar.set_postfix({'Loss': loss_value,
    #                           'loss_ics': loss_ics_value.item(),
    #                           'loss_bcs': loss_bcs_value.item(),
    #                           'loss_res': loss_res_value.item()})
    #
    #         if i % 250 == 0:
    #             # Предсказанное решение
    #             T = 11
    #             P = 11
    #             P_test = 11
    #
    #             x = np.linspace(0, 1, P)
    #             u = np.sin(np.pi * x)
    #
    #             u_test = np.tile(u, (P ** 2, 1))
    #
    #             x = np.linspace(0, 1, P_test)
    #             t = np.linspace(0, 1, P_test)
    #             TT, XX = np.meshgrid(t, x)
    #
    #             y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])
    #
    #             # Предсказание модели
    #             s_pred = self.operator_net(torch.tensor(u_test, dtype=torch.float32),
    #                                        torch.tensor(y_test, dtype=torch.float32)[:, 0],
    #                                        torch.tensor(y_test, dtype=torch.float32)[:, 1])
    #             S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')
    #
    #             for k in range(T - 1):
    #                 u_k = S_pred[:, -1]
    #                 u_test_k = np.tile(u_k, (P_test ** 2, 1))
    #                 s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32),
    #                                            torch.tensor(y_test, dtype=torch.float32))
    #                 S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
    #                 S_pred = np.hstack([S_pred, S_pred_k])
    #
    #             # Точное решение
    #             Nx = 11
    #             Nt = 11 * T
    #
    #             x = np.linspace(0, 1, Nx)
    #             t = np.linspace(0, T, Nt)
    #             TT, XX = np.meshgrid(t, x)
    #
    #             S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)
    #
    #             # Относительная ошибка
    #             error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
    #             print('\nRelative l2 error: {:.3e}\n'.format(error))
    #
    #             # 3D Визуализация решений
    #             Nx = 11
    #             Nt = 11
    #
    #             x = torch.linspace(0, 1, Nx)
    #             t = torch.linspace(0, 1, Nt)
    #
    #             grid = torch.cartesian_prod(x, t)
    #
    #             fig = plt.figure(figsize=(16, 12))
    #             ax = fig.add_subplot(111, projection='3d')
    #
    #             xs = grid[:, 0].detach().numpy().reshape(-1)
    #             ys = grid[:, 1].detach().numpy().reshape(-1)
    #             zs = s_pred.detach().numpy()
    #
    #             ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
    #             ax.set_title('PI DeepONet solution (3D)')
    #             ax.set_xlabel("$x$")
    #             ax.set_ylabel("$t$")
    #
    #             plt.show()
    #
    # def predict_s(self, U_star, Y_star):
    #     s_pred = self.operator_net(U_star, Y_star[:, 0], Y_star[:, 1])
    #     return s_pred
    #
    # def predict_res(self, U_star, Y_star):
    #     r_pred = self.residual_net(U_star, Y_star[:, 0], Y_star[:, 1])
    #     return r_pred


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


# N_train = 121
# N_test = 11
# m = 11
# P_train = 11
# Q_train = 11
#
# # Generate training data
# torch.manual_seed(0)  # use different key for generating training data
# key_train = torch.randint(0, 10 ** 10, (1, ))
# u_ics_train, y_ics_train, s_ics_train, u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = \
#     generate_training_data(key_train, N_train, m, P_train, Q_train)
#
# print('u_ics_train =', u_ics_train, '\ny_ics_train =', y_ics_train, '\ns_ics_train =', s_ics_train)
# print('u_bcs_train =', u_bcs_train, '\ny_bcs_train =', y_bcs_train, '\ns_bcs_train =', s_res_train)
# print('u_res_train =', u_res_train, '\ny_res_train =', y_res_train, '\ns_res_train =', s_res_train)
#
# # Initialize model
# neurons = 100
# branch_layers = [m, neurons, neurons, neurons, neurons]
# trunk_layers = [2, neurons, neurons, neurons, neurons]
# model = PI_DeepONet(branch_layers, trunk_layers)
#
# print(model, end='\n\n')
#
# # Create data set
# batch_size = 1024
# ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
# bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
# res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)
#
# epochs = 5000
#
# # Train
# model.train(ics_dataset, bcs_dataset, res_dataset, nIter=epochs)
#
# # Получение параметров модели
# params = model.state_dict()
#
# # Сохранение параметров модели
# torch.save(params, 'wave_params_512_done.pth')
#
# # Сохранение логов потерь
# np.save('wave_loss_res_512.npy', np.array(model.loss_res_log))
# np.save('wave_loss_ics_512.npy', np.array(model.loss_ics_log))
# np.save('wave_loss_bcs_512.npy', np.array(model.loss_bcs_log))
#
# # Загрузка модели и логов потерь
# flat_params = torch.load('wave_params_512_done.pth')
# loss_ics = np.load('wave_loss_ics_512_done.npy')
# loss_bcs = np.load('wave_loss_bcs_512_done.npy')
# loss_res = np.load('wave_loss_res_512_done.npy')
#
# # # Example
# # loss_res = np.load('weights/0/wave_loss_res_0.npy')
# # loss_res = np.load('wave_loss_res_0.npy')
#
# # Загрузка модели
# params = model.parameters(flat_params)
#
# # Настройка графиков
# plt.rc('font', family='serif')
# plt.rcParams.update(plt.rcParamsDefault)
#
# # График потерь
# fig = plt.figure(figsize=(16, 12))
#
# iters = 1000 * np.arange(len(loss_ics))
# plt.plot(iters, loss_bcs, lw=2, label='$\mathcal{L}_{bc}$')
# plt.plot(iters, loss_ics, lw=2, label='$\mathcal{L}_{ic}$')
# plt.plot(iters, loss_res, lw=2, label='$\mathcal{L}_{r}$')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.yscale('log')
# plt.legend()
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
# plt.tight_layout()
# plt.show()
#
# # Предсказанное решение
# T = 11
# P = 11
# P_test = 11
#
# s_fn = lambda t, x: np.sin(np.pi * x) * np.cos(np.pi * 2 * t)
#
# x = np.linspace(0, 1, P)
# u = np.sin(np.pi * x)
#
# u_test = np.tile(u, (P ** 2, 1))
#
# x = np.linspace(0, 1, P_test)
# t = np.linspace(0, 1, P_test)
# TT, XX = np.meshgrid(t, x)
#
# y_test = np.hstack([TT.flatten()[:, None], XX.flatten()[:, None]])
#
# # Предсказание модели
# s_pred = model.predict_s(torch.tensor(u_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
# S_pred = griddata(y_test, s_pred.detach().numpy(), (TT, XX), method='cubic')
#
# for k in range(T - 1):
#     u_k = S_pred[:, -1]
#     u_test_k = np.tile(u_k, (P_test ** 2, 1))
#     s_pred_k = model.predict_s(torch.tensor(u_test_k, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
#     S_pred_k = griddata(y_test, s_pred_k.detach().numpy().flatten(), (TT, XX), method='cubic')
#     S_pred = np.hstack([S_pred, S_pred_k])
#
# # Точное решение
# Nx = 11
# Nt = 11 * T
#
# x = np.linspace(0, 1, Nx)
# t = np.linspace(0, T, Nt)
# TT, XX = np.meshgrid(t, x)
#
# S_test = np.sin(np.pi * XX) * np.cos(np.pi * 2 * TT)
#
# # Относительная ошибка
# error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
# print('Relative l2 error: {:.3e}'.format(error))
#
# # Визуализация решений
# fig = plt.figure(figsize=(16, 12))
#
# plt.subplot(3, 1, 1)
# plt.pcolor(TT, XX, S_test, cmap='jet')
# plt.xlabel('$t$')
# plt.ylabel('$x$')
# plt.title('Exact')
# plt.colorbar()
# plt.tight_layout()
#
# plt.subplot(3, 1, 2)
# plt.pcolor(TT, XX, S_pred, cmap='jet')
# plt.xlabel('$t$')
# plt.ylabel('$x$')
# plt.title('Predicted')
# plt.colorbar()
# plt.tight_layout()
#
# plt.subplot(3, 1, 3)
# plt.pcolor(TT, XX, np.abs(S_pred - S_test), cmap='jet')
# plt.xlabel('$t$')
# plt.ylabel('$x$')
# plt.colorbar()
# plt.title('Absolute error')
# plt.tight_layout()
# plt.show()
#
# # 3D Визуализация решений
# Nx = 11
# Nt = 11
#
# x = torch.linspace(0, 1, Nx)
# t = torch.linspace(0, 1, Nt)
#
# grid = torch.cartesian_prod(x, t)
#
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111, projection='3d')
#
# xs = grid[:, 0].detach().numpy().reshape(-1)
# ys = grid[:, 1].detach().numpy().reshape(-1)
# zs = s_pred.detach().numpy()
#
# ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
# ax.set_title('PI DeepONet solution (3D)')
# ax.set_xlabel("$x$")
# ax.set_ylabel("$t$")
#
# plt.show()

########################################################################################################################


"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('gpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def wave_experiment(grid_res):
    exp_dict_list = []

    # Use methods: PINN, PI_DeepONet
    domain = Domain(method='PI_DeepONet')
    domain.variable('x', [0, 1], grid_res)
    domain.variable('t', [0, 1], grid_res)

    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-frienly
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)

    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

    Meaning c1*u*d2u/dx2 has the form

    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}

    None is for function without derivatives


    bval=torch.Tensor prescribed values at every point in the boundary
    """


    boundaries = Conditions()

    # Initial conditions at t=0
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=func)

    ## Initial conditions at t=1
    ## u(1,x)=sin(pi*x)
    bop2 = {
        'du/dt':
            {
                'coeff': 1,
                'du/dx': [1],
                'pow': 1,
                'var': 0
            }
    }
    boundaries.operator({'x': [0, 1], 't': 0}, operator=bop2, value=0)

    # Boundary conditions at x=0
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=func)

    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=func)


    """
    Defining wave equation

    Operator has the form

    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-frienly
    description/comments and are not used in model directly thus order of
    items must be preserved as (coeff,op,pow)



    term is a dict term={coefficient:c1,[sterm1,sterm2],'pow': power}

    c1 may be integer, function of grid or tensor of dimension of grid

    Meaning c1*u*d2u/dx2 has the form

    {'coefficient':c1,
     'u*d2u/dx2': [[None],[0,0]],
     'pow':[1,1]}

    None is for function without derivatives

    """

    equation = Equation()

    # operator is 4*d2u/dx2-1*d2u/dt2=0
    wave_eq = {
        'd2u/dt2**1':
            {
                'coeff': 1,
                'd2u/dt2': [1, 1],
                'pow': 1
            },
        '-C*d2u/dx2**1':
            {
                'coeff': -4.,
                'd2u/dx2': [0, 0],
                'pow': 1
            }
    }

    equation.add(wave_eq)

    net = torch.nn.Sequential(
        torch.nn.Linear(2, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 1)
    )

    start = time.time()

    neurons = 100
    branch_layer = [11, neurons, neurons, neurons, neurons]
    trunk_layer = [2, neurons, neurons, neurons, neurons]

    custom_model = PI_DeepONet(branch_layer, trunk_layer)

    model = Model(custom_model, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-5, randomize_parameter=1e-6, info_string_every=500)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    img_dir = os.path.join(os.path.dirname( __file__ ), 'wave_img')

    cb_plots = plot.Plots(save_every=500, print_every=500, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(grid)) ** 2))

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {}= {}'.format(grid_res, end - start))
    print('RMSE {}= {}'.format(grid_res, error_rmse))

    return exp_dict_list

nruns = 10

exp_dict_list = []

for grid_res in range(10, 101, 10):
    for _ in range(nruns):
        exp_dict_list.append(wave_experiment(grid_res))

import pandas as pd

exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
df = pd.DataFrame(exp_dict_list_flatten)
# df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(True)))