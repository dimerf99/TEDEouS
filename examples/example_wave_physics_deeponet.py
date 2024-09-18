# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import torch
import numpy as np
import os
import sys
import time
# import torch.nn.functional as F

from tedeous.data import Domain, Conditions, Equation, s_solutions
from tedeous.model import Model

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device
from tedeous.data import u_func

# Custom models
from tedeous.models import MLP, PI_DeepONet
from tedeous.data import build_deeponet

# import kan
# import efficient_kan
# import fastkan

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Preparing grid

Grid is an essentially torch.Tensor of a n-D points where n is the problem
dimensionality
"""

solver_device('gpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def wave_experiment(grid_res):
    exp_dict_list = []

    neurons = 10
    m = neurons
    P = neurons
    Q = neurons
    N = neurons * 40

    domain = Domain(method='PI_DeepONet', N=N, m=m, P=P, Q=Q)
    domain.variable('x', [0, 1], m)
    domain.variable('t', [0, 1], m)

    torch.manual_seed(0)
    key = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

    train_data = build_deeponet(key, N, m, P, Q)

    u_branch = train_data[0][0]
    s_ic = train_data[0][2]
    s_bc = train_data[1][2]

    """
    Preparing boundary conditions (BC)

    For every boundary we define three items

    bnd=torch.Tensor of a boundary n-D points where n is the problem
    dimensionality

    bop=dict in form {'term1':term1,'term2':term2} -> term1+term2+...=0

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

    boundaries = Conditions(method='PI_DeepONet', N=N, m=m, P=P, Q=Q)

    # Initial conditions at t=0
    boundaries.dirichlet({'x': [0, 1], 't': 0}, value=s_ic)

    # Initial conditions at t=1
    # u(1,x)=sin(pi*x)
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
    boundaries.dirichlet({'x': 0, 't': [0, 1]}, value=s_bc)

    # Boundary conditions at x=1
    boundaries.dirichlet({'x': 1, 't': [0, 1]}, value=s_bc)

    """
    Defining wave equation

    Operator has the form

    op=dict in form {'term1':term1,'term2':term2}-> term1+term2+...=0

    NB! dictionary keys at the current time serve only for user-friendly
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

    # NN mode
    neurons = 50

    # branch_layers = [m, neurons, neurons, neurons, neurons, neurons, neurons]
    # trunk_layers = [2, neurons, neurons, neurons, neurons, neurons, 1]

    # KAN mode
    splines = 20

    # # branch with custom MLP
    # branch_net = MLP(branch_layers, activation=torch.tanh)

    # branch with torch.nn.Sequential
    branch_net = torch.nn.Sequential(
        torch.nn.Linear(m, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons)
    )
    for m in branch_net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # # branch with EfficientKAN
    # branch_net = efficient_kan.KAN(
    #     [grid_res, splines, 1],
    #     grid_size=10,
    #     spline_order=3,
    #     scale_noise=0.1,
    #     scale_base=1.0,
    #     scale_spline=1.0,
    #     base_activation=torch.nn.Tanh,
    #     grid_eps=0.02,
    #     grid_range=[-1, 1]
    # )

    # # branch with FastKAN
    # branch_net = fastkan.FastKAN(
    #     [grid_res, splines, splines, splines, splines],
    #     grid_min=-5.,
    #     grid_max=5.,
    #     num_grids=2,
    #     use_base_update=True,
    #     base_activation=F.tanh,
    #     spline_weight_init_scale=0.05
    # )

    # # trunk with custom MLP
    # trunk_net = MLP(trunk_layers, activation=torch.tanh)

    # trunk with torch.nn.Sequential
    trunk_net = torch.nn.Sequential(
        torch.nn.Linear(2, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, neurons),
        torch.nn.Tanh(),
        torch.nn.Linear(neurons, 1)
    )
    for m in trunk_net.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # # trunk with EfficientKAN
    # trunk_net = efficient_kan.KAN(
    #     [2, splines, splines, splines, 1],
    #     grid_size=10,
    #     spline_order=3,
    #     scale_noise=0.1,
    #     scale_base=1.0,
    #     scale_spline=1.0,
    #     base_activation=torch.nn.Tanh,
    #     grid_eps=0.02,
    #     grid_range=[-1, 1]
    # )

    # # trunk with FastKAN
    # trunk_net = fastkan.FastKAN(
    #     [2, splines, splines, splines, splines],
    #     grid_min=-5.,
    #     grid_max=5.,
    #     num_grids=2,
    #     use_base_update=True,
    #     base_activation=F.tanh,
    #     spline_weight_init_scale=0.05
    # )

    net = PI_DeepONet(branch_net=branch_net, trunk_net=trunk_net)
    # net = MLP(trunk_layers, activation=F.tanh)

    print(f"neurons = {neurons}")
    print(f"u_branch = {u_branch}")

    start = time.time()

    model = Model(net, domain, equation, boundaries, method='PI_DeepONet', u=u_branch)

    model.compile("autograd", lambda_operator=1, lambda_bound=100)

    cb_es = early_stopping.EarlyStopping(eps=1e-6,
                                         randomize_parameter=1e-6,
                                         info_string_every=1)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    img_dir = os.path.join(os.path.dirname(__file__), 'wave_img_PI_DeepONet')

    cb_plots = plot.Plots(save_every=50, print_every=None, img_dir=img_dir, method='PI_DeepONet', u=u_branch)

    optimizer = Optimizer('Adam', {'lr': 1e-4})

    model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_plots, cb_cache])

    end = time.time()

    grid = domain.build('NN').to('cuda')
    net = net.to('cuda')

    # net(grid) --> net(u, grid):
    # G(u)(y) ~ branch(u) * trunk(y)
    error_rmse = torch.sqrt(torch.mean((func(grid).reshape(-1, 1) - net(u_branch, grid)) ** 2))

    exp_dict_list.append({'grid_res': grid_res, 'time': end - start, 'RMSE': error_rmse.detach().cpu().numpy(),
                          'type': 'wave_eqn_physical', 'cache': True})

    print('Time taken {} = {}'.format(grid_res, end - start))
    print('RMSE {} = {}'.format(grid_res, error_rmse))

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
