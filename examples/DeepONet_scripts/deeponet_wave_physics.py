# -*- coding: utf-8 -*-
"""
Created on Mon May 31 12:33:44 2021

@author: user
"""
import numpy as np
import torch
import pandas as pd
import os
import sys
import time

from scipy.interpolate import griddata
import matplotlib.pyplot as plt


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model

# Custom models
from tedeous.models import PI_DeepONet_0, FourierNN, FeedForward, Fourier_embedding
from tedeous.data import DataGenerator, generate_training_data

from tedeous.callbacks import early_stopping, plot, cache
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

# solver_device('gpu')


def func(grid):
    x, t = grid[:, 0], grid[:, 1]
    sln = torch.cos(2 * np.pi * t) * torch.sin(np.pi * x)
    return sln


def wave_experiment(grid_res):
    exp_dict_list = []

    domain = Domain()
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

    model = Model(net, domain, equation, boundaries)

    model.compile("autograd", lambda_operator=1, lambda_bound=50)

    cb_es = early_stopping.EarlyStopping(eps=1e-5, randomize_parameter=1e-6, info_string_every=1000)

    cb_cache = cache.Cache(cache_verbose=True, model_randomize_parameter=1e-6)

    img_dir = os.path.join(os.path.dirname( __file__ ), 'wave_deeponet/wave_img')

    cb_plots = plot.Plots(save_every=1000, print_every=1000, img_dir=img_dir)

    optimizer = Optimizer('Adam', {'lr': 1e-3})

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


# nruns = 10
#
# exp_dict_list = []
#
# for grid_res in range(20, 401, 20):
#     for _ in range(nruns):
#         exp_dict_list.append(wave_experiment(grid_res))
#
#
# exp_dict_list_flatten = [item for sublist in exp_dict_list for item in sublist]
# df = pd.DataFrame(exp_dict_list_flatten)
#
# # df.boxplot(by='grid_res',column='time',fontsize=42,figsize=(20,10))
# # df.boxplot(by='grid_res',column='RMSE',fontsize=42,figsize=(20,10),showfliers=False)
# df.to_csv('examples/benchmarking_data/wave_experiment_physical_10_100_cache={}.csv'.format(str(True)))


# Generate training data
N_train = 121
N_test = 11
m = 11
P_train = 11
Q_train = 11

# Generate training data
torch.manual_seed(0)  # use different key for generating training data
key_train = torch.randint(0, 10 ** 10, (1, ))
u_ics_train, y_ics_train, s_ics_train, u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = \
    generate_training_data(key_train, N_train, m, P_train, Q_train)

print('u_ics_train =', u_ics_train, '\ny_ics_train =', y_ics_train, '\ns_ics_train =', s_ics_train)
print('u_bcs_train =', u_bcs_train, '\ny_bcs_train =', y_bcs_train, '\ns_bcs_train =', s_res_train)
print('u_res_train =', u_res_train, '\ny_res_train =', y_res_train, '\ns_res_train =', s_res_train)

# Initialize model
neurons = 100
branch_layers = [m, neurons, neurons, neurons, neurons]
trunk_layers = [2, neurons, neurons, neurons, neurons]
model = PI_DeepONet_0(branch_layers, trunk_layers)

print(model, end='\n\n')

# Create data set
batch_size = 2048
ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

epochs = 10000

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
T = 11
P = 11
P_test = 11

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
Nx = 11
Nt = 11 * T

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

########################################################################################################################




