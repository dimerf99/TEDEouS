import torch
from scipy.interpolate import griddata

import jax.numpy as np
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.nn import relu, elu

from jax import config
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange

import matplotlib.pyplot as plt

# %matplotlib inline


key = random.PRNGKey(54321)


# Define the neural net
def MLP(layers, activation=relu):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = np.zeros(d_out)
        return W, b

    def init(rng_key):
        U1, b1 = xavier_init(random.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(random.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)

    def apply(params, inputs):
        a = len(params)
        params, U1, b1, U2, b2 = params
        length_params = len(params)
        U = activation(np.dot(inputs, U1) + b1)
        V = activation(np.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(np.dot(inputs, W))
            inputs = np.multiply(outputs, U) + np.multiply(1 - outputs, V)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply


# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=5000, decay_rate=0.9)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params)
        _, self.unravel = ravel_pytree(params)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define the opeartor net
    def operator_net(self, params, u, t, x):
        a = params
        s = len(params)
        # t = type(params)
        branch_params, trunk_params = params

        # neurons = 200
        # branch_layers = [m, neurons, neurons, neurons, neurons, neurons]
        # trunk_layers = [2, neurons, neurons, neurons, neurons, neurons]

        y = np.stack([t, x])
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        outputs = np.sum(B * T)
        return outputs

    def s_t_net(self, params, u, t, x):
        s_t = grad(self.operator_net, argnums=2)(params, u, t, x)
        return s_t

    def residual_net(self, params, u, t, x):
        s_tt = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, t, x)
        s_xx = grad(grad(self.operator_net, argnums=3), argnums=3)(params, u, t, x)
        a = 1 / 4
        res = a * s_tt - s_xx
        return res

    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        s_t_pred = vmap(self.s_t_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        # Compute loss
        loss_1 = np.mean((outputs.flatten() - s_pred) ** 2)
        loss_2 = np.mean(s_t_pred ** 2)
        loss = loss_1 + loss_2
        return loss

    @partial(jit, static_argnums=(0,))
    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        s_bc1_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        s_bc2_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 2], y[:, 3])
        # Compute loss
        loss_s_bc1 = np.mean((s_bc1_pred - outputs[:, 0]) ** 2)
        loss_s_bc2 = np.mean((s_bc2_pred - outputs[:, 1]) ** 2)
        loss_s_bc = loss_s_bc1 + loss_s_bc2

        return loss_s_bc

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])
        # Compute loss
        loss = np.mean(pred ** 2)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss(self, params, ics_batch, bcs_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_ics + loss_bcs + loss_res
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, ics_batch, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, ics_dataset, bcs_dataset, res_dataset, nIter=10000):
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            ics_batch = next(ics_data)
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bcs_batch, res_batch)

            params = self.get_params(self.opt_state)
            # print(params)

            loss_value = self.loss(params, ics_batch, bcs_batch, res_batch)
            loss_ics_value = self.loss_ics(params, ics_batch)
            loss_bcs_value = self.loss_bcs(params, bcs_batch)
            loss_res_value = self.loss_res(params, res_batch)

            self.loss_log.append(loss_value)
            self.loss_ics_log.append(loss_ics_value)
            self.loss_bcs_log.append(loss_bcs_value)
            self.loss_res_log.append(loss_res_value)

            pbar.set_postfix({'Loss': loss_value,
                              'loss_ics': loss_ics_value,
                              'loss_bcs': loss_bcs_value,
                              'loss_res': loss_res_value})

            if it % 200 == 0:
                # Predicted solution
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

                s_pred = self.predict_s(params, u_test, y_test)
                S_pred = griddata(y_test, s_pred, (TT, XX), method='cubic')

                for k in range(T - 1):
                    u_k = S_pred[:, -1]
                    u_test_k = np.tile(u_k, (P_test ** 2, 1))
                    s_pred_k = self.predict_s(params, u_test_k, y_test)
                    S_pred_k = griddata(y_test, s_pred_k.flatten(), (TT, XX), method='cubic')
                    S_pred = np.hstack([S_pred, S_pred_k])

                # Exact solution
                Nx = 21
                Nt = 21 * T

                x = np.linspace(0, 1, Nx)
                t = np.linspace(0, T, Nt)
                TT, XX = np.meshgrid(t, x)

                S_test = vmap(s_fn)(TT, XX)

                error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
                print('Relative l2 error: {:.3e}'.format(error))

                # Exact solution
                Nx = 21
                Nt = 21

                x = torch.linspace(0, 1, Nx)
                t = torch.linspace(0, 1, Nt)

                grid = torch.cartesian_prod(x, t)

                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(111, projection='3d')

                xs = grid[:, 0].detach().numpy().reshape(-1)
                ys = grid[:, 1].detach().numpy().reshape(-1)
                zs = s_pred

                ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
                ax.set_title('PI DeepONet solution (3D)')
                ax.set_xlabel("$x$")
                ax.set_ylabel("$t$")

                plt.show()

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)  # Here you can change to FALSE
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


length_scale = 0.5
output_scale = 10.0
gp_params = (length_scale, output_scale)
# c = 1.0


def RBF(x1, x2, gp_params):
    length_scale, output_scale = gp_params
    diffs = np.expand_dims(x1 / length_scale, 1) - \
            np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def generate_one_gaussain_sample(key, gp_params, N):
    jitter = 1e-10
    X = np.linspace(0.0, 1.0, N)[:, None]
    K = RBF(X, X, gp_params)
    K_jitter = K + jitter * np.eye(N)
    K_size = K_jitter.shape
    L = np.linalg.cholesky(K_jitter)
    gp_sample = np.dot(L, random.normal(key, (N,)))

    return gp_sample


def generate_one_training_data(key, m=100, P=100, Q=100):
    # print(f"generate_one_training_data: key = {key}")
    subkeys = random.split(key, 5)
    # print(f"generate_one_training_data: subkeys = {subkeys}")

    # Generate one input sample
    N = 512
    gp_sample = generate_one_gaussain_sample(subkeys[0], gp_params, N)
    x = np.linspace(0, 1, m)
    X = np.linspace(0, 1, N)[:, None]

    u_fn = lambda x: x * (1 - x) * np.interp(x, X.flatten(), gp_sample)

    def u_fn(x):
        g_samp = gp_sample
        interp_gp = np.interp(x, X.flatten(), gp_sample)
        val1 = x
        val2 = 1 - x
        return x * (1 - x) * interp_gp

    # Input sensor locations and measurements
    u = vmap(u_fn)(x)

    # IC training data
    u_ic = np.tile(u, (P, 1))

    t_0 = np.zeros((P, 1))
    x_0 = random.uniform(subkeys[1], (P, 1))
    y_ic = np.hstack([t_0, x_0])
    s_ic = vmap(u_fn)(x_0)

    # BC training data
    u_bc = np.tile(u, (P, 1))

    t_bc1 = random.uniform(subkeys[2], (P, 1))
    x_bc1 = np.zeros((P, 1))

    t_bc2 = random.uniform(subkeys[3], (P, 1))
    x_bc2 = np.ones((P, 1))

    y_bc1 = np.hstack([t_bc1, x_bc1])
    y_bc2 = np.hstack([t_bc2, x_bc2])
    y_bc = np.hstack([y_bc1, y_bc2])

    s_bc = np.zeros((Q, 2))

    # Residual training data
    u_r = np.tile(u, (Q, 1))
    y_r = random.uniform(subkeys[2], (Q, 2))
    s_r = np.zeros((Q, 1))

    return u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r


def generate_training_data(key, N, m, P, Q):
    # print(f"\ngenerate_training_data: key = {key}")
    config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    # print(f"generate_training_data: keys[0] = {keys[0]}")
    # print(f"generate_training_data: keys = {keys}")
    u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r = vmap(
        generate_one_training_data, (0, None, None, None)
    )(keys, m, P, Q)

    u_ic_train = np.float32(u_ic.reshape(N * P, -1))
    # print(u_ic_train)
    y_ic_train = np.float32(y_ic.reshape(N * P, -1))
    # print(y_ic_train)
    s_ic_train = np.float32(s_ic.reshape(N * P, -1))
    # print(s_ic_train)

    u_bc_train = np.float32(u_bc.reshape(N * P, -1))
    # print(u_bc_train)
    y_bc_train = np.float32(y_bc.reshape(N * P, -1))
    s_bc_train = np.float32(s_bc.reshape(N * P, -1))

    u_r_train = np.float32(u_r.reshape(N * Q, -1))
    y_r_train = np.float32(y_r.reshape(N * Q, -1))
    s_r_train = np.float32(s_r.reshape(N * Q, -1))
    config.update("jax_enable_x64", False)

    return u_ic_train, y_ic_train, s_ic_train, u_bc_train, y_bc_train, s_bc_train, u_r_train, y_r_train, s_r_train


# Generate training data
N_train = 441
N_test = 21
m = 21
P_train = 21
Q_train = 21

# idx = random.choice(key, self.N, (self.batch_size,), replace=True)

# Generate training data
key_train = random.PRNGKey(0)  # use different key for generating training data
u_ics_train, y_ics_train, s_ics_train, u_bcs_train, y_bcs_train, s_bcs_train, u_res_train, y_res_train, s_res_train = \
    generate_training_data(key_train, N_train, m, P_train, Q_train)

print('u_ics_train =', u_ics_train, '\ny_ics_train =', y_ics_train, '\ns_ics_train =', s_ics_train)
print('u_bcs_train =', u_bcs_train, '\ny_bcs_train =', y_bcs_train, '\ns_bcs_train =', s_res_train)
print('u_res_train =', u_res_train, '\ny_res_train =', y_res_train, '\ns_res_train =', s_res_train)

# Initialize model
neurons = 100
branch_layers = [m, neurons, neurons, neurons, neurons, neurons]
trunk_layers = [2, neurons, neurons, neurons, neurons, neurons]
model = PI_DeepONet(branch_layers, trunk_layers)

# Create data set
batch_size = 2048
ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

# Train model
model.train(ics_dataset, bcs_dataset, res_dataset, nIter=10000)

# Save the trained model and losses
flat_params, _ = ravel_pytree(model.get_params(model.opt_state))
np.save('wave_params.npy', flat_params)
np.save('wave_loss_res.npy', model.loss_res_log)
np.save('wave_loss_ics.npy', model.loss_ics_log)
np.save('wave_loss_bcs.npy', model.loss_bcs_log)

# Restore the trained model
params = model.unravel(np.load('wave_params (1).npy'))
loss_ics = np.load('wave_loss_ics (1).npy')
loss_bcs = np.load('wave_loss_bcs (1).npy')
loss_res = np.load('wave_loss_res (1).npy')

plt.rc('font', family='serif')
plt.rcParams.update(plt.rcParamsDefault)

# Losses
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
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.tight_layout()
plt.show()

# Predicted solution
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

s_pred = model.predict_s(params, u_test, y_test)
S_pred = griddata(y_test, s_pred, (TT, XX), method='cubic')

for k in range(T - 1):
    u_k = S_pred[:, -1]
    u_test_k = np.tile(u_k, (P_test ** 2, 1))
    s_pred_k = model.predict_s(params, u_test_k, y_test)
    S_pred_k = griddata(y_test, s_pred_k.flatten(), (TT, XX), method='cubic')
    S_pred = np.hstack([S_pred, S_pred_k])

# Exact solution
Nx = 21
Nt = 21 * T

x = np.linspace(0, 1, Nx)
t = np.linspace(0, T, Nt)
TT, XX = np.meshgrid(t, x)

S_test = vmap(s_fn)(TT, XX)

error = np.linalg.norm(S_pred - S_test, 2) / np.linalg.norm(S_test, 2)
print('Relative l2 error: {:.3e}'.format(error))

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

# Exact solution
Nx = 21
Nt = 21

x = torch.linspace(0, 1, Nx)
t = torch.linspace(0, 1, Nt)

grid = torch.cartesian_prod(x, t)

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

xs = grid[:, 0].detach().numpy().reshape(-1)
ys = grid[:, 1].detach().numpy().reshape(-1)
zs = s_pred

ax.plot_trisurf(xs, ys, zs, cmap=plt.cm.jet, linewidth=0.2, alpha=1)
ax.set_title('PI DeepONet solution (3D)')
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")

plt.show()
