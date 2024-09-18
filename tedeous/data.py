"""module for working with inerface for initialize grid, conditions and equation"""

from typing import List, Union

import torch
import numpy as np
import scipy
import os

from tedeous.device import check_device
from tedeous.input_preprocessing import EquationMixin


def tensor_dtype(dtype: str):
    """convert tensor to dtype format

    Args:
        dtype (str): dtype

    Returns:
        dtype: torch.dtype
    """
    if dtype == 'float32':
        dtype = torch.float32
    elif dtype == 'float64':
        dtype = torch.float64
    elif dtype == 'float16':
        dtype = torch.float16

    return dtype


def RBF(x1: torch.Tensor,
        x2: torch.Tensor,
        length_scale: float = 0.5,
        output_scale: float = 10.0) -> torch.Tensor:
    diffs = torch.unsqueeze(x1 / length_scale, 1) - torch.unsqueeze(x2 / length_scale, 0)
    r2 = torch.sum(diffs ** 2, dim=2)
    K = output_scale * torch.exp(-0.5 * r2)

    return K


def generate_gaussian_sample(key: int,
                             N: int,
                             length_scale: float = 0.5,
                             output_scale: float = 10.0) -> torch.Tensor:
    torch.manual_seed(key)
    jitter = 1e-12
    X = torch.linspace(0.0, 1.0, N, dtype=torch.float64).view(-1, 1)
    K = RBF(X, X, length_scale, output_scale)
    L = torch.linalg.cholesky(K + jitter * torch.eye(N))
    gp_sample = torch.matmul(L, torch.randn(N, dtype=torch.float64))

    return gp_sample


def u_func(x,
           subkey: int = None,
           length_scale: float = 0.5,
           output_scale: float = 10.0,
           N_gauss: int = 512):
    X = torch.linspace(0, 1, N_gauss).view(-1, 1)
    gp_sample = generate_gaussian_sample(subkey, N_gauss, length_scale, output_scale)
    interp_gp = np.interp(x, X.numpy().flatten(), gp_sample.numpy())
    return x * (x - 1) * torch.from_numpy(interp_gp)
    # return torch.cos(2 * torch.pi * x) * torch.sin(torch.pi * torch.from_numpy(interp_gp))


def build_deeponet(key, N=1000, m=10, P=10, Q=10, usol=None):
    length_scale = 0.5  # best var: length_scale = 0.7
    output_scale = 10.0

    torch.manual_seed(key)
    keys = torch.randint(0, 10 ** 10, (N,), dtype=torch.int64)

    results = []
    for key in keys:
        torch.manual_seed(key)
        subkey = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

        # # Generate one input sample
        # N_gauss = 512
        # # num = 1
        # # N_gauss_new = N_gauss * num
        # gp_sample = generate_gaussian_sample(subkey, N_gauss, length_scale, output_scale)
        # x = torch.linspace(0, 1, m)
        # X = torch.linspace(0, 1, N_gauss).view(-1, 1)
        #
        # def u_fn(x):
        #     interp_gp = np.interp(x, X.numpy().flatten(), gp_sample.numpy())
        #     return x * (1 - x) * torch.from_numpy(interp_gp)

        # Generate one input sample
        N_gauss = 512
        x = torch.linspace(0, 1, m)

        # Input sensor locations and measurements
        u = u_func(x, subkey, length_scale, output_scale, N_gauss)

        # IC training data
        u_ic = u.repeat(P, 1)

        t_01 = torch.zeros((P, 1))
        x_01 = torch.rand((P, 1))
        y_ic1 = torch.hstack([x_01, t_01])

        t_02 = torch.zeros((P, 1))
        x_02 = torch.rand((P, 1))
        y_ic2 = torch.hstack([x_02, t_02])

        y_ic = torch.hstack([y_ic1, y_ic2])

        # s_ic = u_fn(x_01).view(-1)
        s_ic = u_func(x_01, subkey).view(-1)

        # BC training data
        u_bc = u.repeat(P, 1)

        t_bc1 = torch.rand((P, 1))
        x_bc1 = torch.zeros((P, 1))

        t_bc2 = torch.rand((P, 1))
        x_bc2 = torch.ones((P, 1))

        y_bc1 = torch.hstack([x_bc1, t_bc1])
        y_bc2 = torch.hstack([x_bc2, t_bc2])
        y_bc = torch.hstack([y_bc1, y_bc2])

        s_bc = torch.zeros(Q)

        # Residual training data
        u_r = u.repeat(Q, 1)

        y_r = torch.rand((Q, 2))
        # x_r1 = torch.rand((P, 1))
        # t_r2 = torch.rand((P, 1))
        # y_r = torch.hstack([x_r1, t_r2])

        s_r = torch.zeros(Q)

        results.append([u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r])

    u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r = zip(*results)

    u_ic = torch.cat(u_ic).float()
    y_ic = torch.cat(y_ic).float()
    s_ic = torch.cat(s_ic).float()

    u_bc = torch.cat(u_bc).float()
    y_bc = torch.cat(y_bc).float()
    s_bc = torch.cat(s_bc).float()

    u_r = torch.cat(u_r).float()
    y_r = torch.cat(y_r).float()
    s_r = torch.cat(s_r).float()

    # # Geneate ics training data corresponding to one input sample
    # def generate_one_ics_training_data(key, u0, m=101, P=101):
    #     torch.manual_seed(key)
    #     t_0 = torch.zeros((P, 1))
    #     x_0 = torch.linspace(0, 1, P)[:, None]
    #
    #     y = torch.hstack([x_0, t_0])
    #     u = torch.tile(u0, (P, 1))
    #     s = u0
    #
    #     return u, y, s
    #
    # # Geneate bcs training data corresponding to one input sample
    # def generate_one_bcs_training_data(key, u0, m=101, P=100):
    #     torch.manual_seed(key)
    #     t_bc = torch.rand((P, 1))
    #     x_bc1 = torch.zeros((P, 1))
    #     x_bc2 = torch.ones((P, 1))
    #
    #     y1 = torch.hstack([x_bc1, t_bc])  # shape = (P, 2)
    #     y2 = torch.hstack([x_bc2, t_bc])  # shape = (P, 2)
    #
    #     u = torch.tile(u0, (P, 1))
    #     y = torch.hstack([y1, y2])  # shape = (P, 4)
    #     s = torch.zeros((P, 1))
    #
    #     return u, y, s
    #
    # # Geneate res training data corresponding to one input sample
    # def generate_one_res_training_data(key, u0, m=101, P=1000):
    #     torch.manual_seed(key)
    #     # subkeys = random.split(key, 2)
    #     subkeys = torch.randint(0, 10 ** 10, (2,), dtype=torch.int64)
    #
    #     torch.manual_seed(int(subkeys[0]))
    #     t_res = torch.rand((P, 1))
    #     torch.manual_seed(int(subkeys[1]))
    #     x_res = torch.rand((P, 1))
    #
    #     u = torch.tile(u0, (P, 1))
    #     y = torch.hstack([x_res, t_res])
    #     s = torch.zeros((P, 1))
    #
    #     return u, y, s
    #
    # # Geneate test data corresponding to one input sample
    # def generate_one_test_data(idx, usol, m=101, P=101):
    #     u = usol[idx]
    #     u0 = u[0, :]
    #
    #     t = torch.linspace(0, 1, P)
    #     x = torch.linspace(0, 1, P)
    #     X, T = np.meshgrid(x, t)
    #
    #     s = u.T.flatten()
    #     u = torch.tile(u0, (P ** 2, 1))
    #     y = torch.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    #
    #     return u, y, s
    #
    # # Load data
    # data = scipy.io.loadmat(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../examples/wolfram_sln/Burgers.mat')))
    # usol = torch.tensor(data['usol'])
    #
    # N = usol.shape[0]  # number of total input samples
    # N_train = 200  # number of input samples used for training
    # N_test = N - N_train  # number of input samples used for test
    # m = 61  # number of sensors for input samples
    # P_ics_train = 61  # number of locations for evulating the initial condition
    # P_bcs_train = 60  # number of locations for evulating the boundary condition
    # P_res_train = 2000  # number of locations for evulating the PDE residual
    # P_test = 61  # resolution of uniform grid for the test data
    #
    # u0_train = usol[:N_train, :]  # input samples
    # # usol_train = usol[:N_train,:,:]
    #
    # torch.manual_seed(key)  # use different key for generating test data
    # keys = torch.randint(0, 10 ** 10, (N_train, ), dtype=torch.int64)
    #
    # # Generate training data for inital condition
    # results_ic = [generate_one_ics_training_data(key, u0_train, m, P_ics_train) for key in keys]
    # u_ics_train, y_ics_train, s_ics_train = zip(*results_ic)
    #
    # u_ic = torch.cat(u_ics_train).float()
    # y_ic = torch.cat(y_ics_train).float()
    # s_ic = torch.cat(s_ics_train).float()
    #
    # # Generate training data for boundary condition
    # results_bc = [generate_one_bcs_training_data(key, u0_train, m, P_bcs_train) for key in keys]
    # u_bcs_train, y_bcs_train, s_bcs_train = zip(*results_bc)
    #
    # u_bc = torch.cat(u_bcs_train).float()
    # y_bc = torch.cat(y_bcs_train).float()
    # s_bc = torch.cat(s_bcs_train).float()
    #
    # # Generate training data for PDE residual
    # results_res = [generate_one_res_training_data(key, u0_train, m, P_res_train) for key in keys]
    # u_res_train, y_res_train, s_res_train = zip(*results_res)
    #
    # u_r = torch.cat(u_res_train).float()
    # y_r = torch.cat(y_res_train).float()
    # s_r = torch.cat(s_res_train).float()

    return (u_ic, y_ic, s_ic), \
           (u_bc, y_bc, s_bc), \
           (u_r, y_r, s_r)


class Domain():
    """class for grid building
    """

    def __init__(self,
                 type='uniform',
                 method='PINN',
                 N=None,
                 m=None,
                 P=None,
                 Q=None,
                 key=0):
        self.type = type
        self.method = method
        self.variable_dict = {}
        self.N = N
        self.m = m
        self.P = P
        self.Q = Q
        self.key = key

    def variable(
            self,
            variable_name: str,
            variable_set: Union[List, torch.Tensor],
            n_points: Union[None, int],
            dtype: str = 'float32') -> None:
        """ determine varibles for grid building.

        Args:
            varible_name (str): varible name.
            variable_set (Union[List, torch.Tensor]): [start, stop] list for spatial variable or torch.Tensor with points for variable.
            n_points (int): number of points in discretization for variable.
            dtype (str, optional): dtype of result vector. Defaults to 'float32'.

        """

        dtype = tensor_dtype(dtype)

        if isinstance(variable_set, torch.Tensor):
            variable_tensor = check_device(variable_set)  # Можно ли убрать?
            variable_tensor = variable_set.to(dtype)
            self.variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                n_points += 1
                start, end = variable_set

                if self.method == 'PINN':
                    variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
                elif self.method == 'PI_DeepONet':
                    variable_tensor = []
                    for _ in range(self.N):
                        variable_tensor.append(torch.rand((n_points - 1, 1)))

                    variable_tensor = torch.cat(variable_tensor, dim=0)

                self.variable_dict[variable_name] = variable_tensor

    def build(self, mode: str) -> torch.Tensor:

        """ building the grid for algorithm

        Args:
            mode (str): mode for equation solution, *mat, autograd, NN*

        Returns:
            torch.Tensor: resulting grid.
        """
        var_lst = list(self.variable_dict.values())
        var_lst = [i.cpu() for i in var_lst]

        grid = None

        if mode in ('autograd', 'NN'):
            if len(self.variable_dict) == 1:
                grid = check_device(var_lst[0].reshape(-1, 1))
            else:
                if self.method == 'PINN':
                    grid = check_device(torch.cartesian_prod(*var_lst))
                elif self.method == 'PI_DeepONet':
                    grid = check_device(torch.cat(var_lst, dim=1).float())

        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = check_device(torch.tensor(np.array(grid)))
            # Можно ли переписать всю эту конструкцию на torch.meshgrid???
            # дефолтным методом индексирования там является indexing='ij'
            # https://pytorch.org/docs/stable/generated/torch.meshgrid.html

        grid = check_device(torch.tensor(np.array(grid)))
        return grid


class Conditions():
    """class for adding the conditions: initial, boundary, and data.
    """

    def __init__(self, method='PINN', N=None, m=None, P=None, Q=None, key=None):
        self.conditions_lst = []
        self.method = method
        self.N = N
        self.m = m
        self.P = P
        self.Q = Q
        self.key = key
        self.s = []

    def dirichlet(
            self,
            bnd: Union[torch.Tensor, dict],
            value: Union[callable, torch.Tensor, float],
            var: int = 0):
        """ determine dirichlet boundary condition.

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values.
            value (Union[callable, torch.Tensor, float]): values at the boundary (bnd)
            if callable: value = function(bnd)
            var (int, optional): variable for system case, for single equation is 0. Defaults to 0.
        """

        self.conditions_lst.append({'bnd': bnd,
                                    'bop': None,
                                    'bval': value,
                                    'var': var,
                                    'type': 'dirichlet'})

    def operator(self,
                 bnd: Union[torch.Tensor, dict],
                 operator: dict,
                 value: Union[callable, torch.Tensor, float]):
        """ determine operator boundary condition

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values
            operator (dict): dictionary with opertor terms: {'operator name':{coeff, term, pow, var}}
            value (Union[callable, torch.Tensor, float]): value on the boundary (bnd).
            if callable: value = function(bnd)
        """
        try:
            var = operator[operator.keys()[0]]['var']
        except:
            var = 0
        operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'operator'})

    def periodic(self,
                 bnd: Union[List[torch.Tensor], List[dict]],
                 operator: dict = None,
                 var: int = 0):
        """Periodic can be: periodic dirichlet (example u(x,t)=u(-x,t))
        if form with bnd and var for system case.
        or periodic operator (example du(x,t)/dx=du(-x,t)/dx)
        in form with bnd and operator.
        Parameter 'bnd' is list: [b_coord1:torch.Tensor, b_coord2:torch.Tensor,..] or
        bnd = [{'x': 1, 't': [0,1]},{'x': -1, 't':[0,1]}]

        Args:
            bnd (Union[List[torch.Tensor], List[dict]]): list with dicionaries or torch.Tensors
            operator (dict, optional): operator dict. Defaults to None.
            var (int, optional): variable for system case and periodic dirichlet. Defaults to 0.
        """
        value = torch.tensor([0.])
        if operator is None:
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})
        else:
            try:
                var = operator[operator.keys()[0]]['var']
            except:
                var = 0
            operator = EquationMixin.equation_unify(operator)
            self.conditions_lst.append({'bnd': bnd,
                                        'bop': operator,
                                        'bval': value,
                                        'var': var,
                                        'type': 'periodic'})

    def data(
            self,
            bnd: Union[torch.Tensor, dict],
            operator: Union[dict, None],
            value: torch.Tensor,
            var: int = 0):
        """ conditions for available solution data

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values
            operator (Union[dict, None]): dictionary with opertor terms: {'operator name':{coeff, term, pow, var}}
            value (Union[torch.Tensor, float]): values at the boundary (bnd)
            var (int, optional): variable for system case and periodic dirichlet. Defaults to 0.
        """
        if operator is not None:
            operator = EquationMixin.equation_unify(operator)
        self.conditions_lst.append({'bnd': bnd,
                                    'bop': operator,
                                    'bval': value,
                                    'var': var,
                                    'type': 'data'})

    def _bnd_grid(self,
                  bnd: Union[torch.Tensor, dict],
                  variable_dict: dict,
                  dtype,
                  count: int = None) -> torch.Tensor:
        """ build subgrid for every condition.

        Args:
            bnd (Union[torch.Tensor, dict]): boundary points can be torch.Tensor
            or dict with keys as coordinates names and values as coordinates values
            variable_dict (dict): dictionary with torch.Tensors for each domain variable
            dtype (dtype): dtype

        Returns:
            torch.Tensor: subgrid for boundary cond-s.
        """

        dtype = variable_dict[list(variable_dict.keys())[0]].dtype

        if isinstance(bnd, torch.Tensor):
            bnd_grid = bnd.to(dtype)
        else:
            var_lst = []
            for var in variable_dict.keys():
                # В этих условных операторах нужно распределить ГС для дипонета
                if isinstance(bnd[var], torch.Tensor):
                    var_lst.append(check_device(bnd[var]).to(dtype))
                elif isinstance(bnd[var], (float, int)):
                    if self.method == 'PINN':
                        var_lst.append(check_device(torch.tensor([bnd[var]])).to(dtype))
                    elif self.method == 'PI_DeepONet':
                        l_max_var = len(max(list(variable_dict.values()), key=lambda x: len(x)))
                        var_lst.append(check_device(torch.full((l_max_var,), bnd[var])
                                                    .reshape((l_max_var, 1))).to(dtype))
                elif isinstance(bnd[var], list):
                    if self.method == 'PINN':
                        lower_bnd = bnd[var][0]
                        upper_bnd = bnd[var][1]
                        grid_var = variable_dict[var]
                        bnd_var = grid_var[(grid_var >= lower_bnd) & (grid_var <= upper_bnd)]
                        var_lst.append(check_device(bnd_var).to(dtype))
                    elif self.method == 'PI_DeepONet':
                        # var_lst.append(check_device(bnd_var.reshape((len(bnd_var), 1))).to(dtype))

                        torch.manual_seed(0)
                        key = torch.randint(0, 10 ** 10, (1, ), dtype=torch.int64)

                        torch.manual_seed(int(key))
                        keys = torch.randint(0, 10 ** 10, (self.N,), dtype=torch.int64)

                        ic1, ic2, bc1, bc2, results_s = [], [], [], [], []

                        for key in keys:
                            torch.manual_seed(key)
                            subkey = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

                            bnd_var_ic1 = torch.rand((self.P, 1))
                            bnd_var_ic2 = bnd_var_ic1
                            bnd_var_bc1 = torch.rand((self.P, 1))
                            bnd_var_bc2 = torch.rand((self.P, 1))

                            ic1.append(bnd_var_ic1)
                            ic2.append(bnd_var_ic2)
                            bc1.append(bnd_var_bc1)
                            bc2.append(bnd_var_bc2)

                            results_s.append(u_func(bnd_var_ic1, subkey))

                        self.s.append(torch.cat(results_s).float())

                        bnd_ic1 = torch.cat(ic1).float()
                        bnd_ic2 = torch.cat(ic2).float()
                        bnd_bc1 = torch.cat(bc1).float()
                        bnd_bc2 = torch.cat(bc2).float()

                    result = [bnd_ic1, bnd_ic2, bnd_bc1, bnd_bc2]
                    var_lst.append(check_device(result[count]).to(dtype))

            if self.method == 'PINN':
                bnd_grid = torch.cartesian_prod(*var_lst).to(dtype)
            elif self.method == 'PI_DeepONet':
                bnd_grid = torch.hstack([*var_lst])

        if len(bnd_grid.shape) == 1:
            bnd_grid = bnd_grid.reshape(-1, 1)

        return bnd_grid

    def build(self,
              variable_dict: dict) -> List[dict]:
        """ preprocessing of initial boundaries data.

        Args:
            variable_dict (dict): dictionary with torch.Tensors for each domain variable

        Returns:
            List[dict]: list with dicts (where is all info obaut bconds)
        """

        if self.conditions_lst == []:
            return None

        try:
            dtype = variable_dict[list(variable_dict.keys())[0]].dtype
        except:
            dtype = variable_dict[list(variable_dict.keys())[0]][0].dtype  # if periodic

        if self.method == 'PI_DeepONet':
            count = 0

        for cond in self.conditions_lst:
            if self.method == 'PI_DeepONet':
                flag = cond['bnd']['t']
            if cond['type'] == 'periodic':
                cond_lst = []
                for bnd in cond['bnd']:
                    cond_lst.append(self._bnd_grid(bnd, variable_dict, dtype))
                cond['bnd'] = cond_lst
            else:
                if self.method == 'PINN':
                    cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype)
                elif self.method == 'PI_DeepONet':
                    cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype, count)
                    count += 1

            if isinstance(cond['bval'], torch.Tensor):
                if self.method == 'PI_DeepONet' and flag == 0:
                    cond['bval'] = self.s[0]
                cond['bval'] = check_device(cond['bval']).to(dtype)
            elif isinstance(cond['bval'], (float, int)):
                cond['bval'] = check_device(
                    torch.ones_like(cond['bnd'][:, 0]) * cond['bval']).to(dtype)
            elif callable(cond['bval']):
                cond['bval'] = check_device(cond['bval'](cond['bnd'])).to(dtype)

        return self.conditions_lst


class Equation():
    """class for adding eqution.
    """

    def __init__(self):
        self.equation_lst = []

    def add(self, eq: dict):
        """ add equation

        Args:
            eq (dict): equation in operator form.
        """
        self.equation_lst.append(eq)


def s_solutions(key, N=1000, m=10, P=10, Q=10, usol=None):
    length_scale = 0.5  # best var: length_scale = 0.7
    output_scale = 10.0

    torch.manual_seed(key)
    keys = torch.randint(0, 10 ** 10, (N,), dtype=torch.int64)

    results = []
    for key in keys:
        torch.manual_seed(key)
        subkey = torch.randint(0, 10 ** 10, (1,), dtype=torch.int64)

        # Generate one input sample
        N_gauss = 512
        gp_sample = generate_gaussian_sample(subkey, N_gauss, length_scale, output_scale)
        x = torch.linspace(0, 1, m)
        X = torch.linspace(0, 1, N_gauss).view(-1, 1)

        def u_fn(x):
            interp_gp = np.interp(x, X.numpy().flatten(), gp_sample.numpy())
            return x * (1 - x) * torch.from_numpy(interp_gp)

        # Input sensor locations and measurements
        u = u_fn(x)

        # IC training data
        u_ic = u.repeat(P, 1)

        t_01 = torch.zeros((P, 1))
        x_01 = torch.rand((P, 1))
        y_ic1 = torch.hstack([x_01, t_01])

        t_02 = torch.zeros((P, 1))
        x_02 = torch.rand((P, 1))
        y_ic2 = torch.hstack([x_02, t_02])

        y_ic = torch.hstack([y_ic1, y_ic2])

        s_ic = u_fn(x_01).view(-1)

        # BC training data
        u_bc = u.repeat(P, 1)

        t_bc1 = torch.rand((P, 1))
        x_bc1 = torch.zeros((P, 1))

        t_bc2 = torch.rand((P, 1))
        x_bc2 = torch.ones((P, 1))

        y_bc1 = torch.hstack([x_bc1, t_bc1])
        y_bc2 = torch.hstack([x_bc2, t_bc2])
        y_bc = torch.hstack([y_bc1, y_bc2])

        s_bc = torch.zeros(Q)

        # Residual training data
        u_r = u.repeat(Q, 1)
        y_r = torch.rand((Q, 2))
        s_r = torch.zeros(Q)

        results.append([u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r])

    u_ic, y_ic, s_ic, u_bc, y_bc, s_bc, u_r, y_r, s_r = zip(*results)

    u_ic = torch.cat(u_ic).float()
    y_ic = torch.cat(y_ic).float()
    s_ic = torch.cat(s_ic).float()

    u_bc = torch.cat(u_bc).float()
    y_bc = torch.cat(y_bc).float()
    s_bc = torch.cat(s_bc).float()

    u_r = torch.cat(u_r).float()
    y_r = torch.cat(y_r).float()
    s_r = torch.cat(s_r).float()

    return (u_ic, y_ic, s_ic), \
           (u_bc, y_bc, s_bc), \
           (u_r, y_r, s_r)
