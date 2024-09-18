# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 15:55:55 2024

@author: user
"""

import torch
import numpy as np
import os
import sys
import time
import shutil
import datetime
import tempfile
import glob
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from tedeous.data import Domain, Conditions, Equation
from typing import List, Union, Tuple, Any
from scipy.spatial import Delaunay
from copy import copy
from copy import deepcopy
from abc import ABC, abstractmethod
from torch.nn import Module
from scipy import linalg
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.optim.lr_scheduler import ExponentialLR
from matplotlib import cm


# import custom models from tedeous.models
# from tedeous.models import MLP, FeedForward, PI_DeepONet


# from tedeous.device import check_device

def check_device(data: Any):
    """ checking the device of the data.
        If the data.device is not same with torch.set_default_device,
        change one.
    Args:
        data (Any): it could be model or torch.Tensors

    Returns:
        data (Any): data with correct device
    """
    device = torch.tensor([0.]).device.type
    if data.device.type != device:
        return data.to(device)
    else:
        return data


def lambda_prepare(val: torch.Tensor,
                   lambda_: Union[int, list, torch.Tensor]) -> torch.Tensor:
    """ Prepares lambdas for corresponding equation or bcond type.

    Args:
        val (_type_): operator tensor or bval tensor
        lambda_ (Union[int, list, torch.Tensor]): regularization parameters values

    Returns:
        torch.Tensor: torch.Tensor with lambda_ values,
        len(lambdas) = number of columns in val
    """

    if isinstance(lambda_, torch.Tensor):
        return lambda_

    if isinstance(lambda_, int):
        try:
            lambdas = torch.ones(val.shape[-1]) * lambda_
        except:
            lambdas = torch.tensor(lambda_)
    elif isinstance(lambda_, list):
        lambdas = torch.tensor(lambda_)

    return lambdas.reshape(1, -1)


# from tedeous.input_preprocessing import EquationMixin
class EquationMixin:
    """
    Auxiliary class. This one contains some methods that uses in other classes.
    """

    @staticmethod
    def equation_unify(equation: dict) -> dict:
        """ Adding 'var' to the 'operator' if it's absent or convert to
        list 'pow' and 'var' if it's int or float.

        Args:
            equation (dict): operator in input form.

        Returns:
            dict: equation with unified for solver parameters.
        """

        for operator_label in equation.keys():
            operator = equation[operator_label]
            dif_dir = list(operator.keys())[1]
            try:
                operator['var']
            except:
                if isinstance(operator['pow'], (int, float)):
                    operator[dif_dir] = [operator[dif_dir]]
                    operator['pow'] = [operator['pow']]
                    operator['var'] = [0]
                elif isinstance(operator['pow'], list):
                    operator['var'] = [0 for _ in operator['pow']]
                continue
            if isinstance(operator['pow'], (int, float)):
                operator[dif_dir] = [operator[dif_dir]]
                operator['pow'] = [operator['pow']]
                operator['var'] = [operator['var']]

        return equation

    @staticmethod
    def closest_point(grid: torch.Tensor, target_point: float) -> int:
        """ Defines the closest boundary point to the grid.

        Args:
            grid (torch.Tensor): grid (domain discretization).
            target_point (float): boundary point.

        Returns:
            int: position of the boundary point on the grid.
        """

        min_dist = np.inf
        pos = 0
        min_pos = 0
        for point in grid:
            dist = torch.linalg.norm(point - target_point)
            if dist < min_dist:
                min_dist = dist
                min_pos = pos
            pos += 1
        return min_pos

    @staticmethod
    def convert_to_double(bnd: Union[list, np.array]) -> float:
        """ Converts points to double type.

        Args:
            bnd (Union[list, np.array]): array or list of arrays
                points that should be converted

        Returns:
            float: bnd with double type.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.convert_to_double(cur_bnd)
            return bnd
        elif isinstance(bnd, np.ndarray):
            return torch.from_numpy(bnd).double()
        return bnd.double()

    @staticmethod
    def search_pos(grid: torch.Tensor, bnd) -> list:
        """ Method for searching position bnd in grid.

        Args:
            grid (torch.Tensor): array of a n-D points.
            bnd (_type_): points that should be converted.

        Returns:
            list: list of positions bnd on grid.
        """

        if isinstance(bnd, list):
            for i, cur_bnd in enumerate(bnd):
                bnd[i] = EquationMixin.search_pos(grid, cur_bnd)
            return bnd
        pos_list = []
        for point in bnd:
            try:
                pos = int(torch.where(torch.all(
                    torch.isclose(grid, point), dim=1))[0])
            except Exception:
                pos = EquationMixin.closest_point(grid, point)
            pos_list.append(pos)
        return pos_list

    @staticmethod
    def bndpos(grid: torch.Tensor, bnd: torch.Tensor) -> Union[list, int]:
        """ Returns the position of the boundary points on the grid.

        Args:
            grid (torch.Tensor): grid for coefficient in form of
            torch.Tensor mapping.
            bnd (torch.Tensor):boundary conditions.

        Returns:
            Union[list, int]: list of positions of the boundary points on the grid.
        """

        if grid.shape[0] == 1:
            grid = grid.reshape(-1, 1)
        grid = grid.double()
        bnd = EquationMixin.convert_to_double(bnd)
        bndposlist = EquationMixin.search_pos(grid, bnd)
        return bndposlist


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


class Domain():
    """class for grid building
    """

    def __init__(self, type='uniform'):
        # STEP 1
        self.type = type
        self.variable_dict = {}

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
            variable_tensor = check_device(variable_set)
            variable_tensor = variable_set.to(dtype)
            self.variable_dict[variable_name] = variable_tensor
        else:
            if self.type == 'uniform':
                # n_points = n_points + 1
                n_points += 1
                start, end = variable_set
                variable_tensor = torch.linspace(start, end, n_points, dtype=dtype)
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

        if mode in ('autograd', 'NN'):
            if len(self.variable_dict) == 1:
                grid = check_device(var_lst[0].reshape(-1, 1))
            else:
                grid = check_device(torch.cartesian_prod(*var_lst))
        else:
            grid = np.meshgrid(*var_lst, indexing='ij')
            grid = check_device(torch.tensor(np.array(grid)))

        return grid


class Conditions():
    """class for adding the conditions: initial, boundary, and data.
    """

    def __init__(self):
        # STEP 2
        self.conditions_lst = []

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

        # STEP 3
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
                  dtype) -> torch.Tensor:
        """ build subgrid for every condition.

        Args:
            bnd (Union[torch.Tensor, dict]):oundary points can be torch.Tensor
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
                if isinstance(bnd[var], torch.Tensor):
                    var_lst.append(check_device(bnd[var]).to(dtype))
                elif isinstance(bnd[var], (float, int)):
                    var_lst.append(check_device(torch.tensor([bnd[var]])).to(dtype))
                elif isinstance(bnd[var], list):
                    lower_bnd = bnd[var][0]
                    upper_bnd = bnd[var][1]
                    grid_var = variable_dict[var]
                    bnd_var = grid_var[(grid_var >= lower_bnd) & (grid_var <= upper_bnd)]
                    var_lst.append(check_device(bnd_var).to(dtype))
            bnd_grid = torch.cartesian_prod(*var_lst).to(dtype)
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

        for cond in self.conditions_lst:
            if cond['type'] == 'periodic':
                cond_lst = []
                for bnd in cond['bnd']:
                    cond_lst.append(self._bnd_grid(bnd, variable_dict, dtype))
                cond['bnd'] = cond_lst
            else:
                cond['bnd'] = self._bnd_grid(cond['bnd'], variable_dict, dtype)

            if isinstance(cond['bval'], torch.Tensor):
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
        # STEP 4
        self.equation_lst = []

    def add(self, eq: dict):
        """ add equation

        Args:
            eq (dict): equation in operator form.
        """
        # STEP 5
        self.equation_lst.append(eq)


class Points_type():
    """
    Discretizing the grid and allocating subsets for Finite Difference method.
    """

    def __init__(self, grid: torch.Tensor):
        """
        Args:
            grid (torch.Tensor): discretization points of comp-l domain.
        """

        self.grid = grid

    @staticmethod
    def shift_points(grid: torch.Tensor, axis: int, shift: float) -> torch.Tensor:
        """ Shifts all values of an array 'grid' on a value 'shift' in a direction of
        axis 'axis', somewhat is equivalent to a np.roll.

        Args:
            grid (torch.Tensor): discretization of comp-l domain.
            axis (int): axis to which the shift is applied.
            shift (float): shift value.

        Returns:
            torch.Tensor: shifted array of a n-D points.
        """

        grid_shift = grid.clone()
        grid_shift[:, axis] = grid[:, axis] + shift
        return grid_shift

    @staticmethod
    def _in_hull(p: torch.Tensor, hull: torch.Tensor) -> np.ndarray:
        """ Test if points in `p` are in `hull`
        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed.
        Args:
            p (torch.Tensor): shifted array of a n-D points.
            hull (torch.Tensor): initial array of a n-D points.
        Returns:
            np.ndarray: array of a n-D boolean type points.
            True - if 'p' in 'hull', False - otherwise.
        """

        if p.shape[1] > 1:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull.cpu())

            return hull.find_simplex(p.cpu()) >= 0
        elif p.shape[1] == 1:
            # this one is not a snippet from a stackexchange it does the same
            # but for a 1-D case, which is not covered in a code above
            upbound = torch.max(hull).cpu()
            lowbound = torch.min(hull).cpu()
            return np.array(((p.cpu() <= upbound) & (p.cpu() >= lowbound)).reshape(-1))

    def point_typization(self) -> dict:
        """ Allocating subsets for FD (i.e., 'f', 'b', 'central').

        Returns:
            dict: type with a points in a 'grid' above. Type may be 'central' - inner point
            and string of 'f' and 'b', where the length of the string is a dimension n. 'f' means that if we add
            small number to a position of corresponding coordinate we stay in the 'hull'. 'b' means that if we
            subtract small number from o a position of corresponding coordinate we stay in the 'hull'.
        """

        direction_list = []
        for axis in range(self.grid.shape[1]):
            for direction in range(2):
                direction_list.append(
                    Points_type._in_hull(Points_type.shift_points(
                        self.grid, axis, (-1) ** direction * 0.0001), self.grid))

        direction_list = np.array(direction_list)
        direction_list = np.transpose(direction_list)

        point_type = {}

        for i, point in enumerate(self.grid):
            if np.all(direction_list[i]):
                point_type[point] = 'central'
            else:
                p_type = ''
                j = 0
                while j < len(direction_list[i]):
                    if (j % 2 == 0 and direction_list[i, j]) or (
                            j % 2 == 0 and direction_list[i, j] and
                            direction_list[i, j + 1]):
                        p_type += 'f'
                    else:
                        p_type += 'b'
                    j += 2
                if self.grid.shape[-1] == 1:
                    point_type[point] = 'central'
                else:
                    point_type[point] = p_type
        return point_type

    def grid_sort(self) -> dict:
        """ Sorting grid points for each subset from result Points_type.point_typization.

        Returns:
            dict: sorted grid in each subset (see Points_type.point_typization).
        """

        point_type = self.point_typization()
        point_types = set(point_type.values())
        grid_dict = {}
        for p_type in point_types:
            grid_dict[p_type] = []
        for point in list(point_type.keys()):
            p_type = point_type[point]
            grid_dict[p_type].append(point)
        for p_type in point_types:
            grid_dict[p_type] = torch.stack(grid_dict[p_type])
        return grid_dict

    def bnd_sort(self, grid_dict: dict, b_coord: Union[torch.Tensor, list]) -> list:
        """ Sorting boundary points

        Args:
            grid_dict (dict): _description_
            b_coord (Union[torch.Tensor, list]): boundary points of grid.
            It will be list if periodic condition is.

        Returns:
            list: bnd_dict is similar to grid_dict but with b_coord values. It
            will be list of 'bnd_dict's if 'b_coord' is list too.
        """

        def bnd_to_dict(grid_dict, b_coord):
            bnd_dict = {}
            for k, v in grid_dict.items():
                bnd_dict[k] = []
                for bnd in b_coord:
                    if ((bnd == v).all(axis=1)).any():
                        bnd_dict[k].append(bnd)
                if bnd_dict[k] == []:
                    del bnd_dict[k]
                else:
                    bnd_dict[k] = torch.stack(bnd_dict[k])
            return bnd_dict

        if isinstance(b_coord, list):
            bnd_dict_list = [bnd_to_dict(grid_dict, bnd) for bnd in b_coord]
            return bnd_dict_list
        else:
            return bnd_to_dict(grid_dict, b_coord)


flatten_list = lambda t: [item for sublist in t for item in sublist]


class First_order_scheme():
    """Class for numerical scheme construction. Central o(h^2) difference scheme
    is used for 'central' points, forward ('f') and backward ('b') o(h) schemes
    are used for boundary points. 'central', and combination 'f','b' are
    corresponding to points_type.

    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2
            if x is first direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'
        """

        self.term = term
        self.nvars = nvars
        if axes_scheme_type == 'central':
            self.direction_list = ['central' for _ in self.term]
        else:
            self.direction_list = [axes_scheme_type[i] for i in self.term]

    # the idea is simple - central difference changes
    # [0]->([1]-[-1])/(2h) (in terms of grid nodes position)
    @staticmethod
    def _finite_diff_shift(diff: list, axis: int, mode: str) -> list:
        """ 1st order points shift for the corresponding finite difference mode.

        Args:
            diff (list): values of finite differences.
            axis (int): axis.
            mode (str): the finite difference mode (i.e., forward, backward, central).

        Returns:
            list: list with shifted points.
        """

        diff_p = copy(diff)
        diff_m = copy(diff)
        if mode == 'central':
            diff_p[axis] = diff_p[axis] + 1
            diff_m[axis] = diff_m[axis] - 1
        elif mode == 'f':
            diff_p[axis] = diff_p[axis] + 1
        elif mode == 'b':
            diff_m[axis] = diff_m[axis] - 1
        return [diff_p, diff_m]

    def scheme_build(self) -> list:
        """ Building first order (in terms of accuracy) finite-difference scheme.
        Start from list of zeros where them numbers equal nvars. After that we
        move value in that axis which corresponding to term. [0,0]->[[1,0],[-1,0]]
        it means that term was [0] (d/dx) and mode (scheme_type) is 'central'.

        Returns:
            list: numerical scheme.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._finite_diff_shift(
                    diff, self.term[i], self.direction_list[i])

                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1 / 2) -> list:
        """ Determines the sign of the derivative for the corresponding transformation
        from Finite_diffs.scheme_build().

        From transformations above, we always start from +1 (1)
        Every +1 changes to ->[+1,-1] when order of differential rises
        [0,0] (+1) ->([1,0]-[-1,0]) ([+1,-1])
        Every -1 changes to [-1,+1]
        [[1,0],[-1,0]] ([+1,-1])->[[1,1],[1,-1],[-1,1],[-1,-1]] ([+1,-1,-1,+1])

        Args:
            h (float, optional): discretizing parameter in finite-
            difference method. Defaults to 1/2.

        Returns:
            list: list, with signs for corresponding points.
        """

        sign_list = [1]
        for _ in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if np.unique(self.direction_list)[0] == 'central':
                    start_list.append([sign * (1 / (2 * h)),
                                       -sign * (1 / (2 * h))])
                else:
                    start_list.append([sign / h, -sign / h])
            sign_list = flatten_list(start_list)
        return sign_list


class Second_order_scheme():
    """
    Crank–Nicolson method. This realization only for boundary points.
    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2 if x is first
                    direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'

        Raises:
            ValueError: _description_
        """

        self.term = term
        self.nvars = nvars
        try:
            axes_scheme_type == 'central'
        except:
            print('These scheme only for "f" and "b" points')
            raise ValueError
        self.direction_list = [axes_scheme_type[i] for i in self.term]

    @staticmethod
    def _second_order_shift(diff, axis, mode) -> list:
        """ 2st order points shift for the corresponding finite difference mode.

        Args:
            diff (list): values of finite differences.
            axis (int): axis.
            mode (str): the finite difference mode (i.e., forward, backward).

        Returns:
            list: list with shifted points.
        """
        diff_1 = copy(diff)
        diff_2 = copy(diff)
        diff_3 = copy(diff)
        if mode == 'f':
            diff_3[axis] = diff_3[axis] + 2
            diff_2[axis] = diff_2[axis] + 1
        elif mode == 'b':
            diff_3[axis] = diff_3[axis] - 2
            diff_2[axis] = diff_2[axis] - 1
        else:
            print('Wrong mode')
        return [diff_3, diff_2, diff_1]

    def scheme_build(self) -> list:
        """Scheme building for Crank-Nicolson variant, it's identical to
        'scheme_build' in first order method, but value is shifted by
        'second_order_shift'.

        Returns:
            list: numerical scheme list.
        """

        order = len(self.term)
        finite_diff = [[0 for _ in range(self.nvars)]]
        # when we increase differential order
        for i in range(order):
            diff_list = []
            for diff in finite_diff:
                # we use [0,0]->[[1,0],[-1,0]] rule for the axis
                f_diff = self._second_order_shift(
                    diff, self.term[i], self.direction_list[i])
                if len(diff_list) == 0:
                    # and put it to the pool of differentials if it is empty
                    diff_list = f_diff
                else:
                    # or add to the existing pool
                    for diffs in f_diff:
                        diff_list.append(diffs)
            # then we go to the next differential if needed
            finite_diff = diff_list
        return finite_diff

    def sign_order(self, h: float = 1 / 2) -> list:
        """ Signs definition for second order schemes.

        Args:
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 1/2.

        Returns:
            list: list, with signs for corresponding points.
        """

        sign_list = [1]
        for i in range(len(self.term)):
            start_list = []
            for sign in sign_list:
                if self.direction_list[i] == 'f':
                    start_list.append([3 * (1 / (2 * h)) * sign,
                                       -4 * (1 / (2 * h)) * sign,
                                       (1 / (2 * h)) * sign])
                elif self.direction_list[i] == 'b':
                    start_list.append([-3 * (1 / (2 * h)) * sign,
                                       4 * (1 / (2 * h)) * sign,
                                       -(1 / (2 * h)) * sign])
            sign_list = flatten_list(start_list)
        return sign_list


class Finite_diffs():
    """
    Class for numerical scheme choosing.
    """

    def __init__(self, term: list, nvars: int, axes_scheme_type: str):
        """
        Args:
            term (list): differentiation direction. Example: [0,0]->d2u/dx2 if x is first
                    direction in the grid.
            nvars (int): task parameters. Example: if grid(x,t) -> nvars = 2.
            axes_scheme_type (str): scheme type: 'central' or combination of 'f' and 'b'
        """

        self.term = term
        self.nvars = nvars
        self.axes_scheme_type = axes_scheme_type

    def scheme_choose(self, scheme_label: str, h: float = 1 / 2) -> list:
        """ Method for numerical scheme choosing via realized above.

        Args:
            scheme_label (str): '2'- for second order scheme (only boundaries points),
                '1' - for first order scheme.
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 1/2.

        Returns:
            list: list where list[0] is numerical scheme and list[1] is signs.
        """

        if self.term == [None]:
            return [[None], [1]]
        elif scheme_label == '2':
            cl_scheme = Second_order_scheme(self.term, self.nvars,
                                            self.axes_scheme_type)
        elif scheme_label == '1':
            cl_scheme = First_order_scheme(self.term, self.nvars,
                                           self.axes_scheme_type)

        scheme = cl_scheme.scheme_build()
        sign = cl_scheme.sign_order(h=h)
        return [scheme, sign]


class Equation_NN(EquationMixin, Points_type):
    """
    Class for preprocessing input data: grid, operator, bconds in unified
    form. Then it will be used for determine solution by 'NN' method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str = '1',
                 boundary_order: str = '2'):
        """ Prepares equation, boundary conditions for *NN* mode.

        Args:
            grid (torch.Tensor): tensor of a n-D points.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions.
            h (float, optional): discretizing parameter in finite difference
            method(i.e., grid resolution for scheme). Defaults to 0.001.
            inner_order (str, optional): accuracy inner order for finite difference.
            Defaults to '1'.
            boundary_order (str, optional):accuracy boundary order for finite difference.
            Defaults to '2'.
        """

        super().__init__(grid)
        self.grid = grid
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def _operator_to_type_op(self,
                             dif_direction: list,
                             nvars: int,
                             axes_scheme_type: str) -> list:
        """ Function serves applying different schemes to a different point types
        for entire differentiation direction.

        Args:
            dif_direction (list): differentiation direction, (example:d2/dx2->[[0,0]])
            nvars (int): dimensionality of the problem.
            axes_scheme_type (str): 'central' or combination of 'f' and 'b'.

        Returns:
            list: list, where the conventional operator changed to
            steps and signs (see scheme_build function description).
        """
        if axes_scheme_type == 'central':
            scheme_variant = self.inner_order
        else:
            scheme_variant = self.boundary_order

        fin_diff_list = []
        s_order_list = []
        for term in dif_direction:
            scheme, s_order = Finite_diffs(
                term, nvars, axes_scheme_type).scheme_choose(
                scheme_variant, h=self.h)
            fin_diff_list.append(scheme)
            s_order_list.append(s_order)
        return [fin_diff_list, s_order_list]

    def _finite_diff_scheme_to_grid_list(self,
                                         finite_diff_scheme: list,
                                         grid_points: torch.Tensor) -> list:
        """ Method that converts integer finite difference steps in term described
        in Finite_diffs class to a grids with shifted points, i.e.
        from field (x,y) -> (x,y+h).

        Args:
            finite_diff_scheme (list): operator_to_type_op one term.
            grid_points (torch.Tensor): grid points that will be shifted
            corresponding to finite diff scheme.

        Returns:
            list: list, where the steps and signs changed to grid and signs.
        """

        s_grid_list = []
        for shifts in finite_diff_scheme:
            if shifts is None:
                s_grid_list.append(grid_points)
            else:
                s_grid = grid_points
                for j, axis in enumerate(shifts):
                    s_grid = self.shift_points(s_grid, j, axis * self.h)
                s_grid_list.append(s_grid)
        return s_grid_list

    def _checking_coeff(self,
                        coeff: Union[int, float, torch.Tensor, callable],
                        grid_points: torch.Tensor) -> torch.Tensor:
        """ Checks the coefficient type

        Args:
            coeff (Union[int, float, torch.Tensor, callable]): coefficient
            in equation operator.
            grid_points (torch.Tensor): if coeff is callable or torch.Tensor

        Raises:
            NameError: coeff" should be: torch.Tensor or callable or int or float!

        Returns:
            torch.Tensor: coefficient
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = (coeff, grid_points)
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            pos = self.bndpos(self.grid, grid_points)
            coeff1 = coeff[pos].reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _type_op_to_grid_shift_op(self, fin_diff_op: list, grid_points) -> list:
        """ Converts operator to a grid_shift form. Includes term coefficient
        conversion.
        Coeff may be integer, function or array, last two are mapped to a
        subgrid that corresponds point type.

        Args:
            fin_diff_op (list): operator_to_type_op result.
            grid_points (_type_): grid points that will be shifted
            corresponding to finite diff scheme.

        Returns:
            list: final form of differential operator used in the algorithm for
            single grid type.
        """

        shift_grid_op = []
        for term1 in fin_diff_op:
            grid_op = self._finite_diff_scheme_to_grid_list(term1, grid_points)
            shift_grid_op.append(grid_op)
        return shift_grid_op

    def _one_operator_prepare(self,
                              operator: dict,
                              grid_points: torch.Tensor,
                              points_type: str) -> dict:
        """ Method for operator preparing, there is construct all predefined
        methods.

        Args:
            operator (dict): operator in input form.
            grid_points (torch.Tensor): see type_op_to_grid_shift_op method.
            points_type (str): points type of grid_points.

        Returns:
            dict: prepared operator
        """

        nvars = self.grid.shape[-1]
        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            dif_term = list(term.keys())[1]
            term['coeff'] = self._checking_coeff(term['coeff'], grid_points)
            term[dif_term] = self._operator_to_type_op(term[dif_term],
                                                       nvars, points_type)
            term[dif_term][0] = self._type_op_to_grid_shift_op(
                term[dif_term][0], grid_points)
        return operator

    def operator_prepare(self) -> list:
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list: list of dictionaries, where every dictionary is the result of
            'one_operator_prepare'
        """

        grid_points = self.grid_sort()['central']
        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self._one_operator_prepare(
                    self.operator[i], grid_points, 'central')
                prepared_operator.append(equation)
        else:
            equation = self._one_operator_prepare(
                self.operator, grid_points, 'central')
            prepared_operator = [equation]

        return prepared_operator

    def _apply_bnd_operators(self, bnd_operator: dict, bnd_dict: dict) -> list:
        """ Method for applying boundary operator for all points type in bnd_dict.

        Args:
            bnd_operator (dict): boundary operator in input form.
            bnd_dict (dict): dictionary (keys is points type, values is boundary points).

        Returns:
            list: final form of differential operator used in the algorithm for
            subset grid types.
        """

        operator_list = []
        for points_type in list(bnd_dict.keys()):
            equation = self._one_operator_prepare(
                deepcopy(bnd_operator), bnd_dict[points_type], points_type)
            operator_list.append(equation)
        return operator_list

    def bnd_prepare(self) -> list:
        """ Method for boundary conditions preparing to final form.

        Returns:
            list: list of dictionaries where every dict is one boundary condition
        """

        grid_dict = self.grid_sort()

        for bcond in self.bconds:
            bnd_dict = self.bnd_sort(grid_dict, bcond['bnd'])
            if bcond['bop'] is not None:
                if bcond['type'] == 'periodic':
                    bcond['bop'] = [self._apply_bnd_operators(
                        bcond['bop'], i) for i in bnd_dict]
                else:
                    bcond['bop'] = self._apply_bnd_operators(
                        bcond['bop'], bnd_dict)
        return self.bconds


class Equation_autograd(EquationMixin):
    """
    Prepares equation for autograd method (i.e., from conventional form to input form).
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list):
        """ Prepares equation for autograd method
        (i.e., from conventional form to input form).

        Args:
            grid (torch.Tensor): tensor of a n-D points.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions in input form.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def _checking_coeff(self,
                        coeff: Union[int, float, torch.Tensor]) -> Union[int, float, torch.Tensor]:
        """ Checks the coefficient type

        Args:
            coeff (Union[int, float, torch.Tensor]): coefficient in equation operator.

        Raises:
            NameError: "coeff" should be: torch.Tensor or callable or int or float!

        Returns:
            Union[int, float, torch.Tensor]: coefficient
        """

        if isinstance(coeff, (int, float)):
            coeff1 = coeff
        elif callable(coeff):
            coeff1 = coeff
        elif isinstance(coeff, torch.Tensor):
            coeff = check_device(coeff)
            coeff1 = coeff.reshape(-1, 1)
        elif isinstance(coeff, torch.nn.parameter.Parameter):
            coeff1 = coeff
        else:
            raise NameError('"coeff" should be: torch.Tensor or callable or int or float!')
        return coeff1

    def _one_operator_prepare(self, operator: dict) -> dict:
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Args:
            operator (dict): operator in input form.

        Returns:
            dict: dict, where coeff is checked.
        """

        operator = self.equation_unify(operator)
        for operator_label in operator:
            term = operator[operator_label]
            term['coeff'] = self._checking_coeff(term['coeff'])
        return operator

    def operator_prepare(self) -> list:
        """ Method for all operators preparing. If system case is, it will call
        'one_operator_prepare' method for number of equations times.

        Returns:
            list: list of dictionaries, where every dictionary is the result of
            'one_operator_prepare'
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(self._one_operator_prepare(equation))
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [self._one_operator_prepare(equation)]

        return prepared_operator

    def bnd_prepare(self) -> list:
        """ Method for boundary conditions preparing to final form

        Returns:
            list: list of dictionaries where every dict is one boundary condition
        """

        if self.bconds is None:
            return None
        else:
            return self.bconds


class Equation_mat(EquationMixin):
    """
    Class realizes input data preprocessing (operator and boundary conditions
    preparing) for 'mat' method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[list, dict],
                 bconds: list):
        """ Prepares equation for autograd method
        (i.e., from conventional form to input form).

        Args:
            grid (torch.Tensor): grid, result of meshgrid.
            operator (Union[list, dict]): operator in input form.
            bconds (list): boundary conditions in input form.
        """

        self.grid = grid
        self.operator = operator
        self.bconds = bconds

    def operator_prepare(self) -> list:
        """ Method realizes operator preparing for 'mat' method
        using only 'equation_unify' method.
        Returns:
            list: final form of differential operator used in the algorithm.
        """

        if isinstance(self.operator, list) and isinstance(self.operator[0], dict):
            num_of_eq = len(self.operator)
            prepared_operator = []
            for i in range(num_of_eq):
                equation = self.equation_unify(self.operator[i])
                prepared_operator.append(equation)
        else:
            equation = self.equation_unify(self.operator)
            prepared_operator = [equation]

        return prepared_operator

    def _point_position(self, bnd: torch.Tensor) -> list:
        """ Define position of boundary points on the grid.

        Args:
            bnd (torch.Tensor): boundary subgrid.

        Returns:
            list: list of positions, where boundary points intersects on the grid.
        """

        bpos = []
        for pt in bnd:
            if self.grid.shape[0] == 1:
                point_pos = (torch.tensor(self.bndpos(self.grid, pt)),)
            else:
                prod = (torch.zeros_like(self.grid[0]) + 1).bool()
                for axis in range(self.grid.shape[0]):
                    axis_intersect = torch.isclose(
                        pt[axis].float(), self.grid[axis].float())
                    prod *= axis_intersect
                point_pos = torch.where(prod == True)
            bpos.append(point_pos)
        return bpos

    def bnd_prepare(self) -> list:
        """ Method for boundary conditions preparing to final form.

        Returns:
            list: list of dictionaries where every dict is one boundary condition.
        """

        for bcond in self.bconds:
            if bcond['type'] == 'periodic':
                bpos = []
                for bnd in bcond['bnd']:
                    bpos.append(self._point_position(bnd))
            else:
                bpos = self._point_position(bcond['bnd'])
            if bcond['bop'] is not None:
                bcond['bop'] = self.equation_unify(bcond['bop'])
            bcond['bnd'] = bpos
        return self.bconds


class Operator_bcond_preproc():
    """
    Interface for preparing equations due to chosen calculation method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str = '1',
                 boundary_order: str = '2'):
        """_summary_

        Args:
            grid (torch.Tensor): grid from cartesian_prod or meshgrid result.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions.
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 0.001.
            inner_order (str, optional): accuracy inner order for finite difference. Defaults to '1'.
            boundary_order (str, optional): accuracy boundary order for finite difference. Defaults to '2'.
        """

        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """ Setting the calculation method.

        Args:
            strategy (str): Calculation method. (i.e., "NN", "autograd", "mat").

        Returns:
            Union[Equation_NN, Equation_mat, Equation_autograd]: A given calculation method.
        """

        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h,
                               inner_order=self.inner_order,
                               boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)


class Operator_bcond_preproc():
    """
    Interface for preparing equations due to chosen calculation method.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 operator: Union[dict, list],
                 bconds: list,
                 h: float = 0.001,
                 inner_order: str = '1',
                 boundary_order: str = '2'):
        """_summary_

        Args:
            grid (torch.Tensor): grid from cartesian_prod or meshgrid result.
            operator (Union[dict, list]): equation.
            bconds (list): boundary conditions.
            h (float, optional): discretizing parameter in finite-
            difference method (i.e., grid resolution for scheme). Defaults to 0.001.
            inner_order (str, optional): accuracy inner order for finite difference. Defaults to '1'.
            boundary_order (str, optional): accuracy boundary order for finite difference. Defaults to '2'.
        """

        self.grid = check_device(grid)
        self.operator = operator
        self.bconds = bconds
        self.h = h
        self.inner_order = inner_order
        self.boundary_order = boundary_order

    def set_strategy(self, strategy: str) -> Union[Equation_NN, Equation_mat, Equation_autograd]:
        """ Setting the calculation method.

        Args:
            strategy (str): Calculation method. (i.e., "NN", "autograd", "mat").

        Returns:
            Union[Equation_NN, Equation_mat, Equation_autograd]: A given calculation method.
        """

        if strategy == 'NN':
            return Equation_NN(self.grid, self.operator, self.bconds, h=self.h,
                               inner_order=self.inner_order,
                               boundary_order=self.boundary_order)
        if strategy == 'mat':
            return Equation_mat(self.grid, self.operator, self.bconds)
        if strategy == 'autograd':
            return Equation_autograd(self.grid, self.operator, self.bconds)


class Callback(ABC):
    """Base class used to build new callbacks.
    """

    def __init__(self):
        self.print_every = None
        self.verbose = 0
        self.validation_data = None
        self._model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def on_epoch_begin(self, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_epoch_end(self, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        pass

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        pass

    def during_epoch(self, logs=None):
        pass


class CallbackList(Callback):
    """Container abstracting a list of callbacks."""

    def __init__(
            self,
            callbacks=None,
            model=None,
            **params,
    ):
        """Container for `Callback` instances.

        This object wraps a list of `Callback` instances, making it possible
        to call them all at once via a single endpoint
        (e.g. `callback_list.on_epoch_end(...)`).

        Args:
            callbacks: List of `Callback` instances.
            model: The `Model` these callbacks are used with.
            **params: If provided, parameters will be passed to each `Callback`
                via `Callback.set_params`.
        """
        self.callbacks = callbacks if callbacks else []

        if model:
            self.set_model(model)
        if params:
            self.set_params(params)

    def set_model(self, model):
        super().set_model(model)
        for callback in self.callbacks:
            callback.set_model(model)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(logs)

    def on_epoch_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)


def create_random_fn(eps: float) -> callable:
    """ Create random tensors to add some variance to torch neural network.

    Args:
        eps (float): randomize parameter.

    Returns:
        callable: creating random params function.
    """

    def randomize_params(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            m.weight.data = m.weight.data + \
                            (2 * torch.randn(m.weight.size()) - 1) * eps
            m.bias.data = m.bias.data + (2 * torch.randn(m.bias.size()) - 1) * eps

    return randomize_params


def samples_count(second_order_interactions: bool,
                  sampling_N: int,
                  op_length: list,
                  bval_length: list) -> Tuple[int, int]:
    """ Count samples for variance based sensitivity analysis.

    Args:
        second_order_interactions (bool): Calculate second-order sensitivities.
        sampling_N (int): essentially determines how often the lambda will be re-evaluated.
        op_length (list): operator values length.
        bval_length (list): boundary value length.

    Returns:
        sampling_amount (int): overall sampling value.
        sampling_D (int): sum of length of grid and boundaries.
    """

    grid_len = sum(op_length)
    bval_len = sum(bval_length)

    sampling_D = grid_len + bval_len

    if second_order_interactions:
        sampling_amount = sampling_N * (2 * sampling_D + 2)
    else:
        sampling_amount = sampling_N * (sampling_D + 2)
    return sampling_amount, sampling_D


def lambda_print(lam: torch.Tensor, keys: List) -> None:
    """ Print lambda value.

    Args:
        lam (torch.Tensor): lambdas values.
        keys (List): types of lambdas.
    """

    lam = lam.reshape(-1)
    for val, key in zip(lam, keys):
        print('lambda_{}: {}'.format(key, val.item()))


def bcs_reshape(
        bval: torch.Tensor,
        true_bval: torch.Tensor,
        bval_length: List) -> Tuple[dict, dict, dict, dict]:
    """ Preprocessing for lambda evaluating.

    Args:
        bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
        true_bval (torch.Tensor): matrix, where each column is true
                            boundary values of one boundary type.
        bval_length (list): list of length of each boundary type column.

    Returns:
        torch.Tensor: vector of difference between bval and true_bval.
    """

    bval_diff = bval - true_bval

    bcs = torch.cat([bval_diff[0:bval_length[i], i].reshape(-1)
                     for i in range(bval_diff.shape[-1])])

    return bcs


def remove_all_files(folder: str) -> None:
    """ Remove all files from folder.

    Args:
        folder (str): folder name.
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mat_op_coeff(equation: Any) -> Any:
    """ Preparation of coefficients in the operator of the *mat* method
        to suit methods *NN, autograd*.

    Args:
        operator (dict): operator (equation dict).

    Returns:
        operator (dict): operator (equation dict) with suitable coefficients.
    """

    for op in equation.equation_lst:
        for label in list(op.keys()):
            term = op[label]
            if isinstance(term['coeff'], torch.Tensor):
                term['coeff'] = term['coeff'].reshape(-1, 1)
            elif callable(term['coeff']):
                print("Warning: coefficient is callable,\
                                it may lead to wrong cache item choice")
    return equation


def model_mat(model: torch.Tensor,
              domain: Any,
              cache_model: torch.nn.Module = None) -> Tuple[torch.Tensor, torch.nn.Module]:
    """ Create model for *NN or autograd* modes from grid
        and model of *mat* mode.

    Args:
        model (torch.Tensor): model from *mat* method.
        grid (torch.Tensor): grid from *mat* method.
        cache_model (torch.nn.Module, optional): neural network that will
                                                    approximate *mat* model. Defaults to None.

    Returns:
        cache_model (torch.nn.Module): model satisfying the *NN, autograd* methods.
    """
    grid = domain.build('mat')
    input_model = grid.shape[0]
    output_model = model.shape[0]

    if cache_model is None:
        cache_model = torch.nn.Sequential(
            torch.nn.Linear(input_model, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, 100),
            torch.nn.Tanh(),
            torch.nn.Linear(100, output_model)
        )

    return cache_model


def save_model_nn(
        cache_dir: str,
        model: torch.nn.Module,
        name: Union[str, None] = None) -> None:
    """
    Saves model in a cache (uses for 'NN' and 'autograd' methods).
    Args:
        cache_dir (str): path to cache folder.
        model (torch.nn.Module): model to save.
        (uses only with mixed precision and device=cuda). Defaults to None.
        name (str, optional): name for a model. Defaults to None.
    """

    if name is None:
        name = str(datetime.datetime.now().timestamp())
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    parameters_dict = {'model': model.to('cpu'),
                       'model_state_dict': model.state_dict()}

    try:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar')
        print(f'model is saved in cache dir: {cache_dir}')
    except RuntimeError:
        torch.save(parameters_dict, cache_dir + '\\' + name + '.tar',
                   _use_new_zipfile_serialization=False)  # cyrillic in path
        print(f'model is saved in cache: {cache_dir}')
    except:
        print(f'Cannot save model in cache: {cache_dir}')


def save_model_mat(cache_dir: str,
                   model: torch.Tensor,
                   domain: Any,
                   cache_model: Union[torch.nn.Module, None] = None,
                   name: Union[str, None] = None) -> None:
    """ Saves model in a cache (uses for 'mat' method).

    Args:
        cache_dir (str): path to cache folder.
        model (torch.Tensor): *mat* model
        grid (torch.Tensor): grid from *mat* mode
        cache_model (Union[torch.nn.Module, None], optional): model to save. Defaults to None.
        name (Union[str, None], optional): name for a model. Defaults to None.
    """

    net_autograd = model_mat(model, domain, cache_model)
    nn_grid = domain.build('autograd')
    optimizer = torch.optim.Adam(net_autograd.parameters(), lr=0.001)
    model_res = model.reshape(-1, model.shape[0])

    def closure():
        optimizer.zero_grad()
        loss = torch.mean((net_autograd(check_device(nn_grid)) - model_res) ** 2)
        loss.backward()
        return loss

    loss = np.inf
    t = 0
    while loss > 1e-5 and t < 1e5:
        loss = optimizer.step(closure)
        t += 1
        print('Interpolate from trained model t={}, loss={}'.format(
            t, loss))

    save_model_nn(cache_dir, net_autograd, name=name)


class PadTransform(Module):
    """Pad tensor to a fixed length with given padding value.

    src: https://pytorch.org/text/stable/transforms.html#torchtext.transforms.PadTransform

    Done to avoid torchtext dependency (we need only this function).
    """

    def __init__(self, max_length: int, pad_value: int) -> None:
        """_summary_

        Args:
            max_length (int): Maximum length to pad to.
            pad_value (int):  Value to pad the tensor with.
        """
        super().__init__()
        self.max_length = max_length
        self.pad_value = float(pad_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Tensor padding

        Args:
            x (torch.Tensor): tensor for padding.

        Returns:
            torch.Tensor: filled tensor with pad value.
        """

        max_encoded_length = x.size(-1)
        if max_encoded_length < self.max_length:
            pad_amount = self.max_length - max_encoded_length
            x = torch.nn.functional.pad(x, (0, pad_amount), value=self.pad_value)
        return x


flatten_list = lambda t: [item for sublist in t for item in sublist]


def solver_device(device: str):
    """ Corresponding to chosen device, all futher
        created tensors will be with the same device

    Args:
        device (str): device mode, **cuda, gpu, cpu*.

    """
    if device in ['cuda', 'gpu'] and torch.cuda.is_available():
        print('CUDA is available and used.')
        return torch.set_default_device('cuda')
    elif device in ['cuda', 'gpu'] and not torch.cuda.is_available():
        print('CUDA is not available, cpu is used!')
        return torch.set_default_device('cpu')
    else:
        print('Default cpu processor is used.')
        return torch.set_default_device('cpu')


def check_device(data: Any):
    """ checking the device of the data.
        If the data.device is not same with torch.set_default_device,
        change one.
    Args:
        data (Any): it could be model or torch.Tensors

    Returns:
        data (Any): data with correct device
    """
    device = torch.tensor([0.]).device.type
    if data.device.type != device:
        return data.to(device)
    else:
        return data


def device_type():
    """ Return the default device.
    """
    return torch.tensor([0.]).device.type


def integration(func: torch.Tensor,
                grid: torch.Tensor,
                power: int = 2) \
        -> Union[Tuple[float, float], Tuple[list, torch.Tensor]]:
    """ Function realize 1-space integrands,
    where func=(L(u)-f)*weak_form subintegrands function and
    definite integral parameter is grid.

    Args:
        func (torch.Tensor): operator multiplied on test function
        grid (torch.Tensor): array of a n-D points.
        power (int, optional): power of func points. Defults to 2.

    Returns:
        'result' is integration result through one grid axis
        'grid' is initial grid without last column or zero (if grid.shape[N,1])
    """
    if grid.shape[-1] == 1:
        column = -1
    else:
        column = -2
    marker = grid[0][column]
    index = [0]
    result = []
    u = 0.
    for i in range(1, len(grid)):
        if grid[i][column] == marker or column == -1:
            u += (grid[i][-1] - grid[i - 1][-1]).item() * \
                 (func[i] ** power + func[i - 1] ** power) / 2
        else:
            result.append(u)
            marker = grid[i][column]
            index.append(i)
            u = 0.
    if column == -1:
        return u, 0.
    else:
        result.append(u)
        grid = grid[index, :-1]
        return result, grid


def dict_to_matrix(bval: dict, true_bval: dict) \
        -> Tuple[torch.Tensor, torch.Tensor, List, List]:
    """ Function for bounaries values matrix creation from dictionary.

    Args:
        bval (dict): dictionary with predicted boundaries values,
              where keys are boundaries types.
        true_bval (dict): dictionary with true boundaries values,
                   where keys are boundaries types.

    Returns:
        matrix_bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
        matrix_true_bval (torch.Tensor):matrix, where each column is true
                           boundary values of one boundary type.
        keys (list): boundary types list corresponding matrix_bval columns.
        len_list (list): list of length of each boundary type column.
    """

    keys = list(bval.keys())
    max_len = max([len(i) for i in bval.values()])
    pad = PadTransform(max_len, 0)
    matrix_bval = pad(bval[keys[0]]).reshape(-1, 1)
    matrix_true_bval = pad(true_bval[keys[0]]).reshape(-1, 1)
    len_list = [len(bval[keys[0]])]
    for key in keys[1:]:
        bval_i = pad(bval[key]).reshape(-1, 1)
        true_bval_i = pad(true_bval[key]).reshape(-1, 1)
        matrix_bval = torch.hstack((matrix_bval, bval_i))
        matrix_true_bval = torch.hstack((matrix_true_bval, true_bval_i))
        len_list.append(len(bval[key]))

    return matrix_bval, matrix_true_bval, keys, len_list


class DerivativeInt():
    """Interface class
    """

    def take_derivative(self, value):
        """Method that should be built in every child class"""
        raise NotImplementedError


class Derivative_NN(DerivativeInt):
    """
    Taking numerical derivative for 'NN' method.
    """

    def __init__(self, model: Any):
        """
        Args:
            model: neural network.
        """
        self.model = model

    def take_derivative(self, term: Union[list, int, torch.Tensor], *args) -> torch.Tensor:
        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (Union[list, int, torch.Tensor]): differential operator in conventional form.
        Returns:
            torch.Tensor: resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        if isinstance(term['coeff'], tuple):
            coeff = term['coeff'][0](term['coeff'][1]).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, scheme in enumerate(term[dif_dir][0]):
            grid_sum = 0.
            for k, grid in enumerate(scheme):
                grid_sum += self.model(grid)[:, term['var'][j]].reshape(-1, 1) \
                            * term[dif_dir][1][j][k]
            der_term = der_term * grid_sum ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_autograd(DerivativeInt):
    """
    Taking numerical derivative for 'autograd' method.
    """

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model (torch.nn.Module): model of *autograd* mode.
        """
        self.model = model

    @staticmethod
    def _nn_autograd(model: torch.nn.Module,
                     points: torch.Tensor,
                     var: int,
                     axis: List[int] = [0]):
        """ Computes derivative on the grid using autograd method.

        Args:
            model (torch.nn.Module): torch neural network.
            points (torch.Tensor): points, where numerical derivative is calculated.
            var (int): number of dependent variables (for single equation is *0*)
            axis (list, optional): term of differentiation, example [0,0]->d2/dx2
                                   if grid_points(x,y). Defaults to [0].

        Returns:
            gradient_full (torch.Tensor): the result of desired function differentiation
                in corresponding axis.
        """

        points.requires_grad = True
        fi = model(points)[:, var].sum(0)
        for ax in axis:
            grads, = torch.autograd.grad(fi, points, create_graph=True)
            fi = grads[:, ax].sum()
        gradient_full = grads[:, axis[-1]].reshape(-1, 1)
        return gradient_full

    def take_derivative(self, term: dict, grid_points: torch.Tensor) -> torch.Tensor:
        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (dict): differential operator in conventional form.
            grid_points (torch.Tensor): points, where numerical derivative is calculated.

        Returns:
            der_term (torch.Tensor): resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        # it is may be int, function of grid or torch.Tensor
        if callable(term['coeff']):
            coeff = term['coeff'](grid_points).reshape(-1, 1)
        else:
            coeff = term['coeff']

        der_term = 1.
        for j, derivative in enumerate(term[dif_dir]):
            if derivative == [None]:
                der = self.model(grid_points)[:, term['var'][j]].reshape(-1, 1)
            else:
                der = self._nn_autograd(
                    self.model, grid_points, term['var'][j], axis=derivative)
            der_term = der_term * der ** term['pow'][j]
        der_term = coeff * der_term

        return der_term


class Derivative_mat(DerivativeInt):
    """
    Taking numerical derivative for 'mat' method.
    """

    def __init__(self, model: torch.Tensor, derivative_points: int):
        """
        Args:
            model (torch.Tensor): model of *mat* mode.
            derivative_points (int): points number for derivative calculation.
        """
        self.model = model
        self.backward, self.farward = Derivative_mat._labels(derivative_points)

        self.alpha_backward = Derivative_mat._linear_system(self.backward)
        self.alpha_farward = Derivative_mat._linear_system(self.farward)

        num_points = int(len(self.backward) - 1)

        self.back = [int(0 - i) for i in range(1, num_points + 1)]

        self.farw = [int(i) for i in range(num_points)]

    @staticmethod
    def _labels(derivative_points: int) -> Tuple[List, List]:
        """ Determine which points are used in derivative calc-n.
            If derivative_points = 2, it return ([-1, 0], [0, 1])

        Args:
            derivative_points (int): points number for derivative calculation.

        Returns:
            labels_backward (list): points labels for backward scheme.
            labels_forward (list): points labels for forward scheme.
        """
        labels_backward = list(i for i in range(-derivative_points + 1, 1))
        labels_farward = list(i for i in range(derivative_points))
        return labels_backward, labels_farward

    @staticmethod
    def _linear_system(labels: list) -> np.ndarray:
        """ To caclulate coeeficints in numerical scheme,
            we have to solve the linear system of algebraic equations.
            A*alpha=b

        Args:
            labels (list): points labels for backward/foraward scheme.

        Returns:
            alpha (np.ndarray): coefficints for numerical scheme.
        """
        points_num = len(labels)  # num_points=number of equations
        labels = np.array(labels)
        A = []
        for i in range(points_num):
            A.append(labels ** i)
        A = np.array(A)

        b = np.zeros_like(labels)
        b[1] = 1

        alpha = linalg.solve(A, b)

        return alpha

    def _derivative_1d(self, u_tensor: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Computes derivative in one dimension for matrix method.

        Args:
            u_tensor (torch.Tensor): dependenet varible of equation,
                                     some part of model.
            h (torch.Tensor): increment of numerical scheme.

        Returns:
            du (torch.Tensor): computed derivative along one dimension.
        """

        shape = u_tensor.shape
        u_tensor = u_tensor.reshape(-1)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)
        du[self.back] = du_back[self.back] / h
        du[self.farw] = du_farw[self.farw] / h

        du = du.reshape(shape)

        return du

    def _step_h(self, h_tensor: torch.Tensor) -> list[torch.Tensor]:
        """ Calculate increment along each axis of the grid.

        Args:
            h_tensor (torch.Tensor): grid of *mat* mode.

        Returns:
            h (list[torch.Tensor]): lsit with increment
                                    along each axis of the grid.
        """
        h = []

        nn_grid = torch.vstack([h_tensor[i].reshape(-1) for i in \
                                range(h_tensor.shape[0])]).T.float()

        for i in range(nn_grid.shape[-1]):
            axis_points = torch.unique(nn_grid[:, i])
            h.append(abs(axis_points[1] - axis_points[0]))
        return h

    def _derivative(self,
                    u_tensor: torch.Tensor,
                    h: torch.Tensor,
                    axis: int) -> torch.Tensor:
        """ Computing derivative for 'mat' method.

        Args:
            u_tensor (torch.Tensor): dependenet varible of equation,
                                     some part of model.
            h (torch.Tensor): increment of numerical scheme.
            axis (int): axis along which the derivative is calculated.

        Returns:
            du (torch.Tensor): computed derivative.
        """

        if len(u_tensor.shape) == 1 or u_tensor.shape[0] == 1:
            du = self._derivative_1d(u_tensor, h)
            return du

        pos = len(u_tensor.shape) - 1

        u_tensor = torch.transpose(u_tensor, pos, axis)

        du_back = 0
        du_farw = 0
        i = 0
        for shift_b, shift_f in zip(self.backward, self.farward):
            du_back += torch.roll(u_tensor, -shift_b) * self.alpha_backward[i]
            du_farw += torch.roll(u_tensor, -shift_f) * self.alpha_farward[i]
            i += 1
        du = (du_back + du_farw) / (2 * h)

        if pos == 1:
            du[:, self.back] = du_back[:, self.back] / h
            du[:, self.farw] = du_farw[:, self.farw] / h
        elif pos == 2:
            du[:, :, self.back] = du_back[:, :, self.back] / h
            du[:, :, self.farw] = du_farw[:, :, self.farw] / h

        du = torch.transpose(du, pos, axis)

        return du

    def take_derivative(self, term: torch.Tensor, grid_points: torch.Tensor) -> torch.Tensor:
        """ Auxiliary function serves for single differential operator resulting field
        derivation.

        Args:
            term (torch.Tensor): differential operator in conventional form.
            grid_points (torch.Tensor): grid points.

        Returns:
            der_term (torch.Tensor): resulting field, computed on a grid.
        """

        dif_dir = list(term.keys())[1]
        der_term = torch.zeros_like(self.model) + 1
        for j, scheme in enumerate(term[dif_dir]):
            prod = self.model[term['var'][j]]
            if scheme != [None]:
                for axis in scheme:
                    if axis is None:
                        continue
                    h = self._step_h(grid_points)[axis]
                    prod = self._derivative(prod, h, axis)
            der_term = der_term * prod ** term['pow'][j]
        if callable(term['coeff']) is True:
            der_term = term['coeff'](grid_points) * der_term
        else:
            der_term = term['coeff'] * der_term
        return der_term


class Derivative():
    """
   Interface for taking numerical derivative due to chosen calculation mode.

   """

    def __init__(self,
                 model: Union[torch.nn.Module, torch.Tensor],
                 derivative_points: int):
        """_summary_

        Args:
            model (Union[torch.nn.Module, torch.Tensor]): neural network or
                                        matrix depending on the selected mode.
            derivative_points (int): points number for derivative calculation.
            If derivative_points=2, numerical scheme will be ([-1,0],[0,1]),
            parameter determine number of poins in each forward and backward scheme.
        """

        self.model = model
        self.derivative_points = derivative_points

    def set_strategy(self,
                     strategy: str) -> Union[Derivative_NN, Derivative_autograd, Derivative_mat]:
        """
        Setting the calculation method.
        Args:
            strategy: Calculation method. (i.e., "NN", "autograd", "mat").
        Returns:
            equation in input form for a given calculation method.
        """
        if strategy == 'NN':
            return Derivative_NN(self.model)

        elif strategy == 'autograd':
            return Derivative_autograd(self.model)

        elif strategy == 'mat':
            return Derivative_mat(self.model, self.derivative_points)


class Operator():
    """
    Class for differential equation calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_operator: Union[list, dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int):
        """
        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_operator (Union[list,dict]): prepared (after Equation class) operator.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (list[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
        """
        self.grid = check_device(grid)
        self.prepared_operator = prepared_operator
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.derivative_points = derivative_points
        if self.mode == 'NN':
            self.grid_dict = Points_type(self.grid).grid_sort()
            self.sorted_grid = torch.cat(list(self.grid_dict.values()))
        elif self.mode in ('autograd', 'mat'):
            self.sorted_grid = self.grid
        self.derivative = Derivative(self.model,
                                     self.derivative_points).set_strategy(self.mode).take_derivative

    def apply_operator(self,
                       operator: list,
                       grid_points: Union[torch.Tensor, None]) -> torch.Tensor:
        """ Deciphers equation in a single grid subset to a field.

        Args:
            operator (list): prepared (after Equation class) operator. See
            input_preprocessing.operator_prepare()
            grid_points (Union[torch.Tensor, None]): Points, where numerical
            derivative is calculated. **Uses only in 'autograd' and 'mat' modes.**

        Returns:
            total (torch.Tensor): Decoded operator on a single grid subset.
        """

        for term in operator:
            term = operator[term]
            dif = self.derivative(term, grid_points)
            try:
                total += dif
            except NameError:
                total = dif
        return total

    def _pde_compute(self) -> torch.Tensor:
        """ Computes PDE residual.

        Returns:
            torch.Tensor: P/O DE residual.
        """

        num_of_eq = len(self.prepared_operator)
        if num_of_eq == 1:
            op = self.apply_operator(
                self.prepared_operator[0], self.sorted_grid).reshape(-1, 1)
        else:
            op_list = []
            for i in range(num_of_eq):
                op_list.append(self.apply_operator(
                    self.prepared_operator[i], self.sorted_grid).reshape(-1, 1))
            op = torch.cat(op_list, 1)
        return op

    def _weak_pde_compute(self) -> torch.Tensor:
        """ Computes PDE residual in weak form.

        Returns:
            torch.Tensor: weak PDE residual.
        """

        device = device_type()
        if self.mode == 'NN':
            grid_central = self.grid_dict['central']
        elif self.mode == 'autograd':
            grid_central = self.grid

        op = self._pde_compute()
        sol_list = []
        for i in range(op.shape[-1]):
            sol = op[:, i]
            for func in self.weak_form:
                sol = sol * func(grid_central).to(device).reshape(-1)
            grid_central1 = torch.clone(grid_central)
            for _ in range(grid_central.shape[-1]):
                sol, grid_central1 = integration(sol, grid_central1)
            sol_list.append(sol.reshape(-1, 1))
        if len(sol_list) == 1:
            return sol_list[0]
        else:
            return torch.cat(sol_list).reshape(1, -1)

    def operator_compute(self):
        """ Corresponding to form (weak or strong) calculate residual of operator.

        Returns:
            torch.Tensor: operator residual.
        """
        if self.weak_form is None or self.weak_form == []:
            return self._pde_compute()
        else:
            return self._weak_pde_compute()


class Bounds():
    """
    Class for boundary and initial conditions calculation.
    """

    def __init__(self,
                 grid: torch.Tensor,
                 prepared_bconds: Union[list, dict],
                 model: Union[torch.nn.Sequential, torch.Tensor],
                 mode: str,
                 weak_form: list[callable],
                 derivative_points: int):
        """_summary_

        Args:
            grid (torch.Tensor): grid (domain discretization).
            prepared_bconds (Union[list,dict]): prepared (after Equation class) baund-y con-s.
            model (Union[torch.nn.Sequential, torch.Tensor]): *mat or NN or autograd* model.
            mode (str): *mat or NN or autograd*
            weak_form (list[callable]): list with basis functions (if the form is *weak*).
            derivative_points (int): points number for derivative calculation.
                                     For details to Derivative_mat class.
        """
        self.grid = check_device(grid)
        self.prepared_bconds = prepared_bconds
        self.model = model.to(device_type())
        self.mode = mode
        self.operator = Operator(self.grid, self.prepared_bconds,
                                 self.model, self.mode, weak_form,
                                 derivative_points)

    def _apply_bconds_set(self, operator_set: list) -> torch.Tensor:
        """ Method only for *NN* mode. Calculate boundary conditions with derivatives
            to use them in _apply_neumann method.

        Args:
            operator_set (list): list with prepared (after Equation_NN class) boundary operators.
            For details to Equation_NN.operator_prepare method.

        Returns:
            torch.Tensor: Decoded boundary operator on the whole grid.
        """

        field_part = []
        for operator in operator_set:
            field_part.append(self.operator.apply_operator(operator, None))
        field_part = torch.cat(field_part)
        return field_part

    def _apply_dirichlet(self, bnd: torch.Tensor, var: int) -> torch.Tensor:
        """ Applies Dirichlet boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            For more deatails to input_preprocessing (bnd_prepare maethos).
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition.
        """

        if self.mode == 'NN' or self.mode == 'autograd':
            b_op_val = self.model(bnd)[:, var].reshape(-1, 1)
        # elif self.mode == 'DeepONet':
        #     b_op_val = torch.rand()[:, var].reshape(-1, 1)
        elif self.mode == 'mat':
            b_op_val = []
            for position in bnd:
                b_op_val.append(self.model[var][position])
            b_op_val = torch.cat(b_op_val).reshape(-1, 1)
        return b_op_val

    def _apply_neumann(self, bnd: torch.Tensor, bop: list) -> torch.Tensor:
        """ Applies boundary conditions with derivative operators.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.

        Returns:
            torch.Tensor: calculated boundary condition.
        """

        if self.mode == 'NN':
            b_op_val = self._apply_bconds_set(bop)
        elif self.mode == 'autograd':
            b_op_val = self.operator.apply_operator(bop, bnd)
        elif self.mode == 'mat':
            var = bop[list(bop.keys())[0]]['var'][0]
            b_op_val = self.operator.apply_operator(bop, self.grid)
            b_val = []
            for position in bnd:
                b_val.append(b_op_val[var][position])
            b_op_val = torch.cat(b_val).reshape(-1, 1)
        return b_op_val

    def _apply_periodic(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """ Applies periodic boundary conditions.

        Args:
            bnd (torch.Tensor): terms (boundary points) of prepared boundary conditions.
            bop (list): terms of prepared boundary derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the boundary condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated boundary condition
        """

        if bop is None:
            b_op_val = self._apply_dirichlet(bnd[0], var).reshape(-1, 1)
            for i in range(1, len(bnd)):
                b_op_val -= self._apply_dirichlet(bnd[i], var).reshape(-1, 1)
        else:
            if self.mode == 'NN':
                b_op_val = self._apply_neumann(bnd, bop[0]).reshape(-1, 1)
                for i in range(1, len(bop)):
                    b_op_val -= self._apply_neumann(bnd, bop[i]).reshape(-1, 1)
            elif self.mode in ('autograd', 'mat'):
                b_op_val = self._apply_neumann(bnd[0], bop).reshape(-1, 1)
                for i in range(1, len(bnd)):
                    b_op_val -= self._apply_neumann(bnd[i], bop).reshape(-1, 1)
        return b_op_val

    def _apply_data(self, bnd: torch.Tensor, bop: list, var: int) -> torch.Tensor:
        """ Method for applying known data about solution.

        Args:
            bnd (torch.Tensor): terms (data points) of prepared boundary conditions.
            bop (list): terms of prepared data derivative operator.
            var (int): indicates for which dependent variable it is necessary to apply
            the data condition. For single equation is 0.

        Returns:
            torch.Tensor: calculated data condition.
        """
        if bop is None:
            b_op_val = self._apply_dirichlet(bnd, var).reshape(-1, 1)
        else:
            b_op_val = self._apply_neumann(bnd, bop).reshape(-1, 1)
        return b_op_val

    def b_op_val_calc(self, bcond: dict) -> torch.Tensor:
        """ Auxiliary function. Serves only to choose *type* of the condition and evaluate one.

        Args:
            bcond (dict): terms of prepared boundary conditions
            (see input_preprocessing module -> bnd_prepare method).

        Returns:
            torch.Tensor: calculated operator on the boundary.
        """

        if bcond['type'] == 'dirichlet':
            b_op_val = self._apply_dirichlet(bcond['bnd'], bcond['var'])
        elif bcond['type'] == 'operator':
            b_op_val = self._apply_neumann(bcond['bnd'], bcond['bop'])
        elif bcond['type'] == 'periodic':
            b_op_val = self._apply_periodic(bcond['bnd'], bcond['bop'],
                                            bcond['var'])
        elif bcond['type'] == 'data':
            b_op_val = self._apply_data(bcond['bnd'], bcond['bop'],
                                        bcond['var'])
        return b_op_val

    def apply_bcs(self) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """ Applies boundary and data conditions for each *type* in prepared_bconds.

        Returns:
            bval (torch.Tensor): matrix, where each column is predicted
                      boundary values of one boundary type.
            true_bval (torch.Tensor):matrix, where each column is true
                            boundary values of one boundary type.
            keys (list): boundary types list corresponding matrix_bval columns.
            bval_length (list): list of length of each boundary type column.
        """

        bval_dict = {}
        true_bval_dict = {}

        for bcond in self.prepared_bconds:
            try:
                bval_dict[bcond['type']] = torch.cat((bval_dict[bcond['type']],
                                                      self.b_op_val_calc(bcond).reshape(-1)))
                true_bval_dict[bcond['type']] = torch.cat((true_bval_dict[bcond['type']],
                                                           bcond['bval'].reshape(-1)))
            except:
                bval_dict[bcond['type']] = self.b_op_val_calc(bcond).reshape(-1)
                true_bval_dict[bcond['type']] = bcond['bval'].reshape(-1)

        bval, true_bval, keys, bval_length = dict_to_matrix(
            bval_dict, true_bval_dict)

        return bval, true_bval, keys, bval_length


class Losses():
    """
    Class which contains all losses.
    """

    def __init__(self,
                 mode: str,
                 weak_form: Union[None, list],
                 n_t: int,
                 tol: Union[int, float]):
        """
        Args:
            mode (str): calculation mode, *NN, autograd, mat*.
            weak_form (Union[None, list]): list of basis functions if form is weak.
            n_t (int): number of unique points in time dinension.
            tol (Union[int, float])): penalty in *casual loss*.
        """

        self.mode = mode
        self.weak_form = weak_form
        self.n_t = n_t
        self.tol = tol
        # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
        # is None + fix causal_loss operator crutch (line 76).

    def _loss_op(self,
                 operator: torch.Tensor,
                 lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Operator term in loss calc-n.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().

            lambda_op (torch.Tensor): regularization parameter for operator term in loss.

        Returns:
            loss_operator (torch.Tensor): operator term in loss.
            op (torch.Tensor): MSE of operator on the whole grid.
        """
        if self.weak_form is not None and self.weak_form != []:
            op = operator
        else:
            op = torch.mean(operator ** 2, 0)

        loss_operator = op @ lambda_op.T
        return loss_operator, op

    def _loss_bcs(self,
                  bval: torch.Tensor,
                  true_bval: torch.Tensor,
                  lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes boundary loss for corresponding type.

        Args:
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss_bnd (torch.Tensor): boundary term in loss.
            bval_diff (torch.Tensor): MSE of all boundary con-s.
        """

        bval_diff = torch.mean((bval - true_bval) ** 2, 0)

        loss_bnd = bval_diff @ lambda_bound.T
        return loss_bnd, bval_diff

    def _default_loss(self,
                      operator: torch.Tensor,
                      bval: torch.Tensor,
                      true_bval: torch.Tensor,
                      lambda_op: torch.Tensor,
                      lambda_bound: torch.Tensor,
                      save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute l2 loss.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
            save_graph (bool, optional): saving computational graph. Defaults to True.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return torch.sum(torch.mean((operator) ** 2, 0))

        loss_oper, op = self._loss_op(operator, lambda_op)
        dtype = op.dtype
        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
        lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        # TODO make decorator and apply it for all losses.
        if not save_graph:
            temp_loss = loss.detach()
            del loss
            torch.cuda.empty_cache()
            loss = temp_loss

        return loss, loss_normalized

    def _causal_loss(self,
                     operator: torch.Tensor,
                     bval: torch.Tensor,
                     true_bval: torch.Tensor,
                     lambda_op: torch.Tensor,
                     lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes causal loss, which is calculated with weights matrix:
        W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
        to t_i moment of time. This loss function should be used when one
        of the DE independent parameter is time.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))

        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper + \
                              lambda_bound_normalized @ bval_diff

        return loss, loss_normalized

    def _weak_loss(self,
                   operator: torch.Tensor,
                   bval: torch.Tensor,
                   true_bval: torch.Tensor,
                   lambda_op: torch.Tensor,
                   lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Weak solution of O/PDE problem.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.

        Returns:
            loss (torch.Tensor): loss.
            loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
        """

        if bval is None:
            return sum(operator)

        loss_oper, op = self._loss_op(operator, lambda_op)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
        loss = loss_oper + loss_bnd

        lambda_op_normalized = lambda_prepare(operator, 1)
        lambda_bound_normalized = lambda_prepare(bval, 1)

        with torch.no_grad():
            loss_normalized = op @ lambda_op_normalized.T + \
                              bval_diff @ lambda_bound_normalized.T

        return loss, loss_normalized

    def compute(self,
                operator: torch.Tensor,
                bval: torch.Tensor,
                true_bval: torch.Tensor,
                lambda_op: torch.Tensor,
                lambda_bound: torch.Tensor,
                save_graph: bool = True) -> Union[_default_loss, _weak_loss, _causal_loss]:
        """ Setting the required loss calculation method.

        Args:
            operator (torch.Tensor): operator calc-n result.
            For more details to eval module -> operator_compute().
            bval (torch.Tensor): calculated values of boundary conditions.
            true_bval (torch.Tensor): true values of boundary conditions.
            lambda_op (torch.Tensor): regularization parameter for operator term in loss.
            lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
            save_graph (bool, optional): saving computational graph. Defaults to True.

        Returns:
            Union[default_loss, weak_loss, causal_loss]: A given calculation method.
        """

        if self.mode in ('mat', 'autograd'):
            if bval is None:
                print('No bconds is not possible, returning infinite loss')
                return np.inf
        inputs = [operator, bval, true_bval, lambda_op, lambda_bound]

        if self.weak_form is not None and self.weak_form != []:
            return self._weak_loss(*inputs)
        elif self.tol != 0:
            return self._causal_loss(*inputs)
        else:
            return self._default_loss(*inputs, save_graph)


class Solution():
    """
    class for different loss functions calculation.
    """

    def __init__(
            self,
            grid: torch.Tensor,
            equal_cls: Union[Equation_NN, Equation_mat, Equation_autograd],
            model: Union[torch.nn.Sequential, torch.Tensor],
            mode: str,
            weak_form: Union[None, list[callable]],
            lambda_operator,
            lambda_bound,
            tol: float = 0,
            derivative_points: int = 2):
        """
        Args:
            grid (torch.Tensor): discretization of comp-l domain.
            equal_cls (Union[Equation_NN, Equation_mat, Equation_autograd]): Equation_{NN, mat, autograd} object.
            model (Union[torch.nn.Sequential, torch.Tensor]): model of *mat or NN or autograd* mode.
            mode (str): *mat or NN or autograd*
            weak_form (Union[None, list[callable]]): list with basis functions, if the form is *weak*.
            lambda_operator (_type_): regularization parameter for operator term in loss.
            lambda_bound (_type_): regularization parameter for boundary term in loss.
            tol (float, optional): penalty in *casual loss*. Defaults to 0.
            derivative_points (int, optional): points number for derivative calculation.
            For details to Derivative_mat class.. Defaults to 2.
        """

        self.grid = check_device(grid)
        if mode == 'NN':
            sorted_grid = Points_type(self.grid).grid_sort()
            self.n_t = len(sorted_grid['central'][:, 0].unique())
        elif mode == 'autograd':
            self.n_t = len(self.grid[:, 0].unique())
        elif mode == 'mat':
            self.n_t = grid.shape[1]
        equal_copy = deepcopy(equal_cls)
        prepared_operator = equal_copy.operator_prepare()
        self._operator_coeff(equal_cls, prepared_operator)
        self.prepared_bconds = equal_copy.bnd_prepare()
        self.model = model.to(device_type())
        self.mode = mode
        self.weak_form = weak_form
        self.lambda_operator = lambda_operator
        self.lambda_bound = lambda_bound
        self.tol = tol
        self.derivative_points = derivative_points

        self.operator = Operator(self.grid, prepared_operator, self.model,
                                 self.mode, weak_form, derivative_points)
        self.boundary = Bounds(self.grid, self.prepared_bconds, self.model,
                               self.mode, weak_form, derivative_points)

        self.loss_cls = Losses(self.mode, self.weak_form, self.n_t, self.tol)
        self.op_list = []
        self.bval_list = []
        self.loss_list = []

    @staticmethod
    def _operator_coeff(equal_cls: Any, operator: list):
        """ Coefficient checking in operator.

        Args:
            equal_cls (Any): Equation_{NN, mat, autograd} object.
            operator (list): prepared operator (result of operator_prepare())
        """
        for i, _ in enumerate(equal_cls.operator):
            eq = equal_cls.operator[i]
            for key in eq.keys():
                if isinstance(eq[key]['coeff'], torch.nn.Parameter):
                    try:
                        operator[i][key]['coeff'] = eq[key]['coeff'].to(device_type())
                    except:
                        operator[key]['coeff'] = eq[key]['coeff'].to(device_type())
                elif isinstance(eq[key]['coeff'], torch.Tensor):
                    eq[key]['coeff'] = eq[key]['coeff'].to(device_type())

    def _model_change(self, new_model: torch.nn.Module) -> None:
        """Change self.model for class and *operator, boundary* object.
            It should be used in cache_lookup and cache_retrain method.

        Args:
            new_model (torch.nn.Module): new self model.
        """
        self.model = new_model
        self.operator.model = new_model
        self.operator.derivative = Derivative(new_model, self.derivative_points).set_strategy(
            self.mode).take_derivative
        self.boundary.model = new_model
        self.boundary.operator = Operator(self.grid,
                                          self.prepared_bconds,
                                          new_model,
                                          self.mode,
                                          self.weak_form,
                                          self.derivative_points)

    def evaluate(self,
                 save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes loss.

        Args:
            second_order_interactions (bool, optional): optimizer iteration
            (serves only for computing adaptive lambdas). Defaults to True.
            sampling_N (int, optional): parameter for accumulation of
            solutions (op, bcs). The more sampling_N, the more accurate the
            estimation of the variance (only for computing adaptive lambdas). Defaults to 1.
            lambda_update (bool, optional): update lambda or not. Defaults to False.
            save_graph (bool, optional): responsible for saving the computational graph. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: loss
        """

        self.op = self.operator.operator_compute()
        self.bval, self.true_bval, \
        self.bval_keys, self.bval_length = self.boundary.apply_bcs()
        dtype = self.op.dtype
        self.lambda_operator = lambda_prepare(self.op, self.lambda_operator).to(dtype)
        self.lambda_bound = lambda_prepare(self.bval, self.lambda_bound).to(dtype)

        self.loss, self.loss_normalized = self.loss_cls.compute(
            self.op,
            self.bval,
            self.true_bval,
            self.lambda_operator,
            self.lambda_bound,
            save_graph)

        return self.loss, self.loss_normalized


class PSO(torch.optim.Optimizer):
    """Custom PSO optimizer.
    """

    def __init__(self,
                 params,
                 pop_size: int = 30,
                 b: float = 0.9,
                 c1: float = 8e-2,
                 c2: float = 5e-1,
                 lr: float = 1e-3,
                 betas: Tuple = (0.99, 0.999),
                 c_decrease: bool = False,
                 variance: float = 1,
                 epsilon: float = 1e-8,
                 n_iter: int = 2000):
        """The Particle Swarm Optimizer class.

        Args:
            pop_size (int, optional): Population of the PSO swarm. Defaults to 30.
            b (float, optional): Inertia of the particles. Defaults to 0.99.
            c1 (float, optional): The *p-best* coeficient. Defaults to 0.08.
            c2 (float, optional): The *g-best* coeficient. Defaults to 0.5.
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.00,
                so there will not be any gradient-based optimization.
            betas (tuple(float, float), optional): same coeff in Adam algorithm. Defaults to (0.99, 0.999).
            c_decrease (bool, optional): Flag for update_pso_params method. Defautls to False.
            variance (float, optional): Variance parameter for swarm creation
                based on model. Defaults to 1.
            epsilon (float, optional): some add to gradient descent like in Adam optimizer.
                Defaults to 1e-8.
        """
        defaults = {'pop_size': pop_size,
                    'b': b, 'c1': c1, 'c2': c2,
                    'lr': lr, 'betas': betas,
                    'c_decrease': c_decrease,
                    'variance': variance,
                    'epsilon': epsilon}
        super(PSO, self).__init__(params, defaults)
        self.params = self.param_groups[0]['params']
        self.pop_size = pop_size
        self.b = b
        self.c1 = c1
        self.c2 = c2
        self.c_decrease = c_decrease
        self.epsilon = epsilon
        self.beta1, self.beta2 = betas
        self.lr = lr * np.sqrt(1 - self.beta2) / (1 - self.beta1)
        self.use_grad = True if self.lr != 0 else False
        self.variance = variance
        self.name = "PSO"
        self.n_iter = n_iter

        vec_shape = self.params_to_vec().shape
        self.vec_shape = list(vec_shape)[0]

        self.swarm = self.build_swarm()

        self.p = copy(self.swarm).detach()

        self.v = self.start_velocities()
        self.m1 = torch.zeros(self.pop_size, self.vec_shape)
        self.m2 = torch.zeros(self.pop_size, self.vec_shape)

        self.indicator = True

    def params_to_vec(self) -> torch.Tensor:
        """ Method for converting model parameters *NN and autograd*
           or model values *mat* to vector.

        Returns:
            torch.Tensor: model parameters/model values vector.
        """
        if not isinstance(self.params, torch.Tensor):
            vec = parameters_to_vector(self.params)
        else:
            self.model_shape = self.params.shape
            vec = self.params.reshape(-1)

        return vec

    def vec_to_params(self, vec: torch.Tensor) -> None:
        """Method for converting vector to model parameters (NN, autograd)
           or model values (mat)

        Args:
            vec (torch.Tensor): The particle of swarm.
        """
        if not isinstance(self.params, torch.Tensor):
            vector_to_parameters(vec, self.params)
        else:
            self.params.data = vec.reshape(self.params).data

    def build_swarm(self):
        """Creates the swarm based on solution class model.

        Returns:
            torch.Tensor: The PSO swarm population.
            Each particle represents a neural network (NN, autograd) or model values (mat).
        """
        vector = self.params_to_vec()
        matrix = []
        for _ in range(self.pop_size):
            matrix.append(vector.reshape(1, -1))
        matrix = torch.cat(matrix)
        variance = torch.FloatTensor(self.pop_size, self.vec_shape).uniform_(
            -self.variance, self.variance).to(device_type())
        swarm = (matrix + variance).clone().detach().requires_grad_(True)
        return swarm

    def update_pso_params(self) -> None:
        """Method for updating pso parameters if c_decrease=True.
        """
        self.c1 -= 2 * self.c1 / self.n_iter
        self.c2 += self.c2 / self.n_iter

    def start_velocities(self) -> torch.Tensor:
        """Start the velocities of each particle in the population (swarm) as `0`.

        Returns:
            torch.Tensor: The starting velocities.
        """
        return torch.zeros((self.pop_size, self.vec_shape))

    def gradient(self, loss: torch.Tensor) -> torch.Tensor:
        """ Calculation of loss gradient by model parameters (NN, autograd)
            or model values (mat).

        Args:
            loss (torch.Tensor): result of loss calculation.

        Returns:
            torch.Tensor: calculated gradient vector.
        """
        dl_dparam = torch.autograd.grad(loss, self.params)

        grads = parameters_to_vector(dl_dparam)

        return grads

    def get_randoms(self) -> torch.Tensor:
        """Generate random values to update the particles' positions.

        Returns:
            torch.Tensor: random tensor
        """
        return torch.rand((2, 1, self.vec_shape))

    def update_p_best(self) -> None:
        """Updates the *p-best* positions."""

        idx = torch.where(self.loss_swarm < self.f_p)

        self.p[idx] = self.swarm[idx]
        self.f_p[idx] = self.loss_swarm[idx].detach()

    def update_g_best(self) -> None:
        """Update the *g-best* position."""
        self.g_best = self.p[torch.argmin(self.f_p)]

    def gradient_descent(self) -> torch.Tensor:
        """ Gradiend descent based on Adam algorithm.

        Returns:
            torch.Tensor: gradient term in velocities vector.
        """
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * self.grads_swarm
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * torch.square(
            self.grads_swarm)
        return self.lr * self.m1 / torch.sqrt(self.m2) + self.epsilon

    def step(self, closure=None) -> torch.Tensor:
        """ It runs ONE step on the particle swarm optimization.

        Returns:
            torch.Tensor: loss value for best particle of thw swarm.
        """

        self.loss_swarm, self.grads_swarm = closure()
        if self.indicator:
            self.f_p = copy(self.loss_swarm).detach()
            self.g_best = self.p[torch.argmin(self.f_p)]
            self.indicator = False

        r1, r2 = self.get_randoms()

        self.v = self.b * self.v + (1 - self.b) * (
                self.c1 * r1 * (self.p - self.swarm) + self.c2 * r2 * (self.g_best - self.swarm))
        if self.use_grad:
            self.swarm = self.swarm + self.v - self.gradient_descent()
        else:
            self.swarm = self.swarm + self.v
        self.update_p_best()
        self.update_g_best()
        self.vec_to_params(self.g_best)
        if self.c_decrease:
            self.update_pso_params()
        min_loss = torch.min(self.f_p)

        return min_loss


class Optimizer():
    def __init__(
            self,
            optimizer: str,
            params: dict,
            gamma: Union[float, None] = None,
            decay_every: Union[int, None] = None):
        self.optimizer = optimizer
        self.params = params
        self.gamma = gamma
        self.decay_every = decay_every

    def optimizer_choice(
            self,
            mode,
            model) -> \
            Union[torch.optim.Adam, torch.optim.SGD, torch.optim.LBFGS, PSO]:
        """ Setting optimizer. If optimizer is string type, it will get default settings,
            or it may be custom optimizer defined by user.

        Args:
           optimizer: optimizer choice (Adam, SGD, LBFGS, PSO).
           learning_rate: determines the step size at each iteration
           while moving toward a minimum of a loss function.

        Returns:
            optimzer: ready optimizer.
        """

        if self.optimizer == 'Adam':
            torch_optim = torch.optim.Adam
        elif self.optimizer == 'SGD':
            torch_optim = torch.optim.SGD
        elif self.optimizer == 'LBFGS':
            torch_optim = torch.optim.LBFGS
        elif self.optimizer == 'PSO':
            torch_optim = PSO

        if mode in ('NN', 'autograd'):
            optimizer = torch_optim(model.parameters(), **self.params)
        elif mode == 'mat':
            optimizer = torch_optim([model.requires_grad_()], **self.params)

        if self.gamma is not None:
            self.scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return optimizer


class Closure():
    def __init__(self,
                 mixed_precision: bool,
                 model):
        self.mixed_precision = mixed_precision
        self.set_model(model)
        self.optimizer = self.model.optimizer
        self.normalized_loss_stop = self.model.normalized_loss_stop
        self.device = device_type()
        self.cuda_flag = True if self.device == 'cuda' and self.mixed_precision else False
        self.dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16
        if self.mixed_precision:
            self._amp_mixed()

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def _amp_mixed(self):
        """ Preparation for mixed precsion operations.

        Args:
            mixed_precision (bool): use or not torch.amp.

        Raises:
            NotImplementedError: AMP and the LBFGS optimizer are not compatible.

        Returns:
            scaler: GradScaler for CUDA.
            cuda_flag (bool): True, if CUDA is activated and mixed_precision=True.
            dtype (dtype): operations dtype.
        """

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        if self.mixed_precision:
            print(f'Mixed precision enabled. The device is {self.device}')
        if self.optimizer.__class__.__name__ == "LBFGS":
            raise NotImplementedError("AMP and the LBFGS optimizer are not compatible.")

    def _closure(self):
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device,
                            dtype=self.dtype,
                            enabled=self.mixed_precision):
            loss, loss_normalized = self.model.solution_cls.evaluate()
        if self.cuda_flag:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()

        self.model.cur_loss = loss_normalized if self.normalized_loss_stop else loss

        return loss

    def _closure_pso(self):
        def loss_grads():
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device,
                                dtype=self.dtype,
                                enabled=self.mixed_precision):
                loss, loss_normalized = self.model.solution_cls.evaluate()

            if self.optimizer.use_grad:
                grads = self.optimizer.gradient(loss)
                grads = torch.where(grads == float('nan'), torch.zeros_like(grads), grads)
            else:
                grads = torch.tensor([0.])

            return loss, grads

        loss_swarm = []
        grads_swarm = []
        for particle in self.optimizer.swarm:
            self.optimizer.vec_to_params(particle)
            loss_particle, grads = loss_grads()
            loss_swarm.append(loss_particle)
            grads_swarm.append(grads.reshape(1, -1))

        losses = torch.stack(loss_swarm).reshape(-1)
        gradients = torch.vstack(grads_swarm)

        self.model.cur_loss = min(loss_swarm)

        return losses, gradients

    def get_closure(self, _type: str):
        if _type == 'PSO':
            return self._closure_pso
        else:
            return self._closure


# from tedeous.model import Model
class Model():
    """class for preprocessing"""

    def __init__(
            self,
            net: Union[torch.nn.Module, torch.Tensor],
            domain: Domain,
            equation: Equation,
            conditions: Conditions):
        """
        Args:
            net (Union[torch.nn.Module, torch.Tensor]): neural network or torch.Tensor for mode *mat*
            grid (Domain): object of class Domain
            equation (Equation): object of class Equation
            conditions (Conditions): object of class Conditions
        """
        self.net = net
        self.domain = domain
        self.equation = equation
        self.conditions = conditions
        self._check = None
        temp_dir = tempfile.gettempdir()
        folder_path = os.path.join(temp_dir, 'tedeous_cache/')
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            pass
        else:
            os.makedirs(folder_path)
        self._save_dir = folder_path

    def compile(
            self,
            mode: str,
            lambda_operator: Union[List[float], float],
            lambda_bound: Union[List[float], float],
            normalized_loss_stop: bool = False,
            h: float = 0.001,
            inner_order: str = '1',
            boundary_order: str = '2',
            derivative_points: int = 2,
            weak_form: List[callable] = None,
            tol: float = 0):
        """ Compile model for training process.

        Args:
            mode (str): *mat, NN, autograd*
            lambda_operator (Union[List[float], float]): weight for operator term.
            It can be float for single equation or list of float for system.
            lambda_bound (Union[List[float], float]): weight for boundary term.
            It can be float for all types of boundary cond-ns or list of float for every condition type.
            normalized_loss_stop (bool, optional): loss with lambdas=1. Defaults to False.
            h (float, optional): increment for finite-difference scheme only for *NN*. Defaults to 0.001.
            inner_order (str, optional): order of finite-difference scheme *'1', '2'* for inner points.
            Only for *NN*. Defaults to '1'.
            boundary_order (str, optional): order of finite-difference scheme *'1', '2'* for boundary points.
            Only for *NN*. Defaults to '2'.
            derivative_points (int, optional): number of points for finite-difference scheme in *mat* mode.
            if derivative_points=2 the central scheme are used. Defaults to 2.
            weak_form (List[callable], optional): basis function for weak loss. Defaults to None.
            tol (float, optional): tolerance for causual loss. Defaults to 0.
        """
        self.mode = mode
        self.lambda_bound = lambda_bound
        self.lambda_operator = lambda_operator
        self.normalized_loss_stop = normalized_loss_stop
        self.weak_form = weak_form

        grid = self.domain.build(mode=mode)
        dtype = grid.dtype
        self.net.to(dtype)
        variable_dict = self.domain.variable_dict
        operator = self.equation.equation_lst
        bconds = self.conditions.build(variable_dict)

        self.equation_cls = Operator_bcond_preproc(grid, operator, bconds, h=h, inner_order=inner_order,
                                                   boundary_order=boundary_order).set_strategy(mode)

        self.solution_cls = Solution(grid, self.equation_cls, self.net, mode, weak_form,
                                     lambda_operator, lambda_bound, tol, derivative_points)

    def _model_save(
            self,
            save_model: bool,
            model_name: str):
        """ Model saving.

        Args:
            save_model (bool): save model or not.
            model_name (str): model name.
        """
        if save_model:
            if self.mode == 'mat':
                save_model_mat(self._save_dir,
                               model=self.net,
                               domain=self.domain,
                               name=model_name)
            else:
                save_model_nn(self._save_dir, model=self.net, name=model_name)

    def train(self,
              optimizer: Optimizer,
              epochs: int,
              info_string_every: Union[int, None] = None,
              mixed_precision: bool = False,
              save_model: bool = False,
              model_name: Union[str, None] = None,
              callbacks: Union[List, None] = None):
        """ train model.

        Args:
            optimizer (Optimizer): the object of Optimizer class
            epochs (int): number of epoch for training.
            info_string_every (Union[int, None], optional): print loss state after *info_string_every* epoch. Defaults to None.
            mixed_precision (bool, optional): apply mixed precision for calculation. Defaults to False.
            save_model (bool, optional): save resulting model in cache. Defaults to False.
            model_name (Union[str, None], optional): model name. Defaults to None.
            callbacks (Union[List, None], optional): callbacks for training process. Defaults to None.
        """

        self.t = 1
        self.stop_training = False

        callbacks = CallbackList(callbacks=callbacks, model=self)

        callbacks.on_train_begin()

        self.net = self.solution_cls.model

        self.optimizer = optimizer.optimizer_choice(self.mode, self.net)

        closure = Closure(mixed_precision, self).get_closure(optimizer.optimizer)

        self.min_loss, _ = self.solution_cls.evaluate()

        self.cur_loss = self.min_loss

        print('[{}] initial (min) loss is {}'.format(
            datetime.datetime.now(), self.min_loss.item()))

        while self.t < epochs and self.stop_training is False:
            callbacks.on_epoch_begin()

            self.optimizer.zero_grad()

            if device_type() == 'cuda' and mixed_precision:
                closure()
            else:
                self.optimizer.step(closure)
            if optimizer.gamma is not None and self.t % optimizer.decay_every == 0:
                optimizer.scheduler.step()

            callbacks.on_epoch_end()

            self.t += 1
            if info_string_every is not None:
                if self.t % info_string_every == 0:
                    loss = self.cur_loss.item() if isinstance(self.cur_loss, torch.Tensor) else self.cur_loss
                    info = 'Step = {} loss = {:.6f}.'.format(self.t, loss)
                    print(info)

        callbacks.on_train_end()

        self._model_save(save_model, model_name)


# from tedeous.callbacks import early_stopping, plot, cache
class EarlyStopping(Callback):
    """ Class for using adaptive stop criterias at training process.
    """

    def __init__(self,
                 eps: float = 1e-5,
                 loss_window: int = 100,
                 no_improvement_patience: int = 1000,
                 patience: int = 5,
                 abs_loss: Union[float, None] = None,
                 normalized_loss: bool = False,
                 randomize_parameter: float = 1e-5,
                 info_string_every: Union[int, None] = None,
                 verbose: bool = True
                 ):
        """_summary_

        Args:
            eps (float, optional): arbitrarily small number that uses for loss comparison criterion. Defaults to 1e-5.
            loss_window (int, optional): width of losses window which is used for average loss estimation. Defaults to 100.
            no_improvement_patience (int, optional):  number of iterations during which
                    the loss may not improve. Defaults to 1000.
            patience (int, optional): maximum number of times the stopping criterion
                                      can be satisfied.. Defaults to 5.
            abs_loss (Union[float, None], optional): absolute loss value using in _absloss_check().. Defaults to None.
            normalized_loss (bool, optional): calculate loss with all lambdas=1. Defaults to False.
            randomize_parameter (float, optional): some error for resulting
                                        model weights to to avoid local optima. Defaults to 1e-5.
            info_string_every (Union[int, None], optional): prints the loss state after every *int*
                                                    step. Defaults to None.
            verbose (bool, optional): print or not info about loss and current state of stopping criteria. Defaults to True.
        """
        super().__init__()
        self.eps = eps
        self.loss_window = loss_window
        self.no_improvement_patience = no_improvement_patience
        self.patience = patience
        self.abs_loss = abs_loss
        self.normalized_loss = normalized_loss
        self._stop_dings = 0
        self._t_imp_start = 0
        self._r = create_random_fn(randomize_parameter)
        self.info_string_every = info_string_every if info_string_every is not None else np.inf
        self.verbose = verbose

    def _line_create(self):
        """ Approximating last_loss list (len(last_loss)=loss_oscillation_window) by the line.

        """
        self._line = np.polyfit(range(self.loss_window), self.last_loss, 1)

    def _window_check(self):
        """ Stopping criteria. We divide angle coeff of the approximating
        line (line_create()) on current loss value and compare one with *eps*
        """
        if self.t % self.loss_window == 0 and self._check is None:
            self._line_create()
            if abs(self._line[0] / self.model.cur_loss) < self.eps and self.t > 0:
                self._stop_dings += 1
                if self.mode in ('NN', 'autograd'):
                    self.model.net.apply(self._r)
                self._check = 'window_check'

    def _patience_check(self):
        """ Stopping criteria. We control the minimum loss and count steps
        when the current loss is bigger then min_loss. If these steps equal to
        no_improvement_patience parameter, the stopping criteria will be achieved.

        """
        if (self.t - self._t_imp_start) == self.no_improvement_patience and self._check is None:
            self._stop_dings += 1
            self._t_imp_start = self.t
            if self.mode in ('NN', 'autograd'):
                self.model.net.apply(self._r)
            self._check = 'patience_check'

    def _absloss_check(self):
        """ Stopping criteria. If current loss absolute value is lower then *abs_loss* param,
        the stopping criteria will be achieved.
        """
        if self.abs_loss is not None and self.model.cur_loss < self.abs_loss and self._check is None:
            self._stop_dings += 1
            self._check = 'absloss_check'

    def verbose_print(self):
        """ print info about loss and stopping criteria.
        """

        if self._check == 'window_check':
            print('[{}] Oscillation near the same loss'.format(
                datetime.datetime.now()))
        elif self._check == 'patience_check':
            print('[{}] No improvement in {} steps'.format(
                datetime.datetime.now(), self.no_improvement_patience))
        elif self._check == 'absloss_check':
            print('[{}] Absolute value of loss is lower than threshold'.format(
                datetime.datetime.now()))

        if self._check is not None or self.t % self.info_string_every == 0:
            try:
                self._line
            except:
                self._line_create()
            loss = self.model.cur_loss.item() if isinstance(self.model.cur_loss, torch.Tensor) else self.mdoel.cur_loss
            info = '[{}] Step = {} loss = {:.6f} normalized loss line= {:.6f}x+{:.6f}. There was {} stop dings already.'.format(
                datetime.datetime.now(), self.t, loss, self._line[0] / loss, self._line[1] / loss, self._stop_dings)
            print(info)

    def on_epoch_end(self, logs=None):
        self._window_check()
        self._patience_check()
        self._absloss_check()

        if self.model.cur_loss < self.model.min_loss:
            self.model.min_loss = self.model.cur_loss
            self._t_imp_start = self.t

        if self.verbose:
            self.verbose_print()
        if self._stop_dings >= self.patience:
            self.model.stop_training = True
        self._check = None

    def on_epoch_begin(self, logs=None):
        self.t = self.model.t
        self.mode = self.model.mode
        self._check = self.model._check
        try:
            self.last_loss[(self.t - 3) % self.loss_window] = self.model.cur_loss
        except:
            self.last_loss = np.zeros(self.loss_window) + float(self.model.min_loss)


def count_output(model: torch.Tensor) -> int:
    """ Determine the out features of the model.

    Args:
        model (torch.Tensor): torch neural network.

    Returns:
        int: number of out features.
    """
    modules, output_layer = list(model.modules()), None
    for layer in reversed(modules):
        if hasattr(layer, 'out_features'):
            output_layer = layer.out_features
            break
    return output_layer


class CachePreprocessing:
    """class for preprocessing cache files.
    """

    def __init__(self,
                 model: Model
                 ):
        """
        Args:
            model (Model): object of Model class
        """
        self.solution_cls = model.solution_cls

    @staticmethod
    def _cache_files(files: list, nmodels: Union[int, None] = None) -> np.ndarray:
        """ At some point we may want to reduce the number of models that are
            checked for the best in the cache.

        Args:
            files (list): list with all model names in cache.
            nmodels (Union[int, None], optional): models quantity for checking. Defaults to None.

        Returns:
            cache_n (np.ndarray): array with random cache files names.
        """

        if nmodels is None:
            # here we take all files that are in cache
            cache_n = np.arange(len(files))
        else:
            # here we take random nmodels from the cache
            cache_n = np.random.choice(len(files), nmodels, replace=False)

        return cache_n

    @staticmethod
    def _model_reform(init_model: Union[torch.nn.Sequential, torch.nn.ModuleList],
                      model: Union[torch.nn.Sequential, torch.nn.ModuleList]):
        """
        As some models are nn.Sequential class objects,
        but another models are nn.Module class objects.
        This method does checking the solver model (init_model)
        and the cache model (model).
        Args:
            init_model (nn.Sequential or nn.ModuleList): solver model.
            model (nn.Sequential or nn.ModuleList): cache model.
        Returns:
            init_model (nn.Sequential or nn.ModuleList): checked init_model.
            model (nn.Sequential or nn.ModuleList): checked model.
        """
        try:
            model[0]
        except:
            model = model.model

        try:
            init_model[0]
        except:
            init_model = init_model.model

        return init_model, model

    def cache_lookup(self,
                     cache_dir: str,
                     nmodels: Union[int, None] = None,
                     save_graph: bool = False,
                     cache_verbose: bool = False) -> Union[None, dict, torch.nn.Module]:
        """Looking for the best model (min loss) model from the cache files.

        Args:
            cache_dir (str): folder where system looks for cached models.
            nmodels (Union[int, None], optional): maximal number of models that are taken from cache dir. Defaults to None.
            save_graph (bool, optional): responsible for saving the computational graph. Defaults to False.
            cache_verbose (bool, optional): verbose cache operations. Defaults to False.

        Returns:
            Union[None, dict, torch.Tensor]: best model with optimizator state.
        """

        files = glob.glob(cache_dir + '\*.tar')

        if cache_verbose:
            print(f"The CACHE will be searched among the models in the folder {cache_dir}.")

        if len(files) == 0:
            best_checkpoint = None
            return best_checkpoint

        cache_n = self._cache_files(files, nmodels)

        min_loss = np.inf
        best_checkpoint = {}

        device = device_type()

        initial_model = self.solution_cls.model

        for i in cache_n:
            file = files[i]
            checkpoint = torch.load(file)

            model = checkpoint['model']
            model.load_state_dict(checkpoint['model_state_dict'])

            # this one for the input shape fix if needed

            solver_model, cache_model = self._model_reform(self.solution_cls.model, model)

            if cache_model[0].in_features != solver_model[0].in_features:
                continue
            try:
                if count_output(model) != count_output(self.solution_cls.model):
                    continue
            except Exception:
                continue

            model = model.to(device)
            self.solution_cls._model_change(model)
            loss, _ = self.solution_cls.evaluate(save_graph=save_graph)

            if loss < min_loss:
                min_loss = loss
                best_checkpoint['model'] = model
                best_checkpoint['model_state_dict'] = model.state_dict()
                if cache_verbose:
                    print('best_model_num={} , loss={}'.format(i, min_loss.item()))

            self.solution_cls._model_change(initial_model)

        if best_checkpoint == {}:
            best_checkpoint = None

        return best_checkpoint

    def scheme_interp(self,
                      trained_model: torch.nn.Module,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ If the cache model has another arcitechure to user's model,
            we will not be able to use it. So we train user's model on the
            outputs of cache model.

        Args:
            trained_model (torch.nn.Module): the best model (min loss) from cache.
            cache_verbose (bool, optional): verbose on/off of cache operations. Defaults to False.

        """

        grid = self.solution_cls.grid

        model = self.solution_cls.model

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss = torch.mean(torch.square(
            trained_model(grid) - model(grid)))

        def closure():
            optimizer.zero_grad()
            loss = torch.mean((trained_model(grid) - model(grid)) ** 2)
            loss.backward()
            return loss

        t = 0
        while loss > 1e-5 and t < 1e5:
            optimizer.step(closure)
            loss = torch.mean(torch.square(
                trained_model(grid) - model(grid)))
            t += 1
            if cache_verbose:
                print('Interpolate from trained model t={}, loss={}'.format(
                    t, loss))

        self.solution_cls._model_change(model)

    def cache_retrain(self,
                      cache_checkpoint: dict,
                      cache_verbose: bool = False) -> torch.nn.Module:
        """ The comparison of the user's model and cache model architecture.
            If they are same, we will use model from cache. In the other case
            we use interpolation (scheme_interp method)

        Args:
            cache_checkpoint (dict): checkpoint of the cache model
            cache_verbose (bool, optional): on/off printing cache operations. Defaults to False.

        """

        model = self.solution_cls.model

        # do nothing if cache is empty
        if cache_checkpoint is None:
            return None
        # if models have the same structure use the cache model state,
        # and the cache model has ordinary structure
        if str(cache_checkpoint['model']) == str(model) and \
                isinstance(model, torch.nn.Sequential) and \
                isinstance(model[0], torch.nn.Linear):
            model = cache_checkpoint['model']
            model.load_state_dict(cache_checkpoint['model_state_dict'])
            model.train()
            self.solution_cls._model_change(model)
            if cache_verbose:
                print('Using model from cache')
        # else retrain the input model using the cache model
        else:
            cache_model = cache_checkpoint['model']
            cache_model.load_state_dict(cache_checkpoint['model_state_dict'])
            cache_model.eval()
            self.scheme_interp(
                cache_model, cache_verbose=cache_verbose)


class Cache(Callback):
    """
    Prepares user's model. Serves for computing acceleration.\n
    Saves the trained model to the cache, and subsequently it is possible to use pre-trained model
    (if it saved and if the new model is structurally similar) to sped up computing.\n
    If there isn't pre-trained model in cache, the training process will start from the beginning.
    """

    def __init__(self,
                 nmodels: Union[int, None] = None,
                 cache_dir: str = 'tedeous_cache',
                 cache_verbose: bool = False,
                 cache_model: Union[torch.nn.Sequential, None] = None,
                 model_randomize_parameter: Union[int, float] = 0,
                 clear_cache: bool = False
                 ):
        """
        Args:
            nmodels (Union[int, None], optional): maximal number of models that are taken from cache dir. Defaults to None. Defaults to None.
            cache_dir (str, optional): directory with cached models. Defaults to '../tedeous_cache/' in temporary directoy of user system.
                If cache_dir is custom, then file will be searched in *torch_de_solver* directory.
            cache_verbose (bool, optional): printing cache operations. Defaults to False.
            cache_model (Union[torch.nn.Sequential, None], optional): model for mat method, which will be saved in cache. Defaults to None.
            model_randomize_parameter (Union[int, float], optional): creates a random model parameters (weights, biases)
                multiplied with a given randomize parameter.. Defaults to 0.
            clear_cache (bool, optional): clear cache directory. Defaults to False.
        """

        self.nmodels = nmodels
        self.cache_verbose = cache_verbose
        self.cache_model = cache_model
        self.model_randomize_parameter = model_randomize_parameter
        if cache_dir == 'tedeous_cache':
            temp_dir = tempfile.gettempdir()
            folder_path = os.path.join(temp_dir, 'tedeous_cache/')
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                pass
            else:
                os.makedirs(folder_path)
            self.cache_dir = folder_path
        else:
            try:
                file = __file__
            except:
                file = os.getcwd()
            self.cache_dir = os.path.normpath((os.path.join(os.path.dirname(file), '..', '..', cache_dir)))
        if clear_cache:
            remove_all_files(self.cache_dir)

    def _cache_nn(self):
        """  take model from cache as initial guess for *NN, autograd* modes.
        """
        cache_preproc = CachePreprocessing(self.model)
        r = create_random_fn(self.model_randomize_parameter)
        cache_checkpoint = cache_preproc.cache_lookup(cache_dir=self.cache_dir,
                                                      nmodels=self.nmodels,
                                                      cache_verbose=self.cache_verbose)

        cache_preproc.cache_retrain(cache_checkpoint, cache_verbose=self.cache_verbose)
        self.model.solution_cls.model.apply(r)

    def _cache_mat(self) -> torch.Tensor:
        """  take model from cache as initial guess for *mat* mode.
        """

        net = self.model.net
        domain = self.model.domain
        equation = mat_op_coeff(deepcopy(self.model.equation))
        conditions = self.model.conditions
        lambda_operator = self.model.lambda_operator
        lambda_bound = self.model.lambda_bound
        weak_form = self.model.weak_form

        net_autograd = model_mat(net, domain)

        autograd_model = Model(net_autograd, domain, equation, conditions)

        autograd_model.compile('autograd', lambda_operator, lambda_bound, weak_form=weak_form)

        r = create_random_fn(self.model_randomize_parameter)

        cache_preproc = CachePreprocessing(autograd_model)

        cache_checkpoint = cache_preproc.cache_lookup(
            cache_dir=self.cache_dir,
            nmodels=self.nmodels,
            cache_verbose=self.cache_verbose)

        if cache_checkpoint is not None:
            cache_preproc.cache_retrain(
                cache_checkpoint,
                cache_verbose=self.cache_verbose)

            autograd_model.solution_cls.model.apply(r)

            model = autograd_model.solution_cls.model(
                autograd_model.solution_cls.grid).reshape(
                self.model.solution_cls.model.shape).detach()

            self.model.solution_cls._model_change(model.requires_grad_())

    def cache(self):
        """ Wrap for cache_mat and cache_nn methods.
        """

        if self.model.mode != 'mat':
            return self._cache_nn()
        elif self.model.mode == 'mat':
            return self._cache_mat()

    def on_train_begin(self, logs=None):
        self.cache()
        self.model._save_dir = self.cache_dir


class Plots(Callback):
    """Class for ploting solutions."""

    def __init__(self,
                 print_every: Union[int, None] = 500,
                 save_every: Union[int, None] = 500,
                 title: str = None,
                 img_dir: str = None):
        """
        Args:
            print_every (Union[int, None], optional): print plots after every *print_every* steps. Defaults to 500.
            save_every (Union[int, None], optional): save plots after every *print_every* steps. Defaults to 500.
            title (str, optional): plots title. Defaults to None.
            img_dir (str, optional): directory title where plots are being saved. Defaults to None.
        """
        super().__init__()
        self.print_every = print_every if print_every is not None else 0.1
        self.save_every = save_every if save_every is not None else 0.1
        self.title = title
        self.img_dir = img_dir

    def _print_nn(self):
        """
        Solution plot for *NN, autograd* mode.

        """

        try:
            nvars_model = self.net[-1].out_features
        except:
            nvars_model = self.net.model[-1].out_features

        nparams = self.grid.shape[1]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net(self.grid)[:, i].detach().cpu().numpy())

            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))

                ax1.plot_trisurf(self.grid[:, 0].detach().cpu().numpy(),
                                 self.grid[:, 1].detach().cpu().numpy(),
                                 self.net(self.grid)[:, i].detach().cpu().numpy(),
                                 cmap=cm.jet, linewidth=0.2, alpha=1)
                ax1.set_xlabel("x1")
                ax1.set_ylabel("x2")

    def _print_mat(self):
        """
        Solution plot for mat mode.
        """

        nparams = self.grid.shape[0]
        nvars_model = self.net.shape[0]
        fig = plt.figure(figsize=(15, 8))
        for i in range(nvars_model):
            if nparams == 1:
                ax1 = fig.add_subplot(1, nvars_model, i + 1)
                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.scatter(self.grid.detach().cpu().numpy().reshape(-1),
                            self.net[i].detach().cpu().numpy().reshape(-1))
            else:
                ax1 = fig.add_subplot(1, nvars_model, i + 1, projection='3d')

                if self.title is not None:
                    ax1.set_title(self.title + ' variable {}'.format(i))
                ax1.plot_trisurf(self.grid[0].detach().cpu().numpy().reshape(-1),
                                 self.grid[1].detach().cpu().numpy().reshape(-1),
                                 self.net[i].detach().cpu().numpy().reshape(-1),
                                 cmap=cm.jet, linewidth=0.2, alpha=1)
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")

    def _dir_path(self, save_dir: str) -> str:
        """ Path for save figures.

        Args:
            save_dir (str): directory where saves in

        Returns:
            str: directory where saves in
        """

        if save_dir is None:
            try:
                img_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'img')
            except:
                current_dir = globals()['_dh'][0]
                img_dir = os.path.join(os.path.dirname(current_dir), 'img')

            if not os.path.isdir(img_dir):
                os.mkdir(img_dir)
            directory = os.path.abspath(os.path.join(img_dir,
                                                     str(datetime.datetime.now().timestamp()) + '.png'))
        else:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            directory = os.path.join(save_dir,
                                     str(datetime.datetime.now().timestamp()) + '.png')
        return directory

    def solution_print(
            self):
        """ printing or saving figures.
        """
        print_flag = self.model.t % self.print_every == 0
        save_flag = self.model.t % self.save_every == 0

        if print_flag or save_flag:
            self.net = self.model.net
            self.grid = self.model.solution_cls.grid
            if self.model.mode == 'mat':
                self._print_mat()
            else:
                self._print_nn()
            if save_flag:
                directory = self._dir_path(self.img_dir)
                plt.savefig(directory)
            if print_flag:
                plt.show()
            plt.close()

    def on_epoch_end(self, logs=None):
        self.solution_print()


from scipy import integrate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tedeous.data import Domain, Conditions, Equation
from tedeous.model import Model
from tedeous.optimizers.optimizer import Optimizer
from tedeous.device import solver_device, check_device, device_type
from tedeous.models import FourierNN, KAN

"""
Preparing grid

Grid is an essentially torch.Tensor  of a n-D points where n is the problem
dimensionality
"""

solver_device('сpu')

alpha = 20.
beta = 20.
delta = 20.
gamma = 20.
x0 = 4.
y0 = 2.
t0 = 0.
tmax = 1.
Nt = 300

domain = Domain()

domain.variable('t', [t0, tmax], Nt)

h = 0.0001

# initial conditions
boundaries = Conditions()
boundaries.dirichlet({'t': 0}, value=x0, var=0)
boundaries.dirichlet({'t': 0}, value=y0, var=1)

# equation system
# eq1: dx/dt = x(alpha-beta*y)
# eq2: dy/dt = y(-delta+gamma*x)

# x var: 0
# y var:1

equation = Equation()

eq1 = {
    'dx/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [0]
    },
    '-x*alpha': {
        'coeff': -alpha,
        'term': [None],
        'pow': 1,
        'var': [0]
    },
    '+beta*x*y': {
        'coeff': beta,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

eq2 = {
    'dy/dt': {
        'coeff': 1,
        'term': [0],
        'pow': 1,
        'var': [1]
    },
    '+y*delta': {
        'coeff': delta,
        'term': [None],
        'pow': 1,
        'var': [1]
    },
    '-gamma*x*y': {
        'coeff': -gamma,
        'term': [[None], [None]],
        'pow': [1, 1],
        'var': [0, 1]
    }
}

equation.add(eq1)
equation.add(eq2)

# net = torch.nn.Sequential(
#     torch.nn.Linear(1, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 100),
#     torch.nn.Tanh(),
#     torch.nn.Linear(100, 2)
# )

net = KAN()  # Default parameters: layers=[2, 100, 1]

model = Model(net, domain, equation, boundaries)

model.compile("NN", lambda_operator=1, lambda_bound=100, h=h)

img_dir = os.path.join(os.path.dirname(__file__), 'img_Lotka_Volterra')

start = time.time()

cb_cache = Cache(cache_verbose=True, model_randomize_parameter=1e-5)

cb_es = EarlyStopping(eps=1e-6,
                      loss_window=100,
                      no_improvement_patience=1000,
                      patience=5,
                      randomize_parameter=1e-5,
                      info_string_every=100)

cb_plots = Plots(save_every=1000, print_every=1000, img_dir=img_dir)

optimizer = Optimizer('Adam', {'lr': 1e-4})

model.train(optimizer, 5e6, save_model=True, callbacks=[cb_es, cb_cache, cb_plots])

end = time.time()

print('Time taken = {}'.format(end - start))


# scipy.integrate solution of Lotka_Volterra equations and comparison with NN results

def deriv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return np.array([dotx, doty])


t = np.linspace(0., tmax, Nt)

X0 = [x0, y0]
res = integrate.odeint(deriv, X0, t, args=(alpha, beta, delta, gamma))
x, y = res.T

grid = domain.build('NN')

plt.figure()
plt.grid()
plt.title("odeint and NN methods comparing")
plt.plot(t, x, '+', label='preys_odeint')
plt.plot(t, y, '*', label="predators_odeint")
plt.plot(grid, net(grid)[:, 0].detach().numpy().reshape(-1), label='preys_NN')
plt.plot(grid, net(grid)[:, 1].detach().numpy().reshape(-1), label='predators_NN')
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend(loc='upper right')
plt.show()

plt.figure()
plt.grid()
plt.title('Phase plane: prey vs predators')
plt.plot(net(grid)[:, 0].detach().numpy().reshape(-1), net(grid)[:, 1].detach().numpy().reshape(-1), '-*', label='NN')
plt.plot(x, y, label='odeint')
plt.xlabel('preys')
plt.ylabel('predators')
plt.legend()
plt.show()
