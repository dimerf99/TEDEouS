"""Module for losses calculation - DONE"""

# from typing import Tuple, Union
# import numpy as np
# import torch
#
# from tedeous.input_preprocessing import lambda_prepare


# class Losses():
#     """
#     Class which contains all losses.
#     """
#
#     def __init__(self,
#                  mode: str,
#                  weak_form: Union[None, list],
#                  n_t: int,
#                  tol: Union[int, float],
#                  method: str = 'PINN'):
#         """
#         Args:
#             mode (str): calculation mode, *NN, autograd, mat*.
#             weak_form (Union[None, list]): list of basis functions if form is weak.
#             n_t (int): number of unique points in time dinension.
#             tol (Union[int, float])): penalty in *casual loss*.
#         """
#
#         self.mode = mode
#         self.weak_form = weak_form
#         self.n_t = n_t
#         self.tol = tol
#         self.method = method
#         # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
#         # is None + fix causal_loss operator crutch (line 76).
#
#     def _loss_op(self,
#                  operator: torch.Tensor,
#                  lambda_op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Operator term in loss calc-n.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#
#         Returns:
#             loss_operator (torch.Tensor): operator term in loss.
#             op (torch.Tensor): MSE of operator on the whole grid.
#         """
#         if self.weak_form is not None and self.weak_form != []:
#             op = operator
#         else:
#             op = torch.mean(operator ** 2, 0)
#
#         loss_operator = op @ lambda_op.T
#         return loss_operator, op
#
#     # Method for PI_DeepONet - ADD NEW FULL
#     @staticmethod
#     def _loss_ics(ival: torch.Tensor,
#                   true_ival: torch.Tensor,
#                   deriv_ival: torch.Tensor,
#                   lambda_init: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """ Computes initial loss for corresponding type.
#
#         Args:
#             ival (torch.Tensor): calculated values of initial conditions.
#             true_ival (torch.Tensor): true values of initial conditions.
#             lambda_init (torch.Tensor): regularization parameter for initial term in loss.
#
#         Returns:
#             loss_init (torch.Tensor): initial term in loss.
#             ival_diff (torch.Tensor): MSE of all initial con-s.
#         """
#         ival_diff_discrepancy = torch.mean((ival - true_ival) ** 2, 0)
#         ival_diff_deriv = torch.mean(deriv_ival ** 2)
#
#         loss_init = (ival_diff_discrepancy + ival_diff_deriv) @ lambda_init.T
#
#         return loss_init, ival_diff_discrepancy, ival_diff_deriv
#
#     # This method was not @staticmethod - CHANGE ORIGINAL
#     @staticmethod
#     def _loss_bcs(bval: torch.Tensor,
#                   true_bval: torch.Tensor,
#                   lambda_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Computes boundary loss for corresponding type.
#
#         Args:
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#
#         Returns:
#             loss_bnd (torch.Tensor): boundary term in loss.
#             bval_diff (torch.Tensor): MSE of all boundary con-s.
#         """
#
#         bval_diff = torch.mean((bval - true_bval) ** 2, 0)
#         loss_bnd = bval_diff @ lambda_bound.T
#
#         return loss_bnd, bval_diff
#
#     # This method had not PI_DeepONet variant of default loss - CHANGE ORIGINAL
#     def _default_loss(self,
#                       operator: torch.Tensor,
#                       ival: torch.Tensor,
#                       bval: torch.Tensor,
#                       true_ival: torch.Tensor,
#                       true_bval: torch.Tensor,
#                       lambda_op: torch.Tensor,
#                       lambda_init: torch.Tensor,
#                       lambda_bound: torch.Tensor,
#                       deriv_ival: torch.Tensor = None,
#                       save_graph: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Compute l2 loss.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#             save_graph (bool, optional): saving computational graph. Defaults to True.
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         if bval is None:
#             return torch.sum(torch.mean((operator) ** 2, 0))
#
#         loss_oper, op = self._loss_op(operator, lambda_op)
#         dtype = op.dtype
#         loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#
#         lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
#         lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)
#
#         if self.method == 'PINN':
#             loss = loss_oper + loss_bnd
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_op_normalized.T + \
#                                   bval_diff @ lambda_bound_normalized.T
#         elif self.method == 'PI_DeepONet':  # begin of change
#             loss_ics, ival_diff_discrepancy, ival_diff_deriv, = self._loss_ics(ival,
#                                                                                true_ival,
#                                                                                deriv_ival,
#                                                                                lambda_init)
#             loss = loss_oper + loss_ics + loss_bnd
#
#             lambda_initial_normalized = lambda_prepare(ival, 1).to(dtype)
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_op_normalized.T + \
#                                   (ival_diff_discrepancy + ival_diff_deriv) @ lambda_initial_normalized.T + \
#                                   bval_diff @ lambda_bound_normalized.T  # end of change
#
#         # TODO make decorator and apply it for all losses.
#         if not save_graph:
#             temp_loss = loss.detach()
#             del loss
#             torch.cuda.empty_cache()
#             loss = temp_loss
#
#         return loss, loss_normalized
#
#     def _causal_loss(self,
#                      operator: torch.Tensor,
#                      ival: torch.Tensor,
#                      bval: torch.Tensor,
#                      true_ival: torch.Tensor,
#                      true_bval: torch.Tensor,
#                      lambda_bound: torch.Tensor,
#                      lambda_init: torch.Tensor,
#                      deriv_ival: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Computes causal loss, which is calculated with weights matrix:
#         W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
#         to t_i moment of time. This loss function should be used when one
#         of the DE independent parameter is time.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
#         res = torch.mean(res, axis=1).reshape(self.n_t, 1)
#         m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
#         with torch.no_grad():
#             w = torch.exp(- self.tol * (m @ res))
#
#         loss_oper = torch.mean(w * res)
#         loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#         lambda_bound_normalized = lambda_prepare(bval, 1)
#
#         if self.method == 'PINN':
#             loss = loss_oper + loss_bnd
#
#             with torch.no_grad():
#                 loss_normalized = loss_oper + \
#                                   lambda_bound_normalized @ bval_diff
#
#         elif self.method == 'PI_DeepONet':  # begin of change
#             loss_ics, ival_diff_discrepancy, ival_diff_deriv, = self._loss_ics(ival,
#                                                                                true_ival,
#                                                                                deriv_ival,
#                                                                                lambda_init)
#             loss = loss_oper + loss_ics + loss_bnd
#
#             lambda_initial_normalized = lambda_prepare(ival, 1)
#
#             with torch.no_grad():
#                 loss_normalized = loss_oper + \
#                                   lambda_initial_normalized @ (ival_diff_discrepancy + ival_diff_deriv) + \
#                                   lambda_bound_normalized @ bval_diff  # end of change
#
#         return loss, loss_normalized
#
#     def _weak_loss(self,
#                    operator: torch.Tensor,
#                    ival: torch.Tensor,
#                    bval: torch.Tensor,
#                    true_ival: torch.Tensor,
#                    true_bval: torch.Tensor,
#                    lambda_op: torch.Tensor,
#                    lambda_init: torch.Tensor,
#                    lambda_bound: torch.Tensor,
#                    deriv_ival: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Weak solution of O/PDE problem.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         if bval is None:
#             return sum(operator)
#
#         loss_oper, op = self._loss_op(operator, lambda_op)
#         loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#
#         lambda_op_normalized = lambda_prepare(operator, 1)
#         lambda_bound_normalized = lambda_prepare(bval, 1)
#
#         if self.method == 'PINN':
#             loss = loss_oper + loss_bnd
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_op_normalized.T + \
#                                   bval_diff @ lambda_bound_normalized.T
#         elif self.method == 'PI_DeepONet':
#             loss_init, ival_diff_discrepancy, ival_diff_deriv = self._loss_ics(ival,
#                                                                                true_ival,
#                                                                                deriv_ival,
#                                                                                lambda_init)
#             lambda_initial_normalized = lambda_prepare(ival, 1)
#
#             loss = loss_oper + loss_init + loss_bnd
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_op_normalized.T + \
#                                   (ival_diff_discrepancy + ival_diff_deriv) @ lambda_initial_normalized.T + \
#                                   bval_diff @ lambda_bound_normalized.T
#
#         return loss, loss_normalized
#
#     def compute(self,
#                 operator: torch.Tensor,
#                 bval: torch.Tensor,
#                 true_bval: torch.Tensor,
#                 lambda_op: torch.Tensor,
#                 lambda_bound: torch.Tensor,
#                 ival: torch.Tensor = None,
#                 true_ival: torch.Tensor = None,
#                 lambda_init: torch.Tensor = None,
#                 deriv_ival: torch.Tensor = None,
#                 save_graph: bool = True) -> Union[_default_loss, _weak_loss, _causal_loss]:
#         """ Setting the required loss calculation method.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#             save_graph (bool, optional): saving computational graph. Defaults to True.
#
#         Returns:
#             Union[default_loss, weak_loss, causal_loss]: A given calculation method.
#         """
#
#         if self.mode in ('mat', 'autograd'):
#             if bval is None:
#                 print('No bconds is not possible, returning infinite loss')
#                 return np.inf
#
#         inputs = [operator,
#                   ival,
#                   bval,
#                   true_ival,
#                   true_bval,
#                   lambda_op,
#                   lambda_init,
#                   lambda_bound,
#                   deriv_ival]
#
#         if self.weak_form is not None and self.weak_form != []:
#             return self._weak_loss(*inputs)
#         elif self.tol != 0:
#             return self._causal_loss(*inputs)
#         else:
#             return self._default_loss(*inputs, save_graph)


# ORIGINAL losses.py file
"""Module for losses calculation"""

from typing import Tuple, Union
import numpy as np
import torch

from tedeous.input_preprocessing import lambda_prepare


# class Losses():
#     """
#     Class which contains all losses.
#     """
#
#     def __init__(self,
#                  mode: str,
#                  weak_form: Union[None, list],
#                  n_t: int,
#                  tol: Union[int, float],
#                  method: str = 'PINN'):
#         """
#         Args:
#             mode (str): calculation mode, *NN, autograd, mat*.
#             weak_form (Union[None, list]): list of basis functions if form is weak.
#             n_t (int): number of unique points in time dinension.
#             tol (Union[int, float])): penalty in *casual loss*.
#         """
#
#         self.mode = mode
#         self.weak_form = weak_form
#         self.n_t = n_t
#         self.tol = tol
#         self.method = method
#         # TODO: refactor loss_op, loss_bcs into one function, carefully figure out when bval
#         # is None + fix causal_loss operator crutch (line 76).
#
#     # Without change
#     def _loss_op(self,
#                  operator: torch.Tensor,
#                  lambda_op: torch.Tensor
#                  ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Operator term in loss calc-n.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#
#         Returns:
#             loss_operator (torch.Tensor): operator term in loss.
#             op (torch.Tensor): MSE of operator on the whole grid.
#         """
#         if self.weak_form is not None and self.weak_form != []:
#             op = operator
#         else:
#             op = torch.mean(operator ** 2, 0)
#
#         loss_operator = op @ lambda_op.T
#         return loss_operator, op
#
#     # Without change
#     def _loss_bcs(self,
#                   bval: torch.Tensor,
#                   true_bval: torch.Tensor,
#                   lambda_bound: torch.Tensor
#                   ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Computes boundary loss for corresponding type.
#
#         Args:
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#
#         Returns:
#             loss_bnd (torch.Tensor): boundary term in loss.
#             bval_diff (torch.Tensor): MSE of all boundary con-s.
#         """
#
#         bval_diff = torch.mean((bval - true_bval) ** 2, 0)
#
#         loss_bnd = bval_diff @ lambda_bound.T
#         return loss_bnd, bval_diff
#
#     # Change for PI_DeepONet
#     def _default_loss(self,
#                       operator: torch.Tensor,
#                       bval: torch.Tensor,
#                       true_bval: torch.Tensor,
#                       lambda_op: torch.Tensor,
#                       lambda_bound: torch.Tensor,
#                       ival: torch.Tensor = None,
#                       ival_deriv: torch.Tensor = None,
#                       true_ival: torch.Tensor = None,
#                       lambda_init: torch.Tensor = None,
#                       save_graph: bool = True
#                       ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Compute l2 loss.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#             ival (torch.Tensor):
#             ival_deriv (torch.Tensor):
#             true_ival (torch.Tensor):
#             lambda_init (torch.Tensor):
#             save_graph (bool, optional): saving computational graph. Defaults to True.
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         loss = None
#
#         if self.method == 'PINN':
#             # Source code
#             if bval is None:
#                 return torch.sum(torch.mean((operator) ** 2, 0))
#
#             loss_oper, op = self._loss_op(operator, lambda_op)
#             dtype = op.dtype
#             loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#             loss = loss_oper + loss_bnd
#
#             lambda_op_normalized = lambda_prepare(operator, 1).to(dtype)
#             lambda_bound_normalized = lambda_prepare(bval, 1).to(dtype)
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_op_normalized.T + \
#                                   bval_diff @ lambda_bound_normalized.T
#
#             # TODO make decorator and apply it for all losses.
#             if not save_graph:
#                 temp_loss = loss.detach()
#                 del loss
#                 torch.cuda.empty_cache()
#                 loss = temp_loss
#
#         elif self.method == 'PI_DeepONet':
#             # ######################################################################
#             # functions signatures -->
#             # def loss_ics(operator_net, u, grid, outputs): MSE
#             # def loss_bcs(operator_net, u, grid, outputs): MSE
#             # def loss_res(operator_net, u, grid): norm
#             #
#             # operator_net --> val (ival or bval or something else)
#             # outputs.flatten() --> true_val
#
#             loss_ic1, ival_diff = self._loss_bcs(ival, true_ival.flatten(), lambda_init)
#             loss_ic2, ival_op = self._loss_op(ival_deriv, lambda_init)
#             loss_ics = loss_ic1 + loss_ic2
#             ival_diff = ival_op + ival_diff
#
#             s_bc1_pred, s_bc2_pred = bval
#
#             loss_bc1, bval_diff1 = self._loss_bcs(s_bc1_pred, true_bval[:, 0], lambda_bound)
#             loss_bc2, bval_diff2 = self._loss_bcs(s_bc2_pred, true_bval[:, 1], lambda_bound)
#             loss_bc = loss_bc1 + loss_bc2
#             bval_diff = bval_diff1 + bval_diff2
#
#             loss_op, op = self._loss_op(operator, lambda_op)
#             dtype = operator.dtype
#
#             loss = loss_ics + loss_bc + loss_op
#
#             # ######################################################################
#
#             lambda_residual_normalized = lambda_prepare(operator, 1).to(dtype)
#             lambda_init_normalized = lambda_prepare(ival, 1).to(dtype) + lambda_prepare(ival_deriv, 1).to(dtype)
#             lambda_bound_normalized = lambda_prepare(s_bc1_pred, 1).to(dtype) + lambda_prepare(s_bc2_pred, 1).to(dtype)
#
#             with torch.no_grad():
#                 loss_normalized = op @ lambda_residual_normalized + \
#                                   ival_diff @ lambda_init_normalized + \
#                                   bval_diff @ lambda_bound_normalized
#
#             # TODO make decorator and apply it for all losses.
#             if not save_graph:
#                 temp_loss = loss.detach()
#                 del loss
#                 torch.cuda.empty_cache()
#                 loss = temp_loss
#
#         return loss, loss_normalized
#
#     # Change for PI_DeepONet
#     def _causal_loss(self,
#                      operator: torch.Tensor,
#                      bval: torch.Tensor,
#                      true_bval: torch.Tensor,
#                      lambda_op: torch.Tensor,
#                      lambda_init: torch.Tensor,
#                      lambda_bound: torch.Tensor,
#                      ival: torch.Tensor = None,
#                      ival_deriv: torch.Tensor = None,
#                      true_ival: torch.Tensor = None,
#                      save_graph: bool = True
#                      ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Computes causal loss, which is calculated with weights matrix:
#         W = exp(-tol*(Loss_i)) where Loss_i is sum of the L2 loss from 0
#         to t_i moment of time. This loss function should be used when one
#         of the DE independent parameter is time.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#             ival (torch.Tensor):
#             ival_deriv (torch.Tensor):
#             true_ival (torch.Tensor):
#             lambda_init (torch.Tensor):
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         res = torch.sum(operator ** 2, dim=1).reshape(self.n_t, -1)
#         res = torch.mean(res, axis=1).reshape(self.n_t, 1)
#         m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
#         with torch.no_grad():
#             w = torch.exp(- self.tol * (m @ res))
#
#         loss_oper = torch.mean(w * res)
#
#         loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#
#         loss = loss_oper + loss_bnd
#
#         lambda_bound_normalized = lambda_prepare(bval, 1)
#         with torch.no_grad():
#             loss_normalized = loss_oper + \
#                               lambda_bound_normalized @ bval_diff
#
#         return loss, loss_normalized
#
#     # Change for PI_DeepONet
#     def _weak_loss(self,
#                    operator: torch.Tensor,
#                    bval: torch.Tensor,
#                    true_bval: torch.Tensor,
#                    lambda_op: torch.Tensor,
#                    lambda_bound: torch.Tensor
#                    ) -> Tuple[torch.Tensor, torch.Tensor]:
#         """ Weak solution of O/PDE problem.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#
#         Returns:
#             loss (torch.Tensor): loss.
#             loss_normalized (torch.Tensor): loss, where regularization parameters are 1.
#         """
#
#         if bval is None:
#             return sum(operator)
#
#         loss_oper, op = self._loss_op(operator, lambda_op)
#
#         loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)
#         loss = loss_oper + loss_bnd
#
#         lambda_op_normalized = lambda_prepare(operator, 1)
#         lambda_bound_normalized = lambda_prepare(bval, 1)
#
#         with torch.no_grad():
#             loss_normalized = op @ lambda_op_normalized.T + \
#                               bval_diff @ lambda_bound_normalized.T
#
#         return loss, loss_normalized
#
#     def compute(self,
#                 operator: torch.Tensor,
#                 bval: torch.Tensor,
#                 true_bval: torch.Tensor,
#                 lambda_op: torch.Tensor,
#                 lambda_bound: torch.Tensor,
#                 ival: torch.Tensor = None,
#                 ival_deriv: torch.Tensor = None,
#                 true_ival: torch.Tensor = None,
#                 lambda_init: torch.Tensor = None,
#                 save_graph: bool = True
#                 ) -> Union[_default_loss, _weak_loss, _causal_loss]:
#         """ Setting the required loss calculation method.
#
#         Args:
#             operator (torch.Tensor): operator calc-n result.
#             For more details to eval module -> operator_compute().
#             bval (torch.Tensor): calculated values of boundary conditions.
#             true_bval (torch.Tensor): true values of boundary conditions.
#             lambda_op (torch.Tensor): regularization parameter for operator term in loss.
#             lambda_bound (torch.Tensor): regularization parameter for boundary term in loss.
#             ival (torch.Tensor):
#             ival_deriv (torch.Tensor):
#             true_ival (torch.Tensor):
#             lambda_init (torch.Tensor):
#             save_graph (bool, optional): saving computational graph. Defaults to True.
#
#         Returns:
#             Union[default_loss, weak_loss, causal_loss]: A given calculation method.
#         """
#
#         if self.mode in ('mat', 'autograd'):
#             if bval is None:
#                 print('No bconds is not possible, returning infinite loss')
#                 return np.inf
#
#         inputs = [operator, bval, true_bval, lambda_op, lambda_bound,
#                   ival, ival_deriv, true_ival, lambda_init]
#
#         if self.weak_form is not None and self.weak_form != []:
#             return self._weak_loss(*inputs)
#         elif self.tol != 0:
#             return self._causal_loss(*inputs)
#         else:
#             return self._default_loss(*inputs, save_graph)


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
            op = torch.mean(operator**2, 0)

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

        bval_diff = torch.mean((bval - true_bval)**2, 0)

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
            loss_normalized = op @ lambda_op_normalized.T +\
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

        res = torch.sum(operator**2, dim=1).reshape(self.n_t, -1)
        res = torch.mean(res, axis=1).reshape(self.n_t, 1)
        m = torch.triu(torch.ones((self.n_t, self.n_t), dtype=res.dtype), diagonal=1).T
        with torch.no_grad():
            w = torch.exp(- self.tol * (m @ res))

        loss_oper = torch.mean(w * res)

        loss_bnd, bval_diff = self._loss_bcs(bval, true_bval, lambda_bound)

        loss = loss_oper + loss_bnd

        lambda_bound_normalized = lambda_prepare(bval, 1)
        with torch.no_grad():
            loss_normalized = loss_oper +\
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
            loss_normalized = op @ lambda_op_normalized.T +\
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
