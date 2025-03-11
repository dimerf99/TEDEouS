import torch
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Callable
from collections import OrderedDict

from tedeous.callbacks.plot import Plots
from tedeous.loss_landscape.generate_plot_surface import PlotLossSurface
from tedeous.loss_landscape.visualization_model import VisualizationModel
from tedeous.loss_landscape.early_stopping_plot import EarlyStopping

from tedeous.data import Domain, Conditions, Equation
from tedeous.optimizers.optimizer import Optimizer
from tedeous.callbacks import early_stopping
from tedeous.callbacks.plot import Plots


# def load_loss_surface():
#     """Load loss landscape data."""
#     return torch.load("loss_surface_data.pt")


def compute_reward(prev_loss, current_loss, method="diff"):
    """
    Calculates the reward for the agent.

    Args:
        prev_loss (float): Error in the previous step.
        current_loss (float): Error at the current step.
        method (str): The method for calculating the reward (“diff” or “absolute”).

    Returns:
        float: The value of the reward.
    """
    if method == "diff":
        return prev_loss - current_loss
    elif method == "absolute":
        return -current_loss
    else:
        raise ValueError("Invalid reward method. Use 'diff' or 'absolute'.")


class OptimizerEnv(gym.Env):
    def __init__(self,
                 optimizer_configs: List[Dict],
                 loss_surface_params: dict = None,
                 equation_params: list = None,
                 AE_model_params: dict = None):
        super(OptimizerEnv, self).__init__()

        # raw_state не должен подаваться в окружение.
        # Он должен генерироваться внутри окружения на основе необходимых параметров,
        # которые как раз и передаются в окружение.
        self.optimizer_configs = optimizer_configs
        self.plot_save_ls_params = None

        path_to_plot_model = r"landscape_visualization\test\landscape_visualization\saved_models\PINN_burgers_adam_state_test\model.pt"
        path_to_trajectories = r"landscape_visualization\test\landscape_visualization\trajectories\burgers\adam_5_stars"

        if AE_model_params is None:
            self.AE_model_params = {
                "mode": "NN",
                "num_of_layers": 3,
                "layers_AE": [
                    991,
                    125,
                    15
                ],
                # "path_to_plot_model": path_to_plot_model,
                "num_models": None,
                "from_last": False,
                "prefix": "model-",
                # "path_to_trajectories": path_to_trajectories,
                "every_nth": 1,
                "grid_step": 0.1,
                "d_max_latent": 2,
                "anchor_mode": "circle",
                "rec_weight": 10000.0,
                "anchor_weight": 0.0,
                "lastzero_weight": 0.0,
                "polars_weight": 0.0,
                "wellspacedtrajectory_weight": 0.0,
                "gridscaling_weight": 0.0
            }
        else:
            self.AE_model_params = AE_model_params

        if loss_surface_params is None:
            # There are params examples
            self.loss_surface_params = {
                "loss_type": "loss_total",
                "every_nth": 1,
                "num_of_layers": 3,
                "layers_AE": [
                    991,
                    125,
                    15
                ],
                "batch_size": 32,
                # "path_to_plot_model": path_to_plot_model,
                "num_models": None,
                "from_last": False,
                "prefix": "model-",
                # "path_to_trajectories": path_to_trajectories,
                "loss_name": "loss_total",
                "x_range": [-1.25, 1.25, 25],
                "vmax": -1.0,
                "vmin": -1.0,
                "vlevel": 30.0,
                "key_models": None,
                "key_modelnames": None,
                "density_type": "CKA",
                "density_p": 2,
                "density_vmax": -1,
                "density_vmin": -1,
                "colorFromGridOnly": True
            }
        else:
            self.loss_surface_params = loss_surface_params

        self.equation_params = equation_params

        self.current_optimizer = None
        self.loss_history = []
        self.tolerance = 1e-4

        # self.loss_surface = raw_state['grid_losses']
        # self.grid_xx = raw_state['grid_xx']
        # self.grid_yy = raw_state['grid_yy']

        ################################################################################################################
        # Размерность нужно вытягивать из кода loss landscape и она будет постоянной, т.к.
        # action_dim - список оптимизаторов, он не меняется
        # state_dim - размерность поверхности, мы используем латентное 2D пространство, для генерации поверхности

        self.visualization_model = VisualizationModel(**self.AE_model_params)
        self.plot_loss_surface = None

        # Action - selecting an optimizer with its parameters
        self.action_space = spaces.Discrete(len(self.optimizer_configs))

        # # State - error surface (can be an array)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.visualization_model.latent_dim,
        #                                     dtype=np.float32)
        self.observation_space = 2

        ################################################################################################################

    def reset(self, loss):
        """Reset environment - load error surface, reset history to zero, select starting point."""
        self.current_error = loss
        self.error_history.append(self.current_error)
        return self.loss_surface

    # There is action will be update model (not new optimizer, but new model)
    # Last version of signature: def step(self, action: int, weights: torch.Tensor, current_loss: float, params: list):
    def step(self,
             weights: List[OrderedDict],
             current_loss: float):
        """Applying an action (optimizer selection) and updating the state."""

        # save_equation_loss_surface function parameters:
        # u_exact_test = u(grid_test).reshape(-1)
        grid_res = 50
        u_exact_test = torch.randn(1).reshape(-1)
        grid_test = torch.cartesian_prod(torch.linspace(0, 1, grid_res), torch.linspace(0, 1, grid_res))

        # grid, domain, equation, boundaries = problem_formulation(grid_res)  # from any tedeous.example
        grid, domain, equation, boundaries = self.equation_params

        neurons = 32

        model_layers = [2, neurons, neurons, 1]  # PINN layers

        # # More important parameters
        # "path_to_plot_model": path_to_plot_model  # we will get this one from Closure update model in real time
        # "path_to_trajectories": path_to_trajectories

        ################################################################################################################
        # There is a training of AE model for create loss landscape
        batch_size = 32
        epochs = 600000
        patience_scheduler = 400000
        every_epoch = 100
        cosine_scheduler_patience = 2000
        learning_rate = 0.0005
        resume = True

        optimizer = Optimizer('RMSprop', {'lr': learning_rate}, cosine_scheduler_patience=cosine_scheduler_patience)
        cb_es = EarlyStopping(patience=patience_scheduler)
        # cb_es = early_stopping.EarlyStopping(patience=patience_scheduler)

        # В результате обучения должны выдаваться веса обученного автоэнкодера,
        # которые нужно передать дальше в PlotLossSurface для генерации состояния и отрисовки поверхности
        AEmodel_weights = self.visualization_model.train(optimizer, epochs, every_epoch, batch_size, resume,
                                       callbacks=[cb_es], saved_model=weights)

        self.plot_save_ls_params = [u_exact_test, grid_test, grid, domain, equation, boundaries,
                                    model_layers, AEmodel_weights]

        ################################################################################################################
        self.plot_loss_surface = PlotLossSurface(**self.loss_surface_params)
        raw_state = self.plot_loss_surface.save_equation_loss_surface(*self.plot_save_ls_params)

        state = raw_state['grid_loss']

        prev_loss = self.loss_history[-1]
        reward = compute_reward(prev_loss, current_loss)
        self.loss_history.append(current_loss)

        done = self.current_error < self.tolerance

        return state, reward, done, {}

    def render(self, img_params: dict = None):
        """Display the current error and convergence history."""

        print(f"Optimizer: {self.current_optimizer}, Error: {self.current_error}")

        # Plotting solution
        plot_solution = Plots(**img_params)
        plot_solution.solution_print(forced_call_flag=True)

        # Plotting loss landscape
        self.plot_loss_surface = PlotLossSurface(**self.loss_surface_params)
        self.plot_loss_surface.plotting_equation_loss_surface()

        # plt.figure(figsize=(10, 5))
        # plt.plot(self.loss_history, label='Error')
        # plt.xlabel("Steps")
        # plt.ylabel("Loss")
        # plt.title("Error Dynamics")
        # plt.legend()
        # plt.show()

    def close(self):
        plt.close('all')
