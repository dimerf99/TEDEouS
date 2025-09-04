# Version -- 1 #########################################################################################################

# import gym
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import deque
# import random
#
# # Установка устройства (CPU или GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # 1. Сеточная нейросеть для аппроксимации Q-функции
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim, neurons=100):
#         super(DQN, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(state_dim, neurons),
#             nn.ReLU(),
#             nn.Linear(neurons, neurons),
#             nn.ReLU(),
#             nn.Linear(neurons, action_dim)
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
#
# # 2. Определение опыта для памяти повторного воспроизведения
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)
#
#     def push(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size):
#         samples = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*samples)
#         return (
#             torch.tensor(states, dtype=torch.float32, device=device),
#             torch.tensor(actions, dtype=torch.int64, device=device),
#             torch.tensor(rewards, dtype=torch.float32, device=device),
#             torch.tensor(next_states, dtype=torch.float32, device=device),
#             torch.tensor(dones, dtype=torch.float32, device=device),
#         )
#
#     def __len__(self):
#         return len(self.buffer)
#
#
# # 3. DQN обучение
# def train_dqn(env,
#               episodes=500,
#               gamma=0.99,
#               epsilon_start=1.0,
#               epsilon_end=0.01,
#               epsilon_decay=500,
#               lr=0.001,
#               batch_size=64,
#               target_update=10,
#               memory_capacity=10000):
#
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n
#
#     # Основная и целевая нейронные сети
#     policy_net = DQN(state_dim, action_dim).to(device)
#     target_net = DQN(state_dim, action_dim).to(device)
#     target_net.load_state_dict(policy_net.state_dict())
#     target_net.eval()
#
#     optimizer = optim.Adam(policy_net.parameters(), lr=lr)
#     memory = ReplayBuffer(memory_capacity)
#
#     epsilon = epsilon_start
#     total_rewards = []
#
#     for episode in range(episodes):
#         state = env.reset()
#         state = np.array(state, dtype=np.float32)
#         total_reward = 0
#
#         while True:
#             # Выбор действия с использованием ε-greedy
#             if random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 with torch.no_grad():
#                     state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#                     action = policy_net(state_tensor).argmax(dim=1).item()
#
#             # Выполнение действия
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.array(next_state, dtype=np.float32)
#
#             # Сохранение опыта в буфер
#             memory.push(state, action, reward, next_state, done)
#
#             state = next_state
#             total_reward += reward
#
#             # Обучение на случайной выборке из памяти
#             if len(memory) >= batch_size:
#                 states, actions, rewards, next_states, dones = memory.sample(batch_size)
#
#                 # Q-значения для текущего состояния
#                 q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
#
#                 # Целевые Q-значения
#                 with torch.no_grad():
#                     next_q_values = target_net(next_states).max(dim=1)[0]
#                     target_q_values = rewards + gamma * next_q_values * (1 - dones)
#
#                 # Обновление нейронной сети
#                 loss = nn.MSELoss()(q_values, target_q_values)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#
#             if done:
#                 break
#
#         # Обновление ε (эпсилон)
#         epsilon = max(epsilon_end, epsilon * np.exp(-1 / epsilon_decay))
#
#         total_rewards.append(total_reward)
#
#         # Обновление целевой сети
#         if episode % target_update == 0:
#             target_net.load_state_dict(policy_net.state_dict())
#
#         print(f"Эпизод: {episode}, Награда: {total_reward}, Epsilon: {epsilon:.4f}")
#
#     return policy_net, total_rewards
#
#
# # 4. Запуск обучения
# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")
#     trained_policy, rewards = train_dqn(env)
#
#     # Тестирование обученной модели
#     state = env.reset()
#     state = np.array(state, dtype=np.float32)
#     total_reward = 0
#
#     while True:
#         env.render()
#         with torch.no_grad():
#             state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#             action = trained_policy(state_tensor).argmax(dim=1).item()
#
#         next_state, reward, done, _ = env.step(action)
#         state = np.array(next_state, dtype=np.float32)
#         total_reward += reward
#
#         if done:
#             print(f"Итоговая награда: {total_reward}")
#             break
#
#     env.close()


# Version -- 2 #########################################################################################################

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import random
# from collections import deque
#
#
# class OptimizerSelectionAgent:
#     def __init__(self, action_space, state_space, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
#                  learning_rate=0.001):
#         # action_space: количество оптимизаторов, которые можно выбрать
#         # state_space: размерность состояния (например, параметры модели PINN)
#
#         self.action_space = action_space
#         self.state_space = state_space
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.learning_rate = learning_rate
#
#         self.memory = deque(maxlen=2000)  # память для обучения
#         self.model = self._build_model()  # модель Q-learning
#         self.target_model = self._build_model()  # целевая модель для Q-learning
#         self.update_target_model()
#
#     def _build_model(self):
#         # Нейросеть для предсказания Q-значений
#         model = nn.Sequential(
#             nn.Linear(self.state_space, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.action_space)
#         )
#         return model
#
#     def update_target_model(self):
#         # Обновление целевой модели
#         self.target_model.load_state_dict(self.model.state_dict())
#
#     def act(self, state):
#         # Выбор действия (оптимизатора) с использованием epsilon-greedy стратегии
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_space)  # случайный выбор
#         state_tensor = torch.FloatTensor(state).unsqueeze(0)
#         q_values = self.model(state_tensor)
#         return torch.argmax(q_values).item()  # выбор действия с наибольшим Q-значением
#
#     def remember(self, state, action, reward, next_state, done):
#         # Сохранение опыта
#         self.memory.append((state, action, reward, next_state, done))
#
#     def replay(self, batch_size):
#         # Обучение на случайном батче из памяти
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory, batch_size)
#
#         for state, action, reward, next_state, done in minibatch:
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
#             next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
#
#             q_values = self.model(state_tensor)
#             q_next = self.target_model(next_state_tensor)
#
#             target = reward + self.gamma * torch.max(q_next) * (1 - done)
#             q_values[0][action] = target
#
#             # Обновление модели
#             self.model.zero_grad()
#             loss = nn.MSELoss()(q_values, target)
#             loss.backward()
#             optim.step()
#
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay
#
#
# # Пример использования:
#
# # Определите пространство состояний и действий
# state_space = 10  # Примерное число параметров для PINN
# action_space = 5  # Примерное число оптимизаторов
#
# # Инициализация агента
# agent = OptimizerSelectionAgent(action_space=action_space, state_space=state_space)
#
# # Пример обучения:
# for episode in range(1000):
#     state = np.random.rand(state_space)  # случайное состояние
#     done = False
#     total_reward = 0
#
#     while not done:
#         action = agent.act(state)
#         next_state = np.random.rand(state_space)  # пример следующего состояния
#         reward = -np.abs(np.random.randn())  # пример вознаграждения, можно менять на основе результата решения
#         done = np.random.rand() > 0.95  # случайное завершение эпизода
#         agent.remember(state, action, reward, next_state, done)
#
#         state = next_state
#         total_reward += reward
#
#     agent.replay(32)  # Обучение модели
#     agent.update_target_model()  # Обновление целевой модели
#     print(f"Episode {episode}, Total Reward: {total_reward}")


# # Version -- 3 #######################################################################################################
#
# # RL wrapper file
#
# import numpy as np
# from tedeous.rl_algorithms import RL_Algorithm


# class RLWrapper:
#     def __init__(self, action_space, state_space, algorithm, **kwargs):
#         """
#         A general wrapper for reinforcement learning algorithms.
#
#         Args:
#             action_space (int): Number of possible actions (e.g., optimizers and steps).
#             state_space (int): Dimension of the state space (e.g., loss surface position).
#             algorithm (RLAlgorithm): An instance of a reinforcement learning algorithm.
#             **kwargs: Additional parameters for specific algorithms.
#         """
#         self.action_space = action_space
#         self.state_space = state_space
#         self.algorithm = algorithm(action_space, state_space, **kwargs)
#
#     def act(self, state):
#         """Select an action based on the current state."""
#         return self.algorithm.act(state)
#
#     def remember(self, state, action, reward, next_state, done):
#         """Store an experience in memory."""
#         self.algorithm.remember(state, action, reward, next_state, done)
#
#     def replay(self, batch_size):
#         """Train the RL model using experience replay."""
#         self.algorithm.replay(batch_size)
#
#     def update_target_model(self):
#         """Update the target model in double-Q architectures."""
#         self.algorithm.update_target_model()
#
#     def get_reward(self, pinn_model, state):
#         """Calculate the reward for a given state."""
#         return self.algorithm.get_reward(pinn_model, state)
#
#
# class PINNModel:
#     def __init__(self):
#         # Инициализация модели, параметров, данных и т.д.
#         pass
#
#     def compute_loss(self, state):
#         # Реализация расчета ошибки (лосса) для текущего состояния
#         # Например, вычисляем отклонение предсказания от истинного значения
#         # Здесь `state` может быть вектором параметров или текущим входом модели
#         a = np.random.random()
#         print('a = ', a)
#         return a
#
#
# # Define state and action space
# state_space = 1
# action_space = 30
#
# epsilon = 0.95
#
# # Initialize the RL agent
# agent = RLWrapper(action_space, state_space, RL_Algorithm)
#
# # Initialize PINN-model
# pinn_model = PINNModel()
#
# # # Example workflow
# # for episode in range(1000):
# #     state = np.random.rand(state_space)
# #     done = False
# #     total_reward = 0
# #
# #     while not done:
# #         action = agent.act(state)
# #         next_state = np.random.rand(state_space)
# #         reward = agent.get_reward(pinn_model, state)  # Replace None with actual PINN model
# #         done = np.random.rand() > epsilon
# #         agent.remember(state, action, reward, next_state, done)
# #         state = next_state
# #         total_reward += reward
# #
# #     agent.replay(32)
# #     agent.update_target_model()
# #     print(f"Episode {episode}, Total Reward: {total_reward}")
#
# trajectory_loader, normalizer = get_trajectory_dataloader(pt_files, batch_size=1, path="путь_к_данным")
#
# trajectory_iterator = iter(trajectory_loader)
#
# import torch.optim as optim
#
# optimizers = [optim.Adam, optim.SGD, optim.RMSprop]
# epochs_choices = [1, 5, 10, 20, 50]
#
# action_space = len(optimizers) * len(epochs_choices)
#
#
# def decode_action(action):
#     optimizer_index = action // len(epochs_choices)
#     epochs_index = action % len(epochs_choices)
#
#     selected_optimizer = optimizers[optimizer_index]
#     selected_epochs = epochs_choices[epochs_index]
#
#     return selected_optimizer, selected_epochs
#
#
# for episode in range(1000):
#     state = next(trajectory_iterator).numpy()
#     done = False
#     total_reward = 0
#
#     while not done:
#         action = agent.act(state)
#         optimizer_class, num_epochs = decode_action(action)
#
#         optimizer = optimizer_class(pinn_model.parameters(), lr=0.001)
#
#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             loss = pinn_model.compute_loss(state)
#             loss.backward()
#             optimizer.step()
#
#         next_state = next(trajectory_iterator).numpy()
#         reward = agent.get_reward(pinn_model, state)
#         done = np.random.rand() > epsilon
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         total_reward += reward
#
#     agent.replay(32)
#     agent.update_target_model()
#     print(f"Episode {episode}, Total Reward: {total_reward}")


# Version -- 4 #########################################################################################################

# RL wrapper file

import torch
import numpy as np
from tedeous.rl_algorithms import RL_Algorithm


class PINNTrainerWithRL:
    def __init__(self, pinn_model, env, state_dim, action_dim, device="cpu"):
        self.pinn_model = pinn_model
        self.env = env
        self.device = device

        self.agent = RL_Algorithm(state_dim, action_dim, device)

        self.loss_history = []

    def train(self, num_iterations=100):
        state = self.env.reset()
        for iteration in range(num_iterations):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = self.agent.select_action(state_tensor)

            optimizer, lr, epochs = self.env.decode_action(action)

            loss = self.train_pinn(optimizer, lr, epochs)
            next_state = self.env.get_state()
            reward = self.compute_reward(loss)

            self.agent.store_transition(state, action, reward, next_state)
            self.agent.optimize()

            state = next_state
            self.loss_history.append(loss.item())

            print(f"Iteration {iteration + 1}, Loss: {loss.item()}, Reward: {reward}")

    def train_pinn(self, optimizer_name, lr, epochs):
        optimizer = self.get_optimizer(optimizer_name, lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.pinn_model.compute_loss()
            loss.backward()
            optimizer.step()

        return loss

    def compute_reward(self, loss):
        if len(self.loss_history) > 0:
            return self.loss_history[-1] - loss.item()
        return 0

    def get_optimizer(self, optimizer_name, lr):
        if optimizer_name == "adam":
            return torch.optim.Adam(self.pinn_model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            return torch.optim.SGD(self.pinn_model.parameters(), lr=lr)
        elif optimizer_name == "rmsprop":
            return torch.optim.RMSprop(self.pinn_model.parameters(), lr=lr)
        else:
            raise ValueError("Неизвестный оптимизатор")












