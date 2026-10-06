"""
Reinforcement Learning Dynamic Dispatching & Headway Regulation Engine.
Implements a Gymnasium-style metro dispatching environment and a Deep Q-Network (DQN) agent
to eliminate train bunching and optimize platform crowd dwell times.
"""
from typing import Tuple, List, Dict, Any, Deque
from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import RL_DISPATCHER_CONFIG, CHECKPOINT_DIR


class MetroDispatchEnv:
    """
    Simulation Environment for dynamic headway regulation on a metro line.
    Models train bunching, passenger queue accumulations, boarding dwell times, and holding actions.
    """

    def __init__(self, num_trains: int = 6, num_stations: int = 7, target_headway: float = 180.0):
        self.num_trains = num_trains
        self.num_stations = num_stations
        self.target_headway = target_headway
        self.nominal_travel_time = 120.0 # 2 minutes run between stations

        # State dimensions: 5 features per train
        # [headway_err, platform_crowd, downstream_crowd, train_load, schedule_delay]
        self.state_dim = 5
        # 5 Discrete Actions:
        # 0: -10s dwell (hurry)
        # 1: 0s dwell (standard 25s)
        # 2: +10s dwell (slight holding)
        # 3: +25s dwell (major holding to prevent bunching)
        # 4: Skip-stop (bypass to relieve downstream crunch)
        self.action_dim = 5

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Resets trains at nominal intervals along the line.
        """
        self.train_positions = np.linspace(0, self.num_stations - 1, self.num_trains)
        self.train_stations = np.floor(self.train_positions).astype(int)
        self.headways = np.full(self.num_trains, self.target_headway, dtype=np.float32)
        self.platform_queues = np.random.uniform(150, 400, size=self.num_stations).astype(np.float32)
        self.train_loads = np.random.uniform(300, 700, size=self.num_trains).astype(np.float32)
        self.schedule_delays = np.zeros(self.num_trains, dtype=np.float32)
        self.current_train_idx = 0
        self.step_count = 0

        return self._get_state(self.current_train_idx)

    def _get_state(self, train_idx: int) -> np.ndarray:
        curr_st = self.train_stations[train_idx]
        next_st = (curr_st + 1) % self.num_stations

        headway_err = (self.headways[train_idx] - self.target_headway) / self.target_headway
        platform_crowd = self.platform_queues[curr_st] / 1500.0
        downstream_crowd = self.platform_queues[next_st] / 1500.0
        train_load = self.train_loads[train_idx] / 1500.0
        delay_norm = self.schedule_delays[train_idx] / 300.0

        return np.array([headway_err, platform_crowd, downstream_crowd, train_load, delay_norm], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Advances the environment based on the dispatcher's dwell/holding decision.
        """
        self.step_count += 1
        t_idx = self.current_train_idx
        st = self.train_stations[t_idx]

        # Action translation
        dwell_offsets = {0: -10.0, 1: 0.0, 2: 10.0, 3: 25.0, 4: -25.0} # 4 is skip
        base_dwell = 25.0
        applied_dwell = max(15.0, base_dwell + dwell_offsets[action])

        is_skip = (action == 4)
        if is_skip:
            applied_dwell = 0.0

        # Passenger boarding/alighting dynamics
        if not is_skip:
            alighting = min(self.train_loads[t_idx], np.random.uniform(50, 120))
            self.train_loads[t_idx] -= alighting
            available_capacity = 1500.0 - self.train_loads[t_idx]
            boarding = min(self.platform_queues[st], available_capacity, applied_dwell * 12.0)
            self.platform_queues[st] -= boarding
            self.train_loads[t_idx] += boarding

        # Update headway of this train relative to leader
        # If dwell is held longer, headway behind this train increases, headway ahead decreases
        leader_idx = (t_idx - 1) % self.num_trains
        stochastic_track_delay = np.random.normal(0.0, 5.0)
        segment_time = self.nominal_travel_time + applied_dwell + stochastic_track_delay

        # Headway update
        self.headways[t_idx] += (segment_time - (self.nominal_travel_time + base_dwell))
        self.schedule_delays[t_idx] = max(0.0, self.schedule_delays[t_idx] + applied_dwell - base_dwell)

        # Passenger arrival at platforms in the meantime (Poisson flow)
        for s in range(self.num_stations):
            new_arrivals = np.random.poisson(lam=35.0)
            self.platform_queues[s] = min(2000.0, self.platform_queues[s] + new_arrivals)

        # Move train to next station
        self.train_stations[t_idx] = (self.train_stations[t_idx] + 1) % self.num_stations

        # -------------------------------------------------------------
        # Reward Function: Multi-Objective Transit Penalties
        # -------------------------------------------------------------
        headway_deviation = abs(self.headways[t_idx] - self.target_headway)
        platform_penalty = np.sum(self.platform_queues) / 1000.0
        delay_penalty = self.schedule_delays[t_idx] / 60.0
        skip_penalty = 15.0 if is_skip else 0.0

        # Objective: minimize variance in headways and minimize platform crowds
        reward = - (0.05 * headway_deviation + 0.02 * platform_penalty + 0.1 * delay_penalty + skip_penalty)

        # Next train turn in line
        self.current_train_idx = (self.current_train_idx + 1) % self.num_trains
        next_state = self._get_state(self.current_train_idx)

        done = self.step_count >= 150 # Episode horizon

        info = {
            "applied_dwell": applied_dwell,
            "headway": self.headways[t_idx],
            "total_waiting_passengers": float(np.sum(self.platform_queues)),
            "is_skip": is_skip
        }

        return next_state, float(reward), done, info


class QNetwork(nn.Module):
    """
    Deep Q-Network for Dwell & Headway Control.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class RLMetroDispatcher:
    """
    Deep Q-Learning Agent with Experience Replay & Target Network.
    """

    def __init__(self, state_dim: int = 5, action_dim: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = RL_DISPATCHER_CONFIG["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.985
        self.batch_size = RL_DISPATCHER_CONFIG["batch_size"]
        self.memory: Deque = deque(maxlen=10000)

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=RL_DISPATCHER_CONFIG["learning_rate"])

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values).item())

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s_t = torch.FloatTensor(np.array(states))
        a_t = torch.LongTensor(actions).unsqueeze(1)
        r_t = torch.FloatTensor(rewards).unsqueeze(1)
        ns_t = torch.FloatTensor(np.array(next_states))
        d_t = torch.FloatTensor(dones).unsqueeze(1)

        # Current Q-values
        curr_q = self.policy_net(s_t).gather(1, a_t)

        # Target Q-values
        with torch.no_grad():
            max_next_q = self.target_net(ns_t).max(1)[0].unsqueeze(1)
            target_q = r_t + (1 - d_t) * self.gamma * max_next_q

        loss = F.mse_loss(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_rl_dispatcher(episodes: int = 100) -> Tuple[RLMetroDispatcher, List[float]]:
    """
    Trains the RL Dispatching Agent on the Metro Simulation Environment.
    """
    env = MetroDispatchEnv()
    agent = RLMetroDispatcher(env.state_dim, env.action_dim)
    episode_rewards = []

    print(f"🚆 Starting RL Headway Dispatcher Training ({episodes} episodes)...")
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update_model()
            state = next_state
            total_reward += reward

        if ep % 5 == 0:
            agent.update_target_network()

        episode_rewards.append(total_reward)
        if ep % 20 == 0 or ep == 1:
            print(f"Episode {ep:03d}/{episodes:03d} | Total Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save trained policy
    torch.save(agent.policy_net.state_dict(), CHECKPOINT_DIR / "rl_dispatcher.pt")
    print(f"✅ RL Dispatcher Policy saved -> {CHECKPOINT_DIR / 'rl_dispatcher.pt'}")
    return agent, episode_rewards
