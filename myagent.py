from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class AgentTransition:
    state_tensor: torch.Tensor
    action_index: int
    log_probability: torch.Tensor
    state_value: torch.Tensor
    reward: float
    done: bool


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_count: int):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state_tensor: torch.Tensor):
        shared_features = self.shared_layers(state_tensor)
        logits = self.policy_head(shared_features)
        value = self.value_head(shared_features)
        return logits, value


class SelfPlayActorCriticAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 0.0005,
        discount_factor: float = 0.99,
        hidden_size: int = 256,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.discount_factor = discount_factor
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.action_size = action_size
        self.model = ActorCriticNetwork(observation_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def preprocess_observation(self, observation: Dict[str, np.ndarray]) -> torch.Tensor:
        board_tensor = observation["board"].astype(np.float32)
        flat_board = board_tensor.reshape(-1)
        state_tensor = torch.from_numpy(flat_board).to(self.device)
        return state_tensor

    def select_action(self, observation: Dict[str, np.ndarray]) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensor = self.preprocess_observation(observation)
        logits, value = self.model(state_tensor)
        action_mask_np = observation["action_mask"].astype(np.float32)
        action_mask = torch.from_numpy(action_mask_np).to(self.device)

        invalid_entries = action_mask <= 0.0
        # Mask invalid actions by pushing their logits far negative before sampling.
        large_negative = torch.full_like(logits, -1000000000.0)
        masked_logits = torch.where(invalid_entries, large_negative, logits)

        distribution = Categorical(logits=masked_logits)
        sampled_action = distribution.sample()
        action_log_prob = distribution.log_prob(sampled_action)
        action_index = int(sampled_action.item())
        return action_index, action_log_prob, value.squeeze(-1), state_tensor

    def _compute_discounted_returns(self, transitions: List[AgentTransition]) -> torch.Tensor:
        # Compute discounted return from the end of each episode segment.
        discounted_returns = []
        running_return = 0.0
        transition_index = len(transitions) - 1
        while transition_index >= 0:
            transition = transitions[transition_index]
            if transition.done:
                running_return = 0.0
            running_return = transition.reward + (self.discount_factor * running_return)
            discounted_returns.append(running_return)
            transition_index -= 1
        discounted_returns.reverse()
        returns_tensor = torch.tensor(discounted_returns, dtype=torch.float32, device=self.device)
        return returns_tensor

    def update_policy(self, transitions: List[AgentTransition]) -> Dict[str, float]:
        if len(transitions) == 0:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        returns_tensor = self._compute_discounted_returns(transitions)
        log_probs = torch.stack([transition.log_probability for transition in transitions])
        values = torch.stack([transition.state_value for transition in transitions]).squeeze(-1)
        advantages = returns_tensor - values.detach()

        # Advantage = (discounted return - critic value). Use it for the policy gradient term.
        policy_loss = -(log_probs * advantages).mean()
        value_loss = torch.mean((returns_tensor - values) ** 2)

        # Entropy term discourages premature collapse to one move.
        entropy_values = []
        for transition in transitions:
            logits, _ = self.model(transition.state_tensor)
            entropy_distribution = Categorical(logits=logits)
            entropy_values.append(entropy_distribution.entropy())
        entropy_tensor = torch.stack(entropy_values).mean()

        total_loss = policy_loss + (self.value_loss_weight * value_loss) - (self.entropy_weight * entropy_tensor)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        metrics = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
        }
        return metrics
