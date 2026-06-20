import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2.0),
    bias_const: float = 0.0,
) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """Policy network used on both actor and learner sides (actor head only)."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_action_and_logprob(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or evaluate) an action and return log-prob and entropy."""
        logits = self.network(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Actor-only variant: no value head is available on the actor side."""
        action, logprob, entropy = self.get_action_and_logprob(x, action)
        return action, logprob, entropy, None


class Critic(nn.Module):
    """State-value function approximator used on the learner side."""

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Agent(nn.Module):
    """Learner-side container owning both actor and critic."""

    def __init__(self, env) -> None:
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape).prod())
        action_dim = env.action_space.n
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic.get_value(x)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return policy action/logprob/entropy and critic value."""
        action, logprob, entropy = self.actor.get_action_and_logprob(x, action)
        value = self.critic.get_value(x)
        return action, logprob, entropy, value