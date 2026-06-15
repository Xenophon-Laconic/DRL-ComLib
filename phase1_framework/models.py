import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )

    def get_action_and_logprob(self, x, action=None):
        logits = self.network(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
    
    def get_action_and_value(self, x, action=None):
        action, logprob, entropy = self.get_action_and_logprob(x, action)
        return action, logprob, entropy, None  # None for value — actor has no critic


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, x):
        return self.network(x)


class Agent(nn.Module):
    """Learner-side container owning both actor and critic."""
    def __init__(self, env):
        super().__init__()
        obs_dim = int(np.array(env.observation_space.shape).prod())
        action_dim = env.action_space.n
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)

    def get_value(self, x):
        return self.critic.get_value(x)

    def get_action_and_value(self, x, action=None):
        action, logprob, entropy = self.actor.get_action_and_logprob(x, action)
        value = self.critic.get_value(x)
        return action, logprob, entropy, value