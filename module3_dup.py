import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

class PPOBuffer:
    """Buffer for storing trajectories for PPO."""
    def __init__(self, obs_dim, act_dim, size, num_envs, device, gamma=0.99, gae_lambda=0.95):
        self.capacity = size
        self.num_envs = num_envs
        self.device = device
        self.obs_buf = torch.zeros((size, num_envs, obs_dim), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size, num_envs, act_dim), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.val_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.term_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.trunc_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.logprob_buf = torch.zeros((size, num_envs), dtype=torch.float32, device=device)
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.ptr = 0

    def store(self, obs, act, rew, val, term, trunc, logprob):
        # CORRECTED: Explicitly convert numpy arrays to tensors on the correct device
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        self.val_buf[self.ptr] = torch.as_tensor(val, dtype=torch.float32, device=self.device)
        self.term_buf[self.ptr] = torch.as_tensor(term, dtype=torch.float32, device=self.device)
        self.trunc_buf[self.ptr] = torch.as_tensor(trunc, dtype=torch.float32, device=self.device)
        self.logprob_buf[self.ptr] = torch.as_tensor(logprob, dtype=torch.float32, device=self.device)
        self.ptr += 1

    def calculate_advantages(self, last_vals, last_terminateds, last_truncateds):
        assert self.ptr == self.capacity, "Buffer not full"
        with torch.no_grad():
            adv_buf = torch.zeros_like(self.rew_buf)
            last_gae = 0.0
            for t in reversed(range(self.capacity)):
                next_vals = last_vals if t == self.capacity - 1 else self.val_buf[t + 1]
                term_mask = 1.0 - last_terminateds if t == self.capacity - 1 else 1.0 - self.term_buf[t + 1]
                trunc_mask = 1.0 - last_truncateds if t == self.capacity - 1 else 1.0 - self.trunc_buf[t + 1]
                delta = self.rew_buf[t] + self.gamma * next_vals * term_mask - self.val_buf[t]
                last_gae = delta + self.gamma * self.gae_lambda * term_mask * trunc_mask * last_gae
                adv_buf[t] = last_gae
            ret_buf = adv_buf + self.val_buf
            return adv_buf, ret_buf

    def get(self):
        assert self.ptr == self.capacity
        self.ptr = 0
        # Flatten the buffer for easy batching
        return (
            self.obs_buf.view(-1, *self.obs_buf.shape[2:]),
            self.act_buf.view(-1, *self.act_buf.shape[2:]),
            self.logprob_buf.view(-1),
            self.val_buf.view(-1),
            self.rew_buf.view(-1),
            self.term_buf.view(-1),
            self.trunc_buf.view(-1)
        )


class PPOAgent(nn.Module):
    """The PPO Actor-Critic Agent."""
    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()
        # Actor Network for mu (mean of action distribution)
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, num_actions), nn.Tanh()  # Actions are in [-1, 1]
        )
        # Diagonal covariance matrix variables (log standard deviation)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 512), nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).clamp(min=1e-6) # Clamp std to avoid numerical issues
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().mean(1), self.get_value(x)