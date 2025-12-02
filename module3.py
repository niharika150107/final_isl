# ... (PPOBuffer class remains the same) ...

class PPOAgent(nn.Module):
    """The PPO Actor-Critic Agent."""
    def __init__(self, num_inputs: int, num_actions: int):
        super(PPOAgent, self).__init__()
        # Actor Network for mu (mean of action distribution)
        # REDUCED LAYER SIZE from 512 to 256
        self.actor_mu = nn.Sequential(
            nn.Linear(num_inputs, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, num_actions), nn.Tanh()
        )
        # Diagonal covariance matrix variables (log standard deviation)
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))

        # Critic Network
        # REDUCED LAYER SIZE from 512 to 256
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).clamp(min=1e-6)
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().mean(1), self.get_value(x)
