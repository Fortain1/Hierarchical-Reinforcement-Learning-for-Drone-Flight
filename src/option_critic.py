import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

DEFAULT_HIDDEN_UNITS = 128

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.weight.data.mul_(1)
        nn.init.constant_(m.bias.data, 0)

def tensor(x, device="cuda"):
    if torch.is_tensor(x):
        return x
    if not torch.cuda.is_available():
            device = "cpu"
    x = np.asarray(x, dtype=np.float32)
    x = torch.tensor(x, device=torch.device(device), dtype=torch.float32)
    return x

class OptionCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_options, worker_index ,device="cuda", ):
        super(OptionCritic, self).__init__()

        self.worker_index = worker_index
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_options = num_options
        self.policy_over_option = PolicyOverOptions(obs_dim, num_options)
        self.options = nn.ModuleList([IntraOptionPolicy(obs_dim, action_dim) for _ in range(num_options)])
        if not torch.cuda.is_available():
            device = "cpu"
        print("using device: %s" % device)
        self.device = device
        self.apply(init_weights)
        self.to(torch.device(self.device))

    def forward(self, x, prev_option, is_init_state):
        x = tensor(x, self.device)

        mean = []
        std = []
        beta = []
        for option_net in self.options:
            mean_action, std_action, termination_prob  = option_net(x)
            mean.append(mean_action.unsqueeze(1))
            std.append(std_action.unsqueeze(1))
            beta.append(termination_prob)
        mean = torch.cat(mean, dim=1)
        std = torch.cat(std, dim=1)
        beta = torch.cat(beta, dim=1)

        option_probs, q_option = self.policy_over_option(x)
        mask = torch.zeros_like(option_probs)
        mask[self.worker_index, prev_option] = 1
        is_init_states = is_init_state.view(-1, 1).expand(-1, option_probs.size(1))
        pi_h = torch.where(is_init_states, option_probs, beta * option_probs + (1 - beta) * mask)
        option = torch.distributions.Categorical(probs = pi_h).sample()
     
        mean = mean[self.worker_index, option]
        std = std[self.worker_index, option]
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        
        return action, beta, option
    
    def evaluate_actions(self, obs, option, action):
        mean = []
        std = []
        for j, i in enumerate(option):
            mean_action, std_action, termination_prob = self.options[i](obs[j].reshape(1,self.obs_dim))
            mean.append(mean_action)
            std.append(std_action)
        dist = torch.distributions.Normal(torch.stack(mean).squeeze(), torch.stack(std).squeeze())
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        option_probs, q_option = self.policy_over_option(obs.to(torch.float))
        return q_option, log_prob, entropy
    
    def evaluate_options(self, obs):
        option_probs, q_option = self.policy_over_option(obs.to(torch.float))
        return option_probs

class PolicyOverOptions(nn.Module):
    def __init__(self, obs_dim, num_options):
        super(PolicyOverOptions, self).__init__()

        self.net_options = nn.Sequential(
            nn.Linear(obs_dim, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, num_options),
            nn.Softmax(dim=-1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, num_options)
        )

    def forward(self, x):
        option_probs = self.net_options(x)
        q_option = self.value_net(x)

        return option_probs, q_option


class IntraOptionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(IntraOptionPolicy, self).__init__()

        self.termination_net = nn.Sequential(
            nn.Linear(obs_dim, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, 1),
            nn.Sigmoid()
        )

        self.net_mean = nn.Sequential(
            nn.Linear(obs_dim, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, DEFAULT_HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(DEFAULT_HIDDEN_UNITS, action_dim),
            nn.Tanh()
        )
        self.std = nn.Parameter(torch.zeros((1, action_dim)))

    def forward(self, x):
        mean_action = self.net_mean(x)
        std_action = F.softplus(self.std).expand(mean_action.size(0), -1) # ?
        termination_prob = self.termination_net(x)

        return mean_action, std_action, termination_prob
