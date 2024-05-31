from typing import NamedTuple
import numpy as np
import torch


class RolloutData(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    options: torch.Tensor
    prev_options: torch.Tensor
    betas: torch.Tensor
    beta_advantages: torch.Tensor
    entropies: torch.Tensor
    episode_starts: torch.Tensor

class RolloutBuffer:
    actions: np.ndarray  
    returns: np.ndarray
    episode_starts: np.ndarray   # is_init_state
    option_values: np.ndarray
    beta_advantages: np.ndarray
    observations: np.ndarray
    log_probs: np.ndarray
    prev_options: np.ndarray
    options: np.ndarray
    entropies: np.ndarray
    advantages: np.ndarray
    rewards: np.ndarray
    betas: np.ndarray

    def __init__(self, batch_size, buffer_size, n_envs, obs_shape, action_dim, n_options, device) -> None:
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.n_options = n_options
        self.device = device
        self.reset()
      
    def store(self, 
              actions, 
              episode_starts, 
              option_values, 
              observations, 
              log_probs, 
              prev_options,
              options, 
              entropies,
              rewards, 
              betas):
        
        self.observations[self.pos] = observations.clone().cpu().numpy()
        self.actions[self.pos] = actions.clone().cpu().numpy()
        self.betas[self.pos] = betas.clone().cpu().numpy()
        self.option_values[self.pos] = option_values.clone().cpu().numpy()
        
        self.rewards[self.pos] = rewards.clone().cpu().numpy().flatten()
        self.episode_starts[self.pos] = episode_starts.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_probs.clone().cpu().numpy().flatten()
        self.options[self.pos] = options.clone().cpu().numpy().flatten()
        self.prev_options[self.pos] = prev_options.clone().cpu().numpy().flatten()
        self.entropies[self.pos] = entropies.clone().cpu().numpy().flatten()
        
        self.pos += 1

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.betas = np.zeros((self.buffer_size, self.n_envs, self.n_options), dtype=np.float32)
        self.option_values = np.zeros((self.buffer_size, self.n_envs, self.n_options), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.beta_advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.options = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.prev_options = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        self.entropies = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.pos = 0


    def _get_by_indices(self, ids):
        return RolloutData(
            observations=torch.tensor(self.observations[ids], device=self.device).flatten(0,1),
            actions=torch.tensor(self.actions[ids], device=self.device).flatten(0,1),
            old_values=torch.tensor(self.option_values[ids], device=self.device).flatten(0,1),
            old_log_prob=torch.tensor(self.log_probs[ids], device=self.device).flatten().unsqueeze(1),
            advantages=torch.tensor(self.advantages[ids], device=self.device).flatten().unsqueeze(1),
            returns=torch.tensor(self.returns[ids], device=self.device).flatten().unsqueeze(1),
            options=torch.tensor(self.options[ids], device=self.device).flatten().unsqueeze(1),
            prev_options=torch.tensor(self.prev_options[ids], device=self.device).flatten().unsqueeze(1),
            betas=torch.tensor(self.betas[ids], device=self.device).flatten(0,1),
            beta_advantages=torch.tensor(self.beta_advantages[ids], device=self.device).flatten().unsqueeze(1),
            entropies=torch.tensor(self.entropies[ids], device=self.device).flatten().unsqueeze(1),
            episode_starts=torch.tensor(self.episode_starts[ids], device=self.device).flatten().unsqueeze(1),
        )
    
    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95, n_envs=1):
        values = np.concatenate((values, next_value.reshape(1,n_envs,1)))
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step +1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def calculate_advantages(self, prev_option, masks, values, beta, q_option, option_probs, worker_index):
        
        betass = beta[worker_index, prev_option]
        ret = (1 - betass) * q_option[worker_index, prev_option] + \
            betass * torch.max(q_option, dim=-1)[0]
        ret = ret.unsqueeze(-1)
        beta_advs = []
        for i in reversed(range(self.buffer_size)):
            v = (q_option * option_probs).sum(-1).unsqueeze(-1)
            util = np.arange(len(self.prev_options[i]))
            q = self.option_values[i][util, self.prev_options[i]]
            beta_advs.append(q - v.cpu().detach().numpy() + 0.01)
        values = torch.stack(values).cpu().detach().numpy()
        masks = torch.stack(masks).cpu().detach().numpy()
        rets = self.compute_gae(ret.cpu().detach().numpy(), self.rewards, masks, values)
        advs = np.array(rets) - values
        advs = (advs - advs.mean()) / advs.std()
        self.returns = np.array(rets).flatten()
        self.advantages = advs
        self.beta_advantages = np.array(beta_advs).flatten()

    
    def __iter__(self):
        current_length = self.observations.shape[0]
        completed_buffer_length = current_length - current_length % self.batch_size

        ids = np.random.permutation(completed_buffer_length)
        ids_per_batch = np.split(ids, len(ids)//self.batch_size)

        for ids_in_batch in ids_per_batch:
            yield self._get_by_indices(ids_in_batch)