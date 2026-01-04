import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, std=0.5):
        super(ActorCritic, self).__init__()
        
        # Deeper network (64 -> 128) for better approximation
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(std))

        # Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        value = self.critic(state)
        mean = self.actor(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

class PPOAgent:
    def __init__(self, state_dim=2, action_dim=2, lr=5e-4, gamma=0.99, eps_clip=0.2, k_epochs=4):
        self.policy = ActorCritic(state_dim, action_dim, std=1.0) # Higher initial std for exploration
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, std=1.0)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        
        self.mse_loss = nn.MSELoss()
        
        # Memory buffers (Arrays for faster append)
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = [] # Not really used for 1-step logic but good for completeness

    def select_action(self, state):
        # State is (N, 2) numpy
        with torch.no_grad():
            state = torch.FloatTensor(state)
            dist, value = self.policy_old(state)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        return action.numpy(), action_logprob.numpy(), value.numpy()

    def store(self, states, actions, logprobs, rewards):
        # Append batches
        # We process 'batch' physics, so we store 'batch' transitions
        # Input shapes: (N, 2), (N, 2), (N, 2), (N,)
        self.states.append(states)
        self.actions.append(actions)
        self.logprobs.append(logprobs)
        self.rewards.append(rewards)

    def update(self):
        if not self.states:
            return

        # Flatten the buffer:
        # List of (N, D) -> (Batch, D)
        states = torch.FloatTensor(np.concatenate(self.states)) 
        actions = torch.FloatTensor(np.concatenate(self.actions))
        old_logprobs = torch.FloatTensor(np.concatenate(self.logprobs))
        rewards = torch.FloatTensor(np.concatenate(self.rewards))
        
        # Normalize Rewards (Critical for stable gradients)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Since these are 1-step episodes (Action -> End), the "Return" is just the Reward.
        returns = rewards 
        
        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluating old actions and values
            dist, state_values = self.policy(states)
            
            # Action logprobs dim handling
            # old_logprobs is (B, 2)
            action_logprobs = dist.log_prob(actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)
            
            # Sum logprobs over action space (independent Gaussian)
            action_logprobs_sum = action_logprobs.sum(dim=1)
            old_logprobs_sum = old_logprobs.sum(dim=1)
            
            # Ratio
            ratios = torch.exp(action_logprobs_sum - old_logprobs_sum)

            # Advantages
            advantages = returns - state_values.detach()
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Loss
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, returns) - 0.01 * dist_entropy.mean()
            
            # Gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
        # Copy new weights
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
