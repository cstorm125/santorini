import random
import numpy as np
from qnetwork import *

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    '''
    Number of changes for better version
    * Double: 1
    * Dueling: 1 + 1 new model class
    * Prioritized: 3 + 1 new replay memory class
    * Noisy: 2 + 1 new layer + 1 new model class
    * Distributional: NA
    * Multi-step: NA
    '''
    def __init__(self, state_size , action_size, replay_memory, seed = 1412,
        lr = 1e-3 / 4, bs = 64, nb_hidden = 256,
        gamma=0.99, tau= 1/300, update_interval = 5):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.npseed = np.random.seed(seed)
        self.lr = lr
        self.bs = bs
        self.gamma = gamma
        self.update_interval = update_interval
        self.tau = tau
        self.losses = []

        #vanilla
#         self.qnetwork_local = QNetwork(state_size, action_size, nb_hidden).to(device)
#         self.qnetwork_target = QNetwork(state_size, action_size, nb_hidden).to(device)
        
        #dueling
#         self.qnetwork_local = DuelingNetwork(state_size, action_size, nb_hidden).to(device)
#         self.qnetwork_target = DuelingNetwork(state_size, action_size, nb_hidden).to(device)
        
        #noisy and dueling
        self.qnetwork_local = NoisyDuelingNetwork(state_size, action_size, nb_hidden).to(device)
        self.qnetwork_target = NoisyDuelingNetwork(state_size, action_size, nb_hidden).to(device)
        
        #optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # replay memory
        self.memory = replay_memory
        # count time steps
        self.t_step = 0
        
    def get_eps(self, i, eps_start = 0.9, eps_end = 0.01, eps_decay = 0.9):
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
    
    def get_beta(self, i, beta_start = 0.4, beta_end = 1, beta_growth = 1.05):
        beta = min(beta_start * (beta_growth ** i), beta_end)
        return(beta)
    
    def step(self, state, action, reward, next_state, done, i):
        #add transition to replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every self.t_step
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            if len(self.memory) > self.bs:
                #vanilla
#                 transitions = self.memory.sample(self.bs)
                #prioritized
                transitions = self.memory.sample(self.bs, self.get_beta(i))
                self.learn(transitions, self.gamma)

    def act(self, state, i, return_list = False):
        eps = self.get_eps(i)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #epsilon greedy
        if random.random() > eps:
            action_vals = action_values.to(device).data.numpy()
            action = np.argmax(action_vals)
            actions = np.argsort(action_vals).reshape(-1)
        else:
            actions = np.arange(self.action_size).reshape(-1)
            action = random.choice(actions)
        
        #return_list
        if return_list: 
            return(actions) 
        else:
            return(action)
        
    def vanilla_loss(self,q_targets,q_expected):
        loss = F.mse_loss(q_expected,q_targets)
        return(loss)
    
    def wis_loss(self, q_expected, q_targets, idx, sampling_weights, small_e):
        losses = F.mse_loss(q_expected, q_targets, reduce=False).squeeze(1) * sampling_weights
        self.memory.update_priority(idx,losses+small_e)
        loss = losses.mean()
        return(loss)
        
    def learn(self, transitions, gamma, small_e = 1e-5):
        #vanilla
#         states, actions, rewards, next_states, dones = transitions
        #prioritized
        states, actions, rewards, next_states, dones, idx, sampling_weights = transitions
        
        #vanilla
        #         q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        #double
        max_actions_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions_next.long())

        #compute loss
        q_targets = rewards + (gamma * q_targets_next) * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions.long())
        #vanilla
#         loss = self.vanilla_loss(q_expected, q_targets)
        #prioritized
        loss = self.wis_loss(q_expected,q_targets,idx, sampling_weights, small_e)
        #append for reporting
        self.losses.append(loss)
        
        #backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #noisy; reset noise
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()
        
        #update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
#         self.hard_update(self.qnetwork_local, self.qnetwork_target, 1/self.tau)
      
    def hard_update(self, local_model, target_model, target_interval=1e2):
        if self.t_step % target_interval==0:
            target_model.load_state_dict(local_model.state_dict())
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)