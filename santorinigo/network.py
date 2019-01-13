import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, seed = 1412):
        super(ActorCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=3, stride = 2, padding = 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1, bias=False),
            nn.ReLU()
            )
        self.feature_size = 64 * (state_dim[1]-1)**2
        self.actor_head = nn.Sequential(
                        nn.Linear(self.feature_size,512),
                        nn.ReLU(),
                        nn.Linear(512,action_dim)
                        )
        self.critic_head = nn.Sequential(
                        nn.Linear(self.feature_size,512),
                        nn.ReLU(),
                        nn.Linear(512,1)
                        )

    def forward(self, state):
        x = self.feature_extract(state)
        x = x.view(x.shape[0], -1)
        prob = F.softmax(self.actor_head(x),1)
        dist = Categorical(prob)
        value = self.critic_head(x)
        return(dist,value)