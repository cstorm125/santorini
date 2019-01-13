from collections import defaultdict, namedtuple, deque
import random
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'ret', 'advantage'))
class VanillaMemory:
    def __init__(self, capacity, seed = 1412):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity) 
        self.seed = random.seed(seed)
    def add(self, *args):
        t = Transition(*args)
        self.memory.append(t)
    def sample(self, batch_size):
        ts = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([t.state for t in ts])).float().to(device)
        actions = torch.from_numpy(np.vstack([t.action for t in ts])).float().to(device)
        log_probs = torch.from_numpy(np.vstack([t.log_prob for t in ts])).float().to(device)
        rets = torch.from_numpy(np.vstack([t.ret for t in ts])).float().to(device)
        advantages = torch.from_numpy(np.vstack([t.advantage for t in ts])).float().to(device)
        return(states,actions,log_probs,rets,advantages)
    def __len__(self):
        return(len(self.memory))