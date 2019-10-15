from collections import deque
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from santorinigo.utils import discounted_returns

class VanillaMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size) 
    def add(self, *args):
        t = {'state':args[0],'action':args[1],'reward':args[2],'next_state':args[3],'done':args[4]}
        self.memory.append(t)
    def sample(self, batch_size):
        ts = random.sample(self.memory, batch_size)
        states = tf.cast(np.vstack([t['state'] for t in ts]),dtype=tf.float32)
        actions = tf.cast(np.vstack([t['action'] for t in ts]),dtype=tf.int32)
        rewards = tf.cast(np.vstack([t['reward'] for t in ts]),dtype=tf.float32)
        next_states = tf.cast(np.vstack([t['next_state'] for t in ts]),dtype=tf.float32)
        dones = tf.cast(np.vstack([t['done'] for t in ts]),dtype=tf.float32)
        return(states,actions,rewards,next_states,dones)
    def __len__(self):
        return len(self.memory)

class PrioritizedMemory:
    def __init__(self, memory_size, alpha = 0.6):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)
        self.alpha = alpha
        self.priority = deque(maxlen=memory_size)
        self.probs = np.zeros(memory_size)
        
    def add(self, *args):
        max_prior = max(self.priority) if self.memory else 1.
        t = {'state':args[0],'action':args[1],'reward':args[2],'next_state':args[3],'done':args[4]}
        self.memory.append(t)
        #give latest transition max priority for optimistic start
        self.priority.append(max_prior)
        
    def prior_to_prob(self):
        #uniform sampling when alpha is 0
        probs = np.array([i**self.alpha for i in self.priority]) 
        self.probs[range(len(self.priority))] = probs
        self.probs /= np.sum(self.probs)
        
    def sample(self, batch_size, beta = 0.4):
        #calculate prob every time we will sample
        self.prior_to_prob()
        idx = np.random.choice(range(self.memory_size), batch_size, replace=False, p=self.probs)
        ts = [self.memory[i] for i in idx]
        
        #stitch tuple
        states = tf.cast(np.vstack([t['state'] for t in ts]),dtype=tf.float32)
        actions = tf.cast(np.vstack([t['action'] for t in ts]),dtype=tf.int32)
        rewards = tf.cast(np.vstack([t['reward'] for t in ts]),dtype=tf.float32)
        next_states = tf.cast(np.vstack([t['next_state'] for t in ts]),dtype=tf.float32)
        dones = tf.cast(np.vstack([t['done'] for t in ts]),dtype=tf.float32)
        
        #importance sampling weights
        #higher beta, higher compensation for prioritized sampling
        sampling_weights = (len(self.memory)*self.probs[idx])**(-beta)
        #normalize by max weight to always scale down
        sampling_weights = sampling_weights / np.max(sampling_weights) 
        sampling_weights = tf.cast(sampling_weights, dtype=tf.float32)
        return(states,actions,rewards,next_states,dones,idx,sampling_weights)
    
    def update_priority(self, idx, losses):
        for i, l in zip(idx, losses): 
            self.priority[i] = l.numpy().squeeze()
        
    def __len__(self):
        return len(self.memory)

class NStepMemory:
    def __init__(self, memory_size, n = 2, update_every=100, gamma = 0.99):
        self.memory_size = memory_size
        self.memory = []
        self.n = n #n=1 is normal q-learning
        self.gamma = 0.99
        self.df = None
        self.update_every = update_every
        
    def add(self, *args):
        t = {'state':args[0],
             'action':args[1],
             'reward':args[2],
             # 'real_next_state':args[3], #for debugging
             'done':args[4],
             'episode':args[5],
             'timestep':args[6]}
        self.memory.append(t)
        if len(self.memory) > self.update_every:
            self.update_df()
            self.memory = []
        
    def update_df(self):
        df = pd.DataFrame(self.memory)
       #self.df['lag_timestep'] = self.df.groupby('episode').timestep.shift(-self.n) #for debugging
        df['next_state'] = df.groupby('episode').state.shift(-self.n)
        df['discounted_return'] = df.groupby('episode').reward.rolling(self.n)\
            .apply(discounted_returns,raw=True).shift(-(self.n-1)).reset_index().reward
        #fill in next_state when done as state; reward as discounted returns
        df.loc[df.next_state.isna(), 'next_state'] = df[df.next_state.isna()]['state']
        df.loc[df.discounted_return.isna(), 'discounted_return'] = df[df.discounted_return.isna()]['reward']

        #concat
        self.df = pd.concat([self.df,df],0).reset_index(drop=True)
        self.df = self.df.tail(self.memory_size)

    def sample(self, batch_size):
        idx = np.random.choice(range(self.df.shape[0]), batch_size, replace=False)
        t = self.df.iloc[idx,:]
        states = tf.cast(np.vstack(t['state']),dtype=tf.float32)
        actions = tf.cast(np.vstack(t['action']),dtype=tf.int32)
        rewards = tf.cast(np.vstack(t['discounted_return']),dtype=tf.float32)
        next_states = tf.cast(np.vstack(t['next_state']),dtype=tf.float32)
        dones = tf.cast(np.vstack(t['done']),dtype=tf.float32)
        return(states,actions,rewards,next_states,dones)
    
    def __len__(self):
        if self.df is None:
            return 0
        else:
            return self.df.shape[0]

class PrioritizedNStepMemory(PrioritizedMemory, NStepMemory):
    def __init__(self, memory_size, alpha=0.6, n = 2, update_every=100, gamma = 0.99):
        self.memory_size = memory_size
        self.memory = []
        self.alpha = 0.6
        self.n = n 
        self.gamma = 0.99
        self.df = None
        self.update_every = update_every
        
    def add(self, *args):
        max_prior = max(self.df.priority) if self.df is not None else 1.
        t = {'state':args[0],
             'action':args[1],
             'reward':args[2],
             # 'real_next_state':args[3], #for debugging
             'done':args[4],
             'episode':args[5],
             'timestep':args[6],
             'priority':max_prior, #give latest transition max priority for optimistic start
             'probs': None
             }
        self.memory.append(t)
        if len(self.memory) > self.update_every:
            self.update_df()
            self.memory = []

    def prior_to_prob(self):
        #uniform sampling when alpha is 0
        self.df['probs'] = self.df.priority**self.alpha
        self.df['probs'] /= self.df.probs.sum()

    def update_priority(self, idx, losses):
        for i, l in zip(idx, losses): 
            self.df.loc[i,'priority'] = l.numpy().squeeze()

    def update_df(self):
        df = pd.DataFrame(self.memory)
        #self.df['lag_timestep'] = self.df.groupby('episode').timestep.shift(-self.n) #for debugging
        df['next_state'] = df.groupby('episode').state.shift(-self.n)
        df['discounted_return'] = df.groupby('episode').reward.rolling(self.n)\
            .apply(discounted_returns,raw=True).shift(-(self.n-1)).reset_index().reward
        #fill in next_state when done as state; reward as discounted returns
        df.loc[df.next_state.isna(), 'next_state'] = df[df.next_state.isna()]['state']
        df.loc[df.discounted_return.isna(), 'discounted_return'] = df[df.discounted_return.isna()]['reward']

        #concat
        self.df = pd.concat([self.df,df],0).reset_index(drop=True)
        self.df = self.df.tail(self.memory_size)

    def sample(self, batch_size, beta = 0.4):
        #calculate prob every time we will sample
        self.prior_to_prob()
        idx = np.random.choice(self.df.shape[0], batch_size, replace=False, p=self.df.probs)
        t = self.df.iloc[idx,:]
        
        #stitch tuple
        states = tf.cast(np.vstack(t['state']),dtype=tf.float32)
        actions = tf.cast(np.vstack(t['action']),dtype=tf.int32)
        rewards = tf.cast(np.vstack(t['discounted_return']),dtype=tf.float32)
        next_states = tf.cast(np.vstack(t['next_state']),dtype=tf.float32)
        dones = tf.cast(np.vstack(t['done']),dtype=tf.float32)
        
        #importance sampling weights
        #higher beta, higher compensation for prioritized sampling
        sampling_weights = (self.df.shape[0]*self.df.probs[idx])**(-beta)
        #normalize by max weight to always scale down
        sampling_weights = sampling_weights / np.max(sampling_weights) 
        sampling_weights = tf.cast(sampling_weights, dtype=tf.float32)
        return(states,actions,rewards,next_states,dones,idx,sampling_weights)
    
    def __len__(self):
        if self.df is None:
            return 0
        else:
            return self.df.shape[0]