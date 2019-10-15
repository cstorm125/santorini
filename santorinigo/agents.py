import tensorflow as tf
from tensorflow.keras import layers, models
import random
import numpy as np
from santorinigo.networks import DenseNetwork, NoisyDenseNetwork
from santorinigo.memories import PrioritizedMemory, NStepMemory

class QAgent:
    def __init__(self, state_size, action_size):
        
        self.state_size = state_size
        self.action_size = action_size
        self.losses = [] #to store losses for visualization
        self.t_step = 0 # count time steps

        '''
        Fill in your codes to create network, optimizer and loss function here
        '''
        self.network_local = DenseNetwork(self.action_size, [64])
        self.network_local.build(input_shape=(None,self.state_size))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3, clipvalue=1.)
        self.loss_fn = tf.keras.losses.mse


    def step(self, states, actions, rewards, next_states, dones):
        self.t_step += 1
        loss = self.learn(states.reshape(1,-1), 
                          np.array(actions).reshape(1,-1), 
                          np.array(rewards).reshape(1,-1), 
                          next_states.reshape(1,-1), 
                          np.array(dones).reshape(1,-1))
        self.losses.append(loss)
            
    def get_eps(self, i, eps_start = 1., eps_end = 0.001, eps_decay = 0.9999):
        '''
        Fill in your codes to calculate epsilon here
        '''
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
                    
    def act(self, state):
        '''
        Fill in codes to perform epsilon greedy action
        '''
        eps = self.get_eps(self.episodes)
        action_values = self.network_local(tf.cast(state[None,:],dtype=tf.float32))
        #epsilon greedy
        if random.random() > eps:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, states, actions, rewards, next_states, dones, gamma=0.99):
        '''
        Fill in the codes to update the network
        '''
        #get q-values for next states
        q_targets_next = tf.reduce_max(self.network_local(next_states),1)[:,tf.newaxis]
        #bellman's equation; check the shape of q_targets
        q_targets = rewards + (gamma * q_targets_next) * (1 - dones)
        q_targets = q_targets[:,0] #flatten q_targets

        with tf.GradientTape() as tape:
            #get q-values of all actions
            preds = self.network_local(states)
            
            #get index of actions used to calculate q_expected
            actions = actions[:,0] #flatten actions
            seq = tf.range(0, actions.shape[0])
            action_idxs = tf.transpose(tf.stack([seq, actions]))

            #get q values only at specific action indice
            #tensorflow has this weird way to gather where you need to put index as
            #[[0, action_idx1],[1, action_idx2],...,[n, action_idxn]]
            q_expected = tf.gather_nd(preds, action_idxs)
            
            #calculate loss
            loss = self.loss_fn(q_targets, q_expected)
            
            #get gradients
            gradients = tape.gradient(loss, self.network_local.trainable_weights)
        
        #apply gradients
        self.optimizer.apply_gradients(zip(gradients,self.network_local.trainable_weights))
        return loss
    
class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory,
        lr=1e-3, bs = 64, clip=1., hidden_sizes = [256,256],
        gamma=0.99, tau= 1e-3, update_interval = 5, update_times = 1,
        double = False, Architecture=DenseNetwork):
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.bs = bs
        self.gamma = gamma
        self.update_interval = update_interval
        self.update_times = update_times
        self.tau = tau
        self.losses = []
        self.clip = clip
        self.double = double
        self.Architecture = Architecture

        #networks
        self.network_local = Architecture(output_size=self.action_size, hidden_sizes = self.hidden_sizes, 
            input_size=state_size)
        self.network_local.build(input_shape=(None,self.state_size))
        self.network_target = Architecture(output_size=self.action_size,hidden_sizes=self.hidden_sizes,
            input_size=state_size)
        self.network_target.build(input_shape=(None,self.state_size))
        self.network_target.set_weights(self.network_local.get_weights())
        
        #optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr, clipvalue=self.clip)
        
        # replay memory
        self.memory = replay_memory
        # count time steps and episodes
        self.t_step = 0
        self.episodes = 0
    
    def step(self, state, action, reward, next_state, done):
        #add transition to replay memory
        self.memory.add(state, action, reward, next_state, done, #for all others
            self.episodes, self.t_step) #for nstep memory

        #increment episodes if done
        if done: 
            self.t_step = 0
            self.episodes+=1
        
        #update target network
        self.soft_update()
        #self.hard_update()
        
        # learn every self.t_step
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            if len(self.memory) > self.bs:
                for _ in range(self.update_times):
                    if isinstance(self.memory,PrioritizedMemory):
                        transitions = self.memory.sample(self.bs, self.get_beta(self.episodes))
                    else:
                        transitions = self.memory.sample(self.bs)
                    loss = self.learn(transitions)
                    self.losses.append(loss)

    def act(self, state):
        eps = self.get_eps(self.episodes)
        action_values = self.network_local(tf.cast(state[None,:],dtype=tf.float32))
        #noisy exploration
        if isinstance(self.network_local, NoisyDenseNetwork):
            return np.argmax(action_values.numpy())
        #epsilon greedy exploration
        if random.random() > eps:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def act_values(self, state):
        eps = self.get_eps(self.episodes)
        action_values = self.network_local(tf.cast(state[None,:],dtype=tf.float32))
        #epsilon greedy
        if random.random() > eps:
            return np.argsort(action_values.numpy().squeeze())
        else:
            return 'random'

    #we use tf.function decorator for training process to switch from eager to static graph
    #NOT compatible with PrioritizedMemory yet
    #with decorator
    #1.64 ms ± 45.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    #without decorator
    #2.93 ms ± 442 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     @tf.function
    def learn(self, transitions):
        #unpack
        if isinstance(self.memory, PrioritizedMemory):
            states, actions, rewards, next_states, dones, idx, sampling_weights = transitions
        else:
            states, actions, rewards, next_states, dones = transitions

        if self.double:
            max_actions_next = tf.cast(tf.argmax(self.network_local(next_states),1), dtype=tf.int32)
            seq = tf.range(0, actions.shape[0])
            max_action_idxs = tf.transpose(tf.stack([seq, max_actions_next]))
            q_targets_next = tf.gather_nd(self.network_target(next_states), max_action_idxs)[:,tf.newaxis]
        else:
            q_targets_next = tf.reduce_max(self.network_target(next_states),1)[:,tf.newaxis]

        if isinstance(self.memory, NStepMemory):
            #these are not real rewards but discounted returns and we use Q for the n-th step
            q_targets = rewards + (self.gamma**self.memory.n * q_targets_next) * (1 - dones)
        else:
            q_targets = rewards + (self.gamma * q_targets_next) * (1 - dones)
        q_targets = q_targets[:,0] #flatten q_targets

        with tf.GradientTape() as tape:
            #get predictions of all actions
            preds = self.network_local(states)
            
            #get index of actions used to calculate q_expected
            actions = actions[:,0] #flatten actions
            seq = tf.range(0, actions.shape[0])
            action_idxs = tf.transpose(tf.stack([seq, actions]))

            #get q values only at specific action indice
            #tensorflow has this weird way to gather where you need to put index as
            #[[0, action_idx1],[1, action_idx2],...,[n, action_idxn]]
            q_expected = tf.gather_nd(preds, action_idxs)
            
            #calculate loss
            if isinstance(self.memory, PrioritizedMemory):
                loss = self.wis_loss(q_targets, q_expected, idx, sampling_weights)
            else:
                loss = self.vanilla_loss(q_targets, q_expected)
            
            #get gradients
            gradients = tape.gradient(loss, self.network_local.trainable_weights)
        
        #apply gradients
        self.optimizer.apply_gradients(zip(gradients,self.network_local.trainable_weights))

        #reset noise
        if isinstance(self.network_local,NoisyDenseNetwork): self.network_local.reset_noise()

        return loss

    def vanilla_loss(self, q_targets, q_expected):
        return tf.keras.losses.mse(q_targets, q_expected)
    
    def wis_loss(self, q_targets, q_expected, idx, sampling_weights, small_e=1e-5):
        losses = tf.pow(q_expected-q_targets, 2) * sampling_weights
        self.memory.update_priority(idx,losses+small_e)
        loss = tf.reduce_mean(losses)
        return loss
    
    def hard_update(self):
        if self.t_step % 1/self.tau==0:
            self.network_target.set_weights(self.network_local.get_weights())

    def soft_update(self):
        weights_local = np.array(self.network_local.get_weights())
        weights_target = np.array(self.network_target.get_weights())
        self.network_target.set_weights(self.tau * weights_local + (1 - self.tau) * weights_target)
        
    def get_eps(self, i, eps_start = 1., eps_end = 0.001, eps_decay = 0.9):
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
    
    def get_beta(self, i, beta_start = 0.4, beta_end = 1, beta_growth = 1.2):
        beta = min(beta_start * (beta_growth ** i), beta_end)
        return(beta)