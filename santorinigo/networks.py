import tensorflow as tf
from tensorflow.keras import layers, models

class DenseNetwork(models.Model):
    def __init__(self, output_size, hidden_sizes, **args):
        super(DenseNetwork, self).__init__()
        self.body = [layers.Dense(i, activation='relu') for i in hidden_sizes]
        self.head = layers.Dense(output_size)
    def call(self, x, training=True):
        for l in self.body: x = l(x) 
        return self.head(x)

class DistDenseNetwork(models.Model):
    def __init__(self, output_size, hidden_sizes, atom_size=51, **args):
        super(DistDenseNetwork, self).__init__()
        self.body = [layers.Dense(i, activation='relu') for i in hidden_sizes]
        self.head = layers.Dense(output_size*atom_size)
        self.output_size = output_size
        self.atom_size = atom_size
    def call(self, x, training=True):
        for l in self.body: x = l(x) 
        x = self.head(x)
        x = tf.reshape(x, shape=[-1,self.output_size,self.atom_size])
        x = tf.nn.softmax(x, dim=-1)
        return x

class DuelingNetwork(models.Model):
    def __init__(self, output_size, hidden_sizes, **args):
        super(DuelingNetwork, self).__init__()
        self.feature = [layers.Dense(i, activation='relu') for i in hidden_sizes[:-1]]
        self.value = [layers.Dense(hidden_sizes[-1], activation='relu'), layers.Dense(1)]
        self.advantage = [layers.Dense(hidden_sizes[-1], activation='relu'), layers.Dense(output_size)]
    def call(self, x, training=True):
        feat = x
        for l in self.feature: feat = l(feat)
        adv = feat
        for a in self.advantage: adv = a(adv)
        val = feat
        for v in self.value: val = v(val)
        return adv + val - tf.reduce_mean(adv)

class NoisyDenseNetwork(models.Model):
    def __init__(self, output_size, hidden_sizes, input_size, **args):
        super(NoisyDenseNetwork, self).__init__()
        # self.body = [NoisyDense(hidden_sizes[0], input_size, tf.nn.relu)]
        # for i in range(1, len(hidden_sizes)):
        #     self.body.append(NoisyDense(hidden_sizes[i],hidden_sizes[i-1], tf.nn.relu))
        self.body = [layers.Dense(i, activation='relu') for i in hidden_sizes]
        self.head = NoisyDense(output_size, hidden_sizes[-1])
        self.reset_noise()
    def call(self, x, training=True):
        for l in self.body: x = l(x) 
        return self.head(x, training)
    def reset_noise(self):
        # for l in self.body: l.reset_noise()
        self.head.reset_noise()

class NoisyDuelingNetwork(NoisyDenseNetwork,DuelingNetwork):
    def __init__(self, output_size, hidden_sizes, input_size, **args):
        super(DuelingNetwork, self).__init__()
        self.feature = [layers.Dense(i, activation='relu') for i in hidden_sizes]
        self.value = [NoisyDense(1, hidden_sizes[-1])]
        # self.value = [layers.Dense(1)]
        # self.advantage = [NoisyDense(output_size, hidden_sizes[-1])]
        self.advantage = [layers.Dense(output_size)]
    def call(self, x, training=True):
        feat = x
        for l in self.feature: feat = l(feat)
        adv = feat
        for a in self.advantage: adv = a(adv)
        val = feat
        for v in self.value: val = v(val)
        return adv + val - tf.reduce_mean(adv)
    def reset_noise(self):
        for l in self.value: l.reset_noise()
        # for l in self.advantage: l.reset_noise()


class NoisyDense(layers.Layer):
    def __init__(self, output_size, input_size, activation=None):
        super(NoisyDense, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.activation = activation

        mu_range = 1 / self.input_size**(1/2)
        self.w_mu = tf.Variable(initial_value=tf.random.uniform(shape=(self.input_size,self.output_size),
        	minval=-mu_range,maxval=mu_range), trainable=True)
        self.b_mu = tf.Variable(initial_value=tf.random.uniform(shape=(self.output_size,),minval=-mu_range,
        	maxval=mu_range), trainable=True)

        sigma_range = 0.1 / self.input_size**(1/2)
        self.w_sigma = tf.Variable(initial_value=tf.random.uniform(shape=(self.input_size,self.output_size),
        	minval=-sigma_range,maxval=sigma_range), trainable=True)
        self.b_sigma = tf.Variable(initial_value=tf.random.uniform(shape=(self.output_size,),
        	minval=-sigma_range,maxval=sigma_range), trainable=True)

        self.reset_noise()

    def reset_noise(self):
        e_in = self.factorize_noise(self.input_size)
        e_out = self.factorize_noise(self.output_size)
        self.w_epsilon = tf.linalg.tensordot(e_in,e_out,axes=0)
        self.b_epsilon = e_out

    def factorize_noise(self, size):
        x = tf.random.normal((size,))
        x = tf.sign(x) * tf.sqrt(tf.abs(x))
        return x
        
    def call(self, x, training=True):
        if training:
            w = self.w_mu + self.w_sigma * self.w_epsilon
            b = self.b_mu + self.b_sigma * self.b_epsilon
        else:
            w = self.w_mu
            b = self.b_mu
        res = x@w + b
        if self.activation is not None: res = self.activation(res)
        return res