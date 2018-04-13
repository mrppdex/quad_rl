import numpy as np
import tensorflow as tf
import random
from collections import deque, namedtuple


# standard replay buffer
class ReplayBuffer:
    def __init__(self, size):
        self.replay_memory = deque(maxlen=size)

    def add_to_memory(self, experience):
        self.replay_memory.append(experience)

    def sample_from_memory(self, minibatch_size):
        return random.sample(self.replay_memory, minibatch_size)

    def len(self):
        return len(self.replay_memory)
        
class OUNoise:
    """Ornstein-Uhlenback process."""
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
class Actor:
    def __init__(self, state_input, action_range, action_dim, scope):
    
        self.training = scope == "local"
        self.dropout_rate_1 = 0.2
        self.dropout_rate_2 = 0.5
        self.dropout_rate_3 = 0.5
        self.net_size = 64
        self.state_input = state_input
        self.action_range = action_range
        self.action_dim = action_dim

        with tf.variable_scope("actor_"+scope):
            self.out = self.build_model()
    
    def build_model(self):
        hidden = tf.layers.dense(self.state_input, self.net_size, activation = tf.nn.relu, name = 'dense')
        hidden_dropout = tf.layers.dropout(hidden, rate=self.dropout_rate_1, training=self.training)
        hidden_2 = tf.layers.dense(hidden_dropout, self.net_size, activation = tf.nn.relu, name = 'dense_1')
        hidden_dropout_2 = tf.layers.dropout(hidden_2, rate=self.dropout_rate_2, training=self.training)
        hidden_3 = tf.layers.dense(hidden_dropout_2, self.net_size, activation = tf.nn.relu, name = 'dense_2')
        hidden_dropout_3 = tf.layers.dropout(hidden_3, rate=self.dropout_rate_3, training=self.training)
        actions_unscaled = tf.layers.dense(hidden_dropout_3, self.action_dim, name = 'dense_3')
        actions = tf.nn.sigmoid(actions_unscaled) * self.action_range
        return actions


class Critic:
    def __init__(self, a_in, s_in, scope, reuse=False):

        self.training = scope == "local"
        self.dropout_rate_1 = 0.2
        self.dropout_rate_2 = 0.2
        self.dropout_rate_3 = 0.2
        self.net_size = 32

        self.action_in = a_in
        self.state_in = s_in

        with tf.variable_scope("critic_"+scope):
            self.q = self.build_model(self.state_in, self.action_in, reuse=reuse)

    def build_model(self, s_in, a_in, reuse):
        inputs = tf.concat([s_in, a_in], axis=1)
        hidden = tf.layers.dense(inputs, self.net_size, activation = tf.nn.relu, name = 'dense', reuse = reuse)
        hidden_dropout = tf.layers.dropout(hidden, rate=self.dropout_rate_1, training=self.training)
        hidden_2 = tf.layers.dense(hidden_dropout, self.net_size, activation = tf.nn.relu, name = 'dense_1', reuse = reuse)
        hidden_dropout_2 = tf.layers.dropout(hidden_2, rate=self.dropout_rate_2, training=self.training)
        hidden_3 = tf.layers.dense(hidden_dropout_2, self.net_size, activation = tf.nn.relu, name = 'dense_2', reuse = reuse)
        hidden_dropout_3 = tf.layers.dropout(hidden_3, rate=self.dropout_rate_3, training=self.training)
        q_logits = tf.layers.dense(hidden_dropout_3, 1, name = 'dense_3', reuse = reuse)
        
        return q_logits

class Agent:
    def __init__(self, task):
         
        self.task = task
        
        tf.reset_default_graph()
        
        self.lr_actor = 1e-5 # learning rate for the actor
        self.lr_critic = 1e-4 # learning rate for the critic
        self.l2_reg_actor = 1e-7 # L2 regularization factor for the actor
        self.l2_reg_critic = 1e-7 # L2 regularization factor for the critic
         
        self.batch_size = 1024
        self.memory = ReplayBuffer(int(1e5))
        self.gamma = 0.99
        self.tau = 1e-2
        
        self.action_range = task.action_high - task.action_low
        
        self.action_dim = task.action_size
        self.state_dim = task.state_size
        
        self.noise = OUNoise(self.action_dim)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.action_dim])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_dim])
        self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        self.actions = Actor(self.state_ph, self.action_range, self.action_dim, "local").out
        self.target_actions = tf.stop_gradient(Actor(self.next_state_ph, self.action_range, self.action_dim, "target").out)

        self.q_det = Critic(self.action_ph, self.state_ph, "local", reuse=False).q
        self.q_inf = Critic(self.actions, self.state_ph, "local", reuse=True).q

        self.target_critic = tf.stop_gradient(Critic(self.target_actions, self.next_state_ph, "target").q)
        
        self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_local')
        self.slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_local')
        self.slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.update_targets_ops = []
        for i, slow_target_actor_var in enumerate(self.slow_target_actor_vars):
            self.update_slow_target_actor_op = slow_target_actor_var.assign(self.tau*self.actor_vars[i]+(1-self.tau)*slow_target_actor_var)
            self.update_targets_ops.append(self.update_slow_target_actor_op)

        for i, slow_target_var in enumerate(self.slow_target_critic_vars):
            self.update_slow_target_critic_op = slow_target_var.assign(self.tau*self.critic_vars[i]+(1.-self.tau)*slow_target_var)
            self.update_targets_ops.append(self.update_slow_target_critic_op)

        self.update_slow_targets_op = tf.group(*self.update_targets_ops, name='update_slow_targets')
        
        self.targets = self.reward_ph[None] + self.is_not_terminal_ph[None]*self.gamma*self.target_critic
        
        self.td_errors = self.targets - self.q_det

        # L2 regularization for critic's weights
        self.critic_loss = tf.reduce_mean(tf.square(self.td_errors))
        for var in self.critic_vars:
            if not 'bias' in var.name:
                self.critic_loss += self.l2_reg_critic * tf.nn.l2_loss(var)

        # critic optimizer
        self.critic_train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic_loss)

        # L2 regularization for actor's weights
        self.actor_loss = -1*tf.reduce_mean(self.q_inf)
        for var in self.actor_vars:
            if not 'bias' in var.name:
                self.actor_loss += self.l2_reg_actor * tf.nn.l2_loss(var)

        # actor optimizer
        # gradient of critic inferenced Q-value w.r.t. the actor's theta
        self.actor_train_op = tf.train.AdamOptimizer(self.lr_actor).minimize(self.actor_loss, var_list=self.actor_vars)

        # initialize session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.total_steps = 0
        
        self.reset_state()
        
    def reset_state(self):
        self.total_reward = 0
        self.steps_in_ep = 0
        return self.task.reset()

    
    def act(self, observation):
        
        # actor
        # input: state
        # output: action
        action_for_state, = self.sess.run(self.actions, feed_dict = {self.state_ph: observation[None]})

        # noise added for exploration
        action_for_state += self.noise.sample()
        
        # take step
        next_observation, reward, done = self.task.step(action_for_state)
        
        self.total_reward += reward
        
        # add an experience to the buffer
        self.memory.add_to_memory((observation, action_for_state, reward, next_observation, 0.0 if done else 1.0))

        # if enough experiences are stored, fetch a batch of them
        if self.memory.len() >= self.batch_size:

            minibatch = self.memory.sample_from_memory(self.batch_size)

            _, _ = self.sess.run([self.critic_train_op, self.actor_train_op], 
                feed_dict = {
                    self.state_ph: np.asarray([elem[0] for elem in minibatch]),
                    self.action_ph: np.asarray([elem[1] for elem in minibatch]),
                    self.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                    self.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                    self.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch])})

            # update slow actor and critic targets towards current actor and critic
            _ = self.sess.run(self.update_slow_targets_op)

        self.observation = next_observation
        self.total_steps += 1
        self.steps_in_ep += 1

        return action_for_state, next_observation, reward, done



