from random import randint
import numpy as np
import tensorflow as tf


np.random.seed(1)
tf.random.set_seed(1)
def policy():
    a_set = set()
    while True:
        a_set.add(randint(250, 2000))
        if len(a_set)==100:
            break
    lst = sorted(list(a_set))

class PolicyGradientAgent:
    def __init__(self, state_size, action_size, gym_agent, learning_rate=0.01, gamma=0.95):

        self.gym_agent = gym_agent
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.observations = []
        self.actions = []
        self.rewards = []

        self._build_model()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=100000)

    def _build_model(self):
        tf.reset_default_graph()
        with tf.name_scope('inputs'):
            # placeholders
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.state_size], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="action_indexes")
            self.tf_rew = tf.placeholder(
                tf.float32, [None, ], name="action_rewards")
        # layer1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=self.state_size*2,  
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1'
        )
        # layer2
        layer = tf.layers.dense(
            inputs=layer,
            units=self.state_size*2,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer2'
        )
        # layer3
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.action_size,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(
                mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer3'
        )

        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            # maximizing total reward (log_p * R) is equal to minimizing
            # -(log_p * R), tensorflow has only have minimizing(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=all_act, labels=self.tf_acts)   
                # this is negative log of the chosen action
            # reward guided loss
            loss = tf.reduce_mean(neg_log_prob * self.tf_rew)
            self.loss = loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(
                self.learning_rate).minimize(loss)
                #This method simply combines calls compute_gradients() and
                #apply_gradients() 

    def act(self, observation):
        '''
        Choose actions with respect to their probabilities
        '''
        # runs one "step" of TensorFlow computation
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
                                     self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def remember(self, state, action, reward):
        '''
        Add state,action,reward to the memory
        '''
        self.observations.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self):
        '''
        Training of the PG agent
        '''

        discounted_normalized_rewards = self._discount_and_normalize_rewards()

        _, loss = self.sess.run((self.train_op, self.loss),
                                feed_dict={
            # shape=[None, n_obs]
            self.tf_obs: np.vstack(self.observations),
            # shape=[None, ]
            self.tf_acts: np.array(self.actions),
            # shape=[None, ]
            self.tf_rew: discounted_normalized_rewards,
        })
        # empty the memory after gradient update
        self.observations = []
        self.actions = []
        self.rewards = []

        return discounted_normalized_rewards, loss

    def _discount_and_normalize_rewards(self):
        '''
        discount and normalize the reward of the episode
        '''
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float64)
        running_add = 0
        for t in reversed(range(0, len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add

        # normalize episode rewards
        discounted_rewards -= np.mean(discounted_rewards, dtype=np.float64)
        discounted_rewards /= (np.std(discounted_rewards,
                                      dtype=np.float64)+1e-6)
        return discounted_rewards

    def load(self, path):
        self.saver.restore(self.sess, path)

    def save(self, path):
        self.saver.save(self.sess, path)
