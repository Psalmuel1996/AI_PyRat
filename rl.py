""" Importations """
import numpy as np
import pickle 
import tensorflow as tf
import random


class NLinearModels(object):
    def __init__(self, num_states , num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        
        # The placeholders
        self._states = None
        self._q_s_a = None
        
        # The output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        
        # Setup the model
        self._define_model()
   
    
    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        
        # Create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 400, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 200, activation=tf.nn.relu)
        fc3 = tf.layers.dropout(
        fc2,
        rate = 0.85,
        noise_shape = None,
        seed = None,
        training = False,
        name = None
        )
        
        self._logits = tf.layers.dense(fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()


    def predict_one(self, sess, state):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._num_states)})

    
    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})


    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})




class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class ExperienceReplay(object):
    """
    During gameplay all the experiences < s, a, r, s’ > are stored in a replay memory. 
    In training, batches of randomly drawn experiences are used to generate the input and target for training.
    """
    def __init__(self, sess, model, max_memory = 500, discount = .8, max_eps = 1,min_eps = 0):
        """
        max_memory: the maximum number of experiences we want to store
        memory: a list of experiences
        discount: the discount factor for future experience
        """
        self.sess = sess
        self.model = model
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps = max_eps
        self.decay = 0.05
        self.memory = Memory(max_memory)
        self.discount = discount


    def remember(self, experience, game_over):
        # Save an experience to memory
        self.memory.add_sample([experience, game_over])
        

    def get_batch(self, model, batch_size = 10):
        
        # How many experiences do we have?
        len_memory = self.memory._max_memory
        
        # Calculate the number of actions that can possibly be taken in the game
        num_actions = 4
        
        # Dimensions of the game field
        env_dim = list(self.memory._samples[0][0][0].shape)
        env_dim[0] = min(len_memory, batch_size)
        
        # The expected outcome
        inputs = np.zeros(env_dim)
        Q = np.zeros((inputs.shape[0], num_actions))
        
        # We draw experiences to learn from randomly
        batch = self.memory.sample(self.model._batch_size)
        states = np.array([val[0][0] for val in batch])
        next_states = np.array([(np.zeros((1,env_dim[1],env_dim[2],env_dim[3]))
                             if val[1] == True else val[0][3]) for val in batch])
    
        
        # Predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states.reshape((batch_size,-1)), self.sess)
        # Predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(next_states.reshape((batch_size,-1)), self.sess)
        
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0]
            # Get the current q values for all actions in state
            current_q = q_s_a[i]
            # Update the q value for action
            if b[1] == True :
                # In this case, the game is completed after action, so there is no max Q(s',a')
                current_q[action] = reward
            else:
                current_q[action] = reward + self.discount * np.amax(q_s_a_d[i])
                
            inputs[i:i+1] = state
            Q[i] = current_q
        return inputs, Q


    def load(self):
        self.memory = pickle.load(open("save_rl/memory.pkl","rb"))
        
        
    def save(self):
        pickle.dump(self.memory,open("save_rl/memory.pkl","wb"))
