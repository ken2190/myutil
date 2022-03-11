#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cPickle

import numpy as np
import tensorflow as tf

# In[2]:


def load_pkl(path):
    """function load_pkl
    Args:
        path:   
    Returns:
        
    """
    with open(path) as f:
        obj = cPickle.load(f)
        return obj


class TLSTM(object):
    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        """ TLSTM:init_weights
        Args:
            input_dim:     
            output_dim:     
            name:     
            std:     
            reg:     
        Returns:
           
        """
        return tf.get_variable(
            name,
            shape=[input_dim, output_dim],
            initializer=tf.random_normal_initializer(0.0, std),
            regularizer=reg,
        )

    def init_bias(self, output_dim, name):
        """ TLSTM:init_bias
        Args:
            output_dim:     
            name:     
        Returns:
           
        """
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        """ TLSTM:no_init_weights
        Args:
            input_dim:     
            output_dim:     
            name:     
        Returns:
           
        """
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        """ TLSTM:no_init_bias
        Args:
            output_dim:     
            name:     
        Returns:
           
        """
        return tf.get_variable(name, shape=[output_dim])

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim):
        """ TLSTM:__init__
        Args:
            input_dim:     
            output_dim:     
            hidden_dim:     
            fc_dim:     
        Returns:
           
        """

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input = tf.placeholder("float", shape=[None, None, self.input_dim])
        self.labels = tf.placeholder("float", shape=[None, output_dim])
        self.time = tf.placeholder("float", shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        self.Wi = self.init_weights(
            self.input_dim, self.hidden_dim, name="Input_Hidden_weight", reg=None
        )
        self.Ui = self.init_weights(
            self.hidden_dim, self.hidden_dim, name="Input_State_weight", reg=None
        )
        self.bi = self.init_bias(self.hidden_dim, name="Input_Hidden_bias")
        self.Wf = self.init_weights(
            self.input_dim, self.hidden_dim, name="Forget_Hidden_weight", reg=None
        )
        self.Uf = self.init_weights(
            self.hidden_dim, self.hidden_dim, name="Forget_State_weight", reg=None
        )
        self.bf = self.init_bias(self.hidden_dim, name="Forget_Hidden_bias")
        self.Wog = self.init_weights(
            self.input_dim, self.hidden_dim, name="Output_Hidden_weight", reg=None
        )
        self.Uog = self.init_weights(
            self.hidden_dim, self.hidden_dim, name="Output_State_weight", reg=None
        )
        self.bog = self.init_bias(self.hidden_dim, name="Output_Hidden_bias")
        self.Wc = self.init_weights(
            self.input_dim, self.hidden_dim, name="Cell_Hidden_weight", reg=None
        )
        self.Uc = self.init_weights(
            self.hidden_dim, self.hidden_dim, name="Cell_State_weight", reg=None
        )
        self.bc = self.init_bias(self.hidden_dim, name="Cell_Hidden_bias")
        self.W_decomp = self.init_weights(
            self.hidden_dim, self.hidden_dim, name="Decomposition_Hidden_weight", reg=None
        )
        self.b_decomp = self.init_bias(self.hidden_dim, name="Decomposition_Hidden_bias_enc")
        self.Wo = self.init_weights(self.hidden_dim, fc_dim, name="Fc_Layer_weight", reg=None)
        self.bo = self.init_bias(fc_dim, name="Fc_Layer_bias")
        self.W_softmax = self.init_weights(fc_dim, output_dim, name="Output_Layer_weight", reg=None)
        self.b_softmax = self.init_bias(output_dim, name="Output_Layer_bias")

    def TLSTM_Unit(self, prev_hidden_memory, concat_input):
        """ TLSTM:TLSTM_Unit
        Args:
            prev_hidden_memory:     
            concat_input:     
        Returns:
           
        """
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)

        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # Dealing with time irregularity

        # Map elapse time in days or months
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)
        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)
        # Current Memory cell
        Ct = f * prev_cell + i * C
        # Current Hidden state
        current_hidden_state = o * tf.nn.tanh(Ct)
        return tf.stack([current_hidden_state, Ct])

    def get_states(self):  # Returns all hidden states for the samples in a batch
        """ TLSTM:get_states
        Args:
        Returns:
           
        """

    def get_output(self, state):
        """ TLSTM:get_output
        Args:
            state:     
        Returns:
           
        """
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    def get_outputs(self):
        """ TLSTM:get_outputs
        Args:
        Returns:
           
        """
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]
        return output

    def get_cost_acc(self):
        """ TLSTM:get_cost_acc
        Args:
        Returns:
           
        """
        logits = self.get_outputs()
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        )
        y_pred = tf.argmax(logits, 1)
        y = tf.argmax(self.labels, 1)
        return cross_entropy, y_pred, y, logits, self.labels

    def map_elapse_time(self, t):
        """ TLSTM:map_elapse_time
        Args:
            t:     
        Returns:
           
        """
        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)
        T = tf.div(c1, tf.log(t + c2), name="Log_elapse_time")
        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)
        T = tf.matmul(T, Ones)
        return T


# In[3]:


train_times = load_pkl("Split0/elapsed_train.pkl")
train_X = load_pkl("Split0/data_train.pkl")
train_Y = load_pkl("Split0/label_train.pkl")


# In[4]:


tf.reset_default_graph()
sess = tf.InteractiveSession()
lstm = TLSTM(train_X[0].shape[2], train_Y[0].shape[1], 256, 128)
cross_entropy, y_pred, y, logits, labels = lstm.get_cost_acc()
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())


# In[8]:


for i in range(10):
    total_lost, total_acc = 0, 0
    for k in range(len(train_X)):
        out, lost, _ = sess.run(
            [logits, cross_entropy, optimizer],
            feed_dict={
                lstm.input: train_X[k],
                lstm.labels: train_Y[k],
                lstm.time: train_times[k][:, 0, :],
                lstm.keep_prob: 0.5,
            },
        )
        total_lost += lost
        total_acc += np.mean(np.argmax(out, axis=1) == np.argmax(train_Y[k], axis=1))
    total_lost /= len(train_X)
    total_acc /= len(train_X)
    print("epoch %d, avg loss %f, avg acc %f" % (i + 1, total_lost, total_acc))


# In[ ]:
