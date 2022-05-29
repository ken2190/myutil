#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time
import types

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources

import seaborn as sns

sns.set()


# In[2]:


def get_imports():
    """function get_imports.
    Doc::
            
            Args:
            Returns:
                
    """
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            name = val.__name__.split(".")[0]
        elif isinstance(val, type):
            name = val.__module__.split(".")[0]
        poorly_named_packages = {"PIL": "Pillow", "sklearn": "scikit-learn"}
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]
        yield name


imports = list(set(get_imports()))
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name != "pip":
        requirements.append((m.project_name, m.version))

for r in requirements:
    print("{}=={}".format(*r))


# In[3]:


def get_state(data, t, n):
    """function get_state.
    Doc::
            
            Args:
                data:   
                t:   
                n:   
            Returns:
                
    """
    d = t - n + 1
    block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]
    res = []
    for i in range(n - 1):
        res.append(block[i + 1] - block[i])
    return np.array([res])


# TSLA Time Period: **Mar 23, 2018 - Mar 23, 2019**

# In[6]:


df = pd.read_csv("../dataset/TSLA.csv")
df.head()


# In[7]:


close = df.Close.values.tolist()
window_size = 30
skip = 1
l = len(close) - 1


# In[8]:


class Deep_Evolution_Strategy:

    inputs = None

    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        """ Deep_Evolution_Strategy:__init__.
        Doc::
                
                    Args:
                        weights:     
                        reward_function:     
                        population_size:     
                        sigma:     
                        learning_rate:     
                    Returns:
                       
        """
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        """ Deep_Evolution_Strategy:_get_weight_from_population.
        Doc::
                
                    Args:
                        weights:     
                        population:     
                    Returns:
                       
        """
        weights_population = []
        for index, i in enumerate(population):
            jittered = self.sigma * i
            weights_population.append(weights[index] + jittered)
        return weights_population

    def get_weights(self):
        """ Model:get_weights.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        """ Deep_Evolution_Strategy:get_weights
        Args:
        Returns:
           
        """
        return self.weights

    def train(self, epoch=100, print_every=1):
        """ Deep_Evolution_Strategy:train.
        Doc::
                
                    Args:
                        epoch:     
                        print_every:     
                    Returns:
                       
        """
        lasttime = time.time()
        for i in range(epoch):
            population = []
            rewards = np.zeros(self.population_size)
            for k in range(self.population_size):
                x = []
                for w in self.weights:
                    x.append(np.random.randn(*w.shape))
                population.append(x)
            for k in range(self.population_size):
                weights_population = self._get_weight_from_population(self.weights, population[k])
                rewards[k] = self.reward_function(weights_population)
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = (
                    w
                    + self.learning_rate
                    / (self.population_size * self.sigma)
                    * np.dot(A.T, rewards).T
                )
            if (i + 1) % print_every == 0:
                print("iter %d. reward: %f" % (i + 1, self.reward_function(self.weights)))
        print("time taken to train:", time.time() - lasttime, "seconds")


class Model:
    def __init__(self, input_size, layer_size, output_size):
        """ Model:__init__.
        Doc::
                
                    Args:
                        input_size:     
                        layer_size:     
                        output_size:     
                    Returns:
                       
        """
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(layer_size, 1),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        """ Model:predict.
        Doc::
                
                    Args:
                        inputs:     
                    Returns:
                       
        """
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        decision = np.dot(feed, self.weights[1])
        buy = np.dot(feed, self.weights[2])
        return decision, buy

    def get_weights(self):
        """ Model:get_weights.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        return self.weights

    def set_weights(self, weights):
        """ Model:set_weights.
        Doc::
                
                    Args:
                        weights:     
                    Returns:
                       
        """
        self.weights = weights


# In[9]:


class Agent:

    POPULATION_SIZE = 15
    SIGMA = 0.1
    LEARNING_RATE = 0.03

    def __init__(self, model, money, max_buy, max_sell):
        """ Agent:__init__.
        Doc::
                
                    Args:
                        model:     
                        money:     
                        max_buy:     
                        max_sell:     
                    Returns:
                       
        """
        self.model = model
        self.initial_money = money
        self.max_buy = max_buy
        self.max_sell = max_sell
        self.es = Deep_Evolution_Strategy(
            self.model.get_weights(),
            self.get_reward,
            self.POPULATION_SIZE,
            self.SIGMA,
            self.LEARNING_RATE,
        )

    def act(self, sequence):
        """ Agent:act.
        Doc::
                
                    Args:
                        sequence:     
                    Returns:
                       
        """
        decision, buy = self.model.predict(np.array(sequence))
        return np.argmax(decision[0]), int(buy[0])

    def get_reward(self, weights):
        """ Agent:get_reward.
        Doc::
                
                    Args:
                        weights:     
                    Returns:
                       
        """
        initial_money = self.initial_money
        starting_money = initial_money
        self.model.weights = weights
        state = get_state(close, 0, window_size + 1)
        inventory = []
        quantity = 0
        for t in range(0, l, skip):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, window_size + 1)
            if action == 1 and initial_money >= close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * close[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
            elif action == 2 and len(inventory) > 0:
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                quantity -= sell_units
                total_sell = sell_units * close[t]
                initial_money += total_sell

            state = next_state
        return ((initial_money - starting_money) / starting_money) * 100

    def fit(self, iterations, checkpoint):
        """ Agent:fit.
        Doc::
                
                    Args:
                        iterations:     
                        checkpoint:     
                    Returns:
                       
        """
        self.es.train(iterations, print_every=checkpoint)

    def buy(self):
        """ Agent:buy.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        initial_money = self.initial_money
        state = get_state(close, 0, window_size + 1)
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        quantity = 0
        for t in range(0, l, skip):
            action, buy = self.act(state)
            next_state = get_state(close, t + 1, window_size + 1)
            if action == 1 and initial_money >= close[t]:
                if buy < 0:
                    buy = 1
                if buy > self.max_buy:
                    buy_units = self.max_buy
                else:
                    buy_units = buy
                total_buy = buy_units * close[t]
                initial_money -= total_buy
                inventory.append(total_buy)
                quantity += buy_units
                states_buy.append(t)
                print(
                    "day %d: buy %d units at price %f, total balance %f"
                    % (t, buy_units, total_buy, initial_money)
                )
            elif action == 2 and len(inventory) > 0:
                bought_price = inventory.pop(0)
                if quantity > self.max_sell:
                    sell_units = self.max_sell
                else:
                    sell_units = quantity
                if sell_units < 1:
                    continue
                quantity -= sell_units
                total_sell = sell_units * close[t]
                initial_money += total_sell
                states_sell.append(t)
                try:
                    invest = ((total_sell - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    "day %d, sell %d units at price %f, investment %f %%, total balance %f,"
                    % (t, sell_units, total_sell, invest, initial_money)
                )
            state = next_state

        invest = ((initial_money - starting_money) / starting_money) * 100
        print(
            "\ntotal gained %f, total investment %f %%" % (initial_money - starting_money, invest)
        )
        plt.figure(figsize=(20, 10))
        plt.plot(close, label="true close", c="g")
        plt.plot(close, "X", label="predict buy", markevery=states_buy, c="b")
        plt.plot(close, "o", label="predict sell", markevery=states_sell, c="r")
        plt.legend()
        plt.show()


# In[10]:


model = Model(window_size, 500, 3)
agent = Agent(model, 10000, 5, 5)
agent.fit(500, 10)


# In[11]:


agent.buy()


# In[ ]:
