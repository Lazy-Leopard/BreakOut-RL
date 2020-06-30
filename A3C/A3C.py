## Here a basic description of updates and function names will be added
##class name A3C_Atari contains all submodules

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k
import matplotlib.pyplot as plt

# More libraries as per need may be added with time

class A3C_Atari:

    def __init__(self, game_name, lr, n_workers, n_actions, action_space, NMaxEp, frequency, gamma):
        self.lr = lr
        self.game_name = game_name
        self.n_actions = int(n_actions)
        self.action_space = action_space
        self.n_workers = int(n_workers)
        self.model = self.Model()
        self.NMaxEp = NMaxEp
        self.frequency = frequency
        self.gamma = gamma

    def Model(self):
        input_ = k.layers.Input(shape=(80, 80, 4))
        conv1 = k.layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4),
                                kernel_initializer=k.initializers.glorot_normal(),
                                activation=k.activations.relu, padding='valid')(input_)
        conv2 = k.layers.Conv2D(16, kernel_size=(4, 4), strides=(2, 2),
                                kernel_initializer=k.initializers.glorot_normal(),
                                activation=k.activations.relu, padding='valid')(conv1)

        #conv3 = k.layers.Conv2D(16, kernel_size=(4, 4), strides=(1, 1),
                              #  kernel_initializer=k.initializers.glorot_normal(),
                              #  activation=k.activations.relu, padding='valid')(conv2)

        dense1 = k.layers.Flatten()(conv2)
        dense2 = k.layers.Dense(256, activation=k.activations.relu,
                                kernel_initializer=k.initializers.glorot_normal(),
                                bias_initializer=k.initializers.glorot_normal())(dense1)
        actions = k.layers.Dense(self.n_actions, activation=k.activations.softmax,
                                 kernel_initializer=k.initializers.glorot_normal(),
                                 bias_initializer=k.initializers.glorot_normal())(dense2)
        value = k.layers.Dense(1, activation=None,
                               kernel_initializer=k.initializers.glorot_normal(),
                               bias_initializer=k.initializers.glorot_normal())(dense2)
        print(input_.shape)

        model = k.Model(inputs=input_, outputs=[actions, value])
        model.compile(optimizer=k.optimizers.Adam(self.lr), loss=[self.custom_loss(),k.losses.mse],
                      loss_weights=[1, 0.5])

        model.load_weights('model_breakout_6.h5')

        model.summary()
        return model

    
