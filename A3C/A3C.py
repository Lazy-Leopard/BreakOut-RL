## Here a basic description of updates and function names will be added
##class name A3C_Atari contains all submodules

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as k
import matplotlib.pyplot as plt

# More libraries as per need may be added with time

tf.enable_eager_execution()
EPISODE,running_score,G_t_step = 59000,190,37500000

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

    def proccess_input(self, state):
        state = state[35:195:2, 0:160:2, :]
        state = cv2.resize(state, (80, 80), interpolation=cv2.INTER_CUBIC)
        state = 0.299 * state[:, :, 0] + 0.587 * state[:, :, 1] + 0.114 * state[:, :, 2]
        state = state / 255.0
        return state

    def custom_loss(self):
        def loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 0.00001, 0.99999)
            entropy = -tf.reduce_mean(tf.reduce_sum(y_pred * tf.log(y_pred),axis=0))
            policy_loss = -tf.reduce_mean(y_true * tf.log(y_pred))
            loss = policy_loss - 0.001 * entropy
            return loss

        return loss_fn


    def train(self):

        envs = [gym.make(self.game_name) for i in range(self.n_workers)]
        lock = threading.Lock()
        workers = [threading.Thread(target=self.run_thread, daemon=True, args=(envs[i], i, lock)) for i in
                   range(self.n_workers)]
        for worker in workers:
            worker.start()
            time.sleep(0.1)
        [worker.join() for worker in workers]

    def update(self, states, actions, rewards, done):

        #print(done)
        #print(rewards)
        Q = []
        if done[-1] == True:
            R = 0
        else:
            R = self.model.predict(np.expand_dims(states[-1], axis=0))[1][0][0]
        for r in reversed(rewards):
            R = self.gamma * R + r
            Q.append(R)

        Q = np.flip(Q)
       # print(Q)
        #Q -= np.mean(Q)
        #std = np.std(Q)
        #if std != 0 :
         #   Q /= std


        V = self.model.predict(np.asarray(states))[1]
        print(Q)

        advantage = np.zeros([len(rewards), self.n_actions])
        input = np.empty([len(rewards),80,80,4])

        for i in range(len(rewards)):
            advantage[i][actions[i]] = Q[i] - V[i][0]
            input[i] = states[i]

        self.model.fit(input, [advantage, Q], verbose=0)

    def run_thread(self, env, i, lock):
        global EPISODE, running_score, G_t_step
        while EPISODE < self.NMaxEp and running_score < 300:
            EPISODE += 1
            t,t_step, score , prevlives = 0, 0, 0, 5
            state = self.proccess_input(env.reset())
            state = np.stack([state] * 4, axis=2)
            state_list, reward_list, action_list, done_list, probability_list = [], [], [], [], []
            while prevlives > 0:
                t_step += 1
                G_t_step += 1
                lock.acquire()
                probability = np.clip(self.model.predict(np.expand_dims(state, axis=0))[0][0], 0.00001, 0.99999)
                #if i == 1:
                    
                     #env.render()
                time.sleep(0.01)
                lock.release()
                action = np.random.choice(self.action_space, 1, p=probability)
                next_state, reward, done, info = env.step(action[0])
                next_state = self.proccess_input(next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, axis=2), axis=2)
                if info['ale.lives']-prevlives == -1:
                    prevlives -= 1
                    reward = -0.25
                    done = True
                state_list.append(state)
                reward_list.append(reward)
                action_list.append(action_space.index(action))
                done_list.append(done)
                probability_list.append(probability)
                score += reward
                state = next_state
                if (t_step-t == self.frequency  or done == True):
                    state_list.append(state)
                    lock.acquire()
                    if len(action_list):
                        self.update(state_list, action_list, reward_list, done_list)
                    lock.release()
                    state_list, action_list, reward_list, done_list = [], [], [], []
                    t=t_step

            lock.acquire()
            running_score = 0.95 * running_score + 0.05 * score
            print('EPISODE : ', (EPISODE), 'G_tstep : ', (G_t_step), 'running score : ', (running_score),
                      'score :',
                      (score), 't_step : ', (t_step))
            if EPISODE % 50 == 0:
                self.model.save('model_breakout_7.h5')
                print('model saved to disc')
                print(random.sample(probability_list,10))
            lock.release()



env=gym.make('BreakoutDeterministic-v4').env
print(env.action_space.n)
print(env.unwrapped.get_action_meanings())
action_space = [1,2,3]
n_actions = len(action_space)
env.close()
agent = a3c_atari(lr=0.0001,game_name='BreakoutDeterministic-v4',gamma=0.99,n_actions=n_actions,action_space = action_space,n_workers=8,NMaxEp=200000,frequency=5)
agent.train()
