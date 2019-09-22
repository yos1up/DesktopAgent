import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from collections import Counter, defaultdict, deque
import os, sys, glob, copy, json, time, pickle

from chainer import Chain, ChainList, cuda, gradient_check, Function, Link, \
    optimizers, serializers, utils, Variable, dataset, datasets, using_config, training, iterators
from chainer.training import extensions
from chainer import functions as F
from chainer import links as L
import chainer

class Agent:
    """
    General Q-Learning Agent for an environment
    whose states are given by images and whose actions are discrete
    """
    def __init__(self, num_action):
        """
        num_action (int) : 可能な行動の個数．
        """
        self.sars_list = deque()
        self.num_action = num_action
        self.gamma = 0.95
        self.epsilon = 0.1
        self.iteration = 0
        self.batch_size = 1
        self.learn_interval = 1 # TODO: 目標関数を一定時間固定する．
        self.Q = QFunc(num_action)
        self.opt = optimizers.MomentumSGD().setup(self.Q)
        self.opt.add_hook(chainer.optimizer.GradientClipping(5.0))
        
    def observe(self, sars):
        """
        sars: [s0, a, r, s1]
            s0, s1: 3-dim image[c,h,w]，  a : int, r : float  (32bit-ize is not needed)
        """
        self.sars_list.append(sars)
        if len(self.sars_list) >= 10000:
            self.sars_list.popleft()            
        # 容量がしんどそう　　VAEなどで次元削減しておけばあるいは．
        # 驚きの小さかったイベントを優先的に忘れた方が良いのでは？
        self.iteration += 1
        if self.iteration % self.learn_interval == 0: self._learn()
    def _learn(self):
        idx = np.random.choice(len(self.sars_list), self.batch_size, replace=False)
        s0, a, r, s1 = zip(*[self.sars_list[i] for i in idx])
        s0, r, s1 = np.array(s0, dtype=np.float32), np.array(r, dtype=np.float32), np.array(s1, dtype=np.float32)
        a = np.array(a, dtype=np.int32)
        # calculate loss of Q-learning
        with chainer.no_backprop_mode():
            rhs = r + self.gamma * F.max(self.Q(s1), axis=1)
        lhs = F.select_item(self.Q(s0), a)
        loss = F.mean((rhs - lhs) ** 2) # scalar
        self.Q.cleargrads()
        loss.backward()
        self.opt.update()
    def determine(self, s):
        """
        s : single array => a : single int
        """
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q(Variable(np.array([s],dtype=np.float32))).data.flatten())
        else:
            return np.random.randint(self.num_action)
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f).__dict__)
        
        
class QFunc(Chain):
    def __init__(self, num_action):
        super().__init__()
        with self.init_scope():
            self.add_link('b1', L.BatchNormalization(3))
            self.add_link('c1', L.Convolution2D(3, 6, ksize=5, stride=4))
            self.add_link('b2', L.BatchNormalization(6))
            self.add_link('c2', L.Convolution2D(6, 12, ksize=5, stride=4))
            self.add_link('b3', L.BatchNormalization(12))
            self.add_link('c3', L.Convolution2D(12, 24, ksize=5, stride=4))
            self.add_link('l1', L.Linear(None, 64))
            self.add_link('l2', L.Linear(64, num_action))
    def __call__(self, x):
        '''
        x (Variable) => (Variable)
        '''
        assert x.ndim == 4
        assert x.shape[1] == 3
        h = x
        h = self.b1(h)
        h = F.relu(self.c1(h))
        h = self.b2(h)
        h = F.relu(self.c2(h))
        h = self.b3(h)
        h = F.relu(self.c3(h))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h
