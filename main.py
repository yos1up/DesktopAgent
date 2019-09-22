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

from agent import Agent
from trainer import DesktopTrainer

from input_utils import input_box_position

print('ゲーム画面全体の位置を教えてください．')
ltrb = input_box_position()
print('スコアが表示される位置を教えてください．')
ltrb_score = input_box_position()


# ltrb = (0,0,160,120)
# ltrb_score = (25,159,158,207)
agent = Agent(num_action=3)
dt = DesktopTrainer(agent, ltrb=ltrb, ltrb_score=ltrb_score,
        action=[[], ['left'], ['right']], wait=None, press_mode='hold')

dt.run()
