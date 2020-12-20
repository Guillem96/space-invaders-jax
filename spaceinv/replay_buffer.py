# -*- coding:utf-8 -*-

import random
from pathlib import Path
from collections import deque
from typing import NamedTuple, Union

from gym.wrappers import LazyFrames

import numpy as np


class Transition(NamedTuple):

    state: Union[np.ndarray, LazyFrames]
    action: Union[int, np.ndarray]
    reward: Union[float, np.ndarray]
    is_terminal: Union[bool, np.ndarray]
    next_state: Union[np.ndarray, LazyFrames]


class ReplayBuffer:

    def __init__(self, N: int) -> None:
        self.N = int(N)
        self.buffer = deque(maxlen=self.N)

    def experience(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, n: int) -> Transition:
        sample = random.sample(self.buffer, n)
        sample = Transition(*zip(*sample))

        states = np.stack([np.array(o) for o in sample.state])
        next_states = np.stack([np.array(o) for o in sample.next_state])
        rewards = np.array(sample.reward)
        actions = np.array(sample.action)
        is_terminal = np.array(sample.is_terminal)

        return Transition(state=states, 
                          next_state=next_states, 
                          reward=rewards,
                          action=actions,
                          is_terminal=is_terminal)

    def __len__(self) -> int:
        return len(self.buffer)

