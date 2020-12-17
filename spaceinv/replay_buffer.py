# -*- coding:utf-8 -*-

from pathlib import Path
from typing import NamedTuple, Union, Tuple

import h5py
import numpy as np


class Transition(NamedTuple):

    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: Union[float, np.ndarray]
    is_terminal: Union[bool, np.ndarray]
    next_state: Union[np.ndarray, np.ndarray]


class ReplayBuffer:

    def __init__(self,
                 N: int,
                 state_shape: Tuple[int],
                 state_dtype: str = 'uint8', 
                 action_dtype: str = 'int8') -> None:

        self.N = int(N)
        self.i = 0
        self.states = np.empty((self.N, *state_shape), dtype=state_dtype)
        self.next_states = np.empty((self.N, *state_shape), dtype=state_dtype)
        self.actions = np.empty((self.N, 1), dtype=action_dtype)
        self.rewards = np.empty((self.N, 1), dtype='float16')
        self.is_terminal = np.empty((self.N, 1), dtype='bool')

    def experience(self, t: Transition) -> None:
        self.states[self.i % self.N] = t.state
        self.next_states[self.i % self.N] = t.next_state
        self.actions[self.i % self.N] = t.action
        self.rewards[self.i % self.N] = t.reward
        self.is_terminal[self.i % self.N] = t.is_terminal
        self.i += 1

    def sample(self, n: int) -> Transition:
        indices = np.random.randint(low=0, 
                                    high=(self.i % self.N) + 1, 
                                    size=(n,))

        return Transition(state=self.states[indices],
                          next_state=self.next_states[indices],
                          action=self.actions[indices],
                          reward=self.rewards[indices],
                          is_terminal=self.is_terminal[indices])

    def save(self, 
             f: Union[str, Path, h5py.File],
             append: bool = False) -> None:

        if append and not isinstance(f, h5py.File):
            raise ValueError('To append the ReplayBuffer to a file it must be'
                             ' already opened.')

        if isinstance(f, (str, Path)):
            f = h5py.File(str(f), 'w')

        f.create_dataset('rb/states', data=self.states)
        f.create_dataset('rb/next_states', data=self.next_states)
        f.create_dataset('rb/rewards', data=self.rewards)
        f.create_dataset('rb/is_terminal', data=self.is_terminal)
        f.create_dataset('rb/actions', data=self.actions)
        f.create_dataset('rb/N', data=self.N)
        f.create_dataset('rb/i', data=self.i)

        if not append:
            f.close()

    @classmethod
    def load(cls, f: Union[str, Path, h5py.File],
             close: bool = True) -> 'ReplayBuffer':
        if isinstance(f, (str, Path)):
            f = h5py.File(str(f), 'r')

        instance = cls(1, state_shape=(1,))
        instance.states = np.array(f['rb/states'])
        instance.next_states = np.array(f['rb/next_states'])
        instance.rewards = np.array(f['rb/rewards'])
        instance.is_terminal = np.array(f['rb/is_terminal'])
        instance.actions = np.array(f['rb/actions'])
        instance.N = np.array(f['rb/N']).item()
        instance.i = np.array(f['rb/i']).item()

        if close:
            f.close()

        return instance

    def __len__(self) -> int:
        return self.i

