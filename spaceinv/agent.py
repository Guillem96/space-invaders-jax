# -*- coding:utf-8 -*-

import pickle
import random
from collections import deque
from typing import NamedTuple

import jax
import jax.numpy as np
import optax

import gym

import spaceinv.nn as nn


class Transition(NamedTuple):

    state: np.ndarray
    action: int
    reward: float
    is_terminal: bool
    next_state: np.ndarray


class Agent:

    def __init__(self, 
                 env: gym.Env, 
                 stack_frames: int,
                 gamma: float = .99,
                 batch_size: int = 32,
                 N: int = 1e6) -> None:

        self.env = env
        self.gamma = gamma
        self.stack_frames = stack_frames
        self.batch_size = batch_size

        self.N = N
        self.replay_buffer = deque(maxlen=int(N))
        self.steps = 0
        self.epsilon = 1

        key = jax.random.PRNGKey(0)
        self.Q = _build_q_net(key, 
                              stack_frames, 
                              self.env.action_space.n)

        self.training = True

        # Initialize the optimizer
        self.optim = optax.rmsprop(1e-4)
        self.optim_state = self.optim.init(self.Q.parameters)

        # Backward step
        def loss_fn(params, target, states, actions):
            V = self.Q.forward(params, states)
            V = V[np.arange(V.shape[0]), actions]
            return np.mean((V - target) ** 2)

        def update_Q(params, target, optim_state, observations, actions):
            grads = self.backward_fn(params, target, observations, actions)
            grads, optim_state = self.optim.update(grads, optim_state, params)
            new_params = optax.apply_updates(params, grads)
            return new_params, optim_state

        self.Q_star = jax.jit(self.Q.forward)
        self.backward_fn = jax.grad(jax.jit(loss_fn))
        self.update_Q = jax.jit(update_Q)

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def experience(self, transition: Transition) -> None:
        if not self.training:
            return

        self.replay_buffer.append(transition)
        self._train()
        self.steps += 1

    def _train(self) -> None:
        # Sample from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = random.sample(self.replay_buffer, self.batch_size)
        transitions = Transition(*zip(*transitions))

        # States: [BATCH, FRAMES, HEIGHT, WIDTH]
        states = _lazies_to_np(transitions.state)
        next_states = _lazies_to_np(transitions.next_state)

        # States: [BATCH, H, W, FRAMES]
        states = np.transpose(states, (0, 2, 3, 1))
        next_states = np.transpose(next_states, (0, 2, 3, 1))

        states = states.astype('float32') / 255.
        next_states = next_states.astype('float32') / 255.

        actions = np.array(transitions.action)
        is_terminal = np.array(transitions.is_terminal).astype('bool')
        rewards = np.array(transitions.reward).astype('float32')
        rewards = rewards.reshape(-1)

        V = self.Q_star(self.Q.parameters, next_states)
        yj = np.where(is_terminal, 
                      rewards,
                      rewards + self.gamma * np.max(V, axis=-1))

        new_params, self.optim_state = self.update_Q(
                params=self.Q.parameters, 
                target=yj, 
                optim_state=self.optim_state,
                observations=states,
                actions=actions)

        self.Q = self.Q.update(new_params)

    def take_action(self, state: np.ndarray) -> int:
        eps = _schedule_epsilon(self.epsilon, self.steps)

        if self.training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = state.astype('float32') / 255.
            state = np.expand_dims(np.array(state), 0)
            state = np.transpose(state, (0, 2, 3, 1))
            action = np.argmax(self.Q(state))
            return action.item()

    def __str__(self) -> str:
        eps = _schedule_epsilon(self.epsilon, self.steps)
        eps = f'{eps:.3f}'

        return (f'Agent(steps={self.steps}, '
                      f'epsilon={eps}, '
                      f'gamma={self.gamma}, '
                      f'stack_frames={self.stack_frames})')

    def save(self, f: str) -> None:
        checkpoint = {
            'Q': self.Q.parameters,
            'optim': self.optim_state,

            'steps': self.steps,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'replay_buffer': self.replay_buffer,
            'N': self.N,
            'stack_frames': self.stack_frames
        }

        pickle.dump(checkpoint, open(str(f), 'wb'))

    @classmethod
    def load(cls, f: str, env: gym.Env) -> 'Agent':
        chekpoint = pickle.load(open(str(f), 'rb'))
        instance = cls(env=env, 
                       batch_size=checkpoint['batch_size'],
                       gamma=checkpoint['gamma'],
                       N=checkpoint['N'],
                       stack_frames=checkpoint['stack_frames'])

        instance.Q.update(checkpoint['parameters'])
        instance.optim_state = checkpoint['optim']
        instance.steps = checkpoint['steps']
        instance.replay_buffer = checkpoint['replay_buffer']
        return instance


@jax.jit
def _schedule_epsilon(eps, step):
    def decay(step):
        m = (.1 - 1.) / 1e6
        return m * step + 1.

    return jax.lax.cond(step > 1e6, 
                        lambda _: .1,
                        decay,
                        operand=step)


def _lazies_to_np(lazy):
    return np.stack([np.array(o) for o in lazy])


def _build_q_net(key: jax.random.PRNGKey, 
                 n_frames: int, n_actions: int) -> nn.Layer:

    l_keys = jax.random.split(key, 4)

    return nn.sequential(nn.conv_2d(l_keys[0],
                                    in_channels=n_frames,
                                    out_channels=16,
                                    kernel_size=8,
                                    stride=4,
                                    activation=jax.nn.relu),

                         nn.conv_2d(l_keys[1],
                                    in_channels=16,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    activation=jax.nn.relu),
                         nn.flatten(),
                         nn.linear(l_keys[2],
                                   in_features=2592,
                                   out_features=256,
                                   activation=jax.nn.relu),
                         nn.linear(l_keys[3],
                                   in_features=256,
                                   out_features=n_actions))




