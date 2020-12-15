# -*- coding:utf-8 -*-

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
                 epsilon: float = .5,
                 gamma: float = .99,
                 N: int = 1e6) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.stack_frames = stack_frames
        self.replay_buffer = deque(maxlen=int(N))

        key = jax.random.PRNGKey(0)
        self.Q = _build_q_net(key, 
                              stack_frames, 
                              self.env.action_space.n)

        # Initialize the optimizer
        self.optim = optax.adam(1e-3)
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

        self.backward_fn = jax.grad(jax.jit(loss_fn))
        self.update_Q = jax.jit(update_Q)

    def experience(self, transition: Transition) -> None:
        self.replay_buffer.append(transition)
        self._train()

    def _train(self) -> None:
        # Sample from replay buffer
        batch_size = 32 # TODO: parametrize this

        if len(self.replay_buffer) < batch_size:
            return

        transitions = random.sample(self.replay_buffer, batch_size)
        transitions = Transition(*zip(*transitions))

        # States: [BATCH, FRAMES, HEIGHT, WIDTH]
        states = _lazies_to_np(transitions.state).astype('float32')
        next_states = _lazies_to_np(transitions.next_state).astype('float32')
        states = states / 255.
        next_sates = next_states / 255.

        # States: [BATCH, H, W, FRAMES]
        states = np.transpose(states, (0, 2, 3, 1))
        next_states = np.transpose(next_states, (0, 2, 3, 1))

        actions = np.array(transitions.action)
        is_terminal = np.array(transitions.is_terminal).astype('bool')
        rewards = np.array(transitions.reward).astype('float32')
        rewards = rewards.reshape(-1)

        V = jax.jit(self.Q.forward)(self.Q.parameters, next_states)
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
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = np.expand_dims(np.array(state), 0)
            state = state.astype('float32') / 255.
            state = np.transpose(state, (0, 2, 3, 1))
            action = np.argmax(self.Q(state))
            return action.item()


def _lazies_to_np(lazy):
    return np.stack([np.array(o) for o in lazy])


def _build_q_net(key: jax.random.PRNGKey, 
                 n_frames: int, n_actions: int) -> nn.Layer:

    l_keys = jax.random.split(key, 6)

    return nn.sequential(nn.conv_2d(l_keys[0],
                                    in_channels=n_frames,
                                    out_channels=32,
                                    kernel_size=5,
                                    activation=jax.nn.relu),
                         nn.max_pool_2d(),

                         nn.conv_2d(l_keys[1],
                                    in_channels=32,
                                    out_channels=64,
                                    kernel_size=3,
                                    activation=jax.nn.relu),
                         nn.conv_2d(l_keys[2],
                                    in_channels=64,
                                    out_channels=128,
                                    kernel_size=3,
                                    activation=jax.nn.relu),
                         nn.max_pool_2d(),
                         
                         nn.conv_2d(l_keys[3],
                                    in_channels=128,
                                    out_channels=256,
                                    kernel_size=3,
                                    activation=jax.nn.relu),
                         nn.global_avg_pooling(),

                         nn.linear(l_keys[4],
                                   in_features=256,
                                   out_features=n_actions))




