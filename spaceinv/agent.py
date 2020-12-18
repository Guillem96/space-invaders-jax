# -*- coding:utf-8 -*-

import random

import jax
import jax.numpy as np
import optax

import h5py

import gym

import spaceinv.nn as nn
from spaceinv.replay_buffer import ReplayBuffer, Transition


class Agent:

    def __init__(self, 
                 env: gym.Env, 
                 stack_frames: int,
                 gamma: float = .99,
                 batch_size: int = 32,
                 N: int = 1e5) -> None:

        self.env = env
        self.gamma = gamma
        self.stack_frames = stack_frames
        self.batch_size = batch_size

        self.N = N
        self.replay_buffer = ReplayBuffer(
                N=self.N,
                state_shape=self.env.observation_space.shape)

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

        self.replay_buffer.experience(transition)
        self._train()
        self.steps += 1

    def _train(self) -> None:
        # Sample from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return

        transitions = self.replay_buffer.sample(self.batch_size)

        # States: [BATCH, FRAMES, HEIGHT, WIDTH]
        states = transitions.state.astype('float32') / 255.
        next_states = transitions.next_state.astype('float32') / 255.

        # States: [BATCH, H, W, FRAMES]
        states = np.transpose(states, (0, 2, 3, 1))
        next_states = np.transpose(next_states, (0, 2, 3, 1))

        actions = transitions.action.astype('int32')
        actions = actions.reshape(-1)

        is_terminal = transitions.is_terminal.astype('bool')
        is_terminal = is_terminal.reshape(-1)

        rewards = transitions.reward.astype('float32')
        rewards = rewards.reshape(-1)

        V_star = self.Q_star(self.Q.parameters, next_states)
        yj = np.where(is_terminal, 
                      rewards,
                      rewards + self.gamma * np.max(V_star, axis=-1))

        new_params, self.optim_state = self.update_Q(
                params=self.Q.parameters, 
                target=yj, 
                optim_state=self.optim_state,
                observations=states,
                actions=actions)

        self.Q = self.Q.update(new_params)

    def take_action(self, state: np.ndarray) -> int:
        eps = _schedule_epsilon(self.epsilon, self.steps)

        if self.training and random.random() < eps:
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
        f = h5py.File(str(f), 'w')

        nn.save_tree(self.Q.parameters, f, 'Q', append=True)
        self.replay_buffer.save(f, append=True)

        f.create_dataset('steps', data=self.steps)
        f.create_dataset('gamma', data=self.gamma)
        f.create_dataset('batch_size', data=self.batch_size)
        f.create_dataset('N', data=self.N)
        f.create_dataset('stack_frames', data=self.stack_frames)
        f.close()

    @classmethod
    def load(cls, f: str, env: gym.Env) -> 'Agent':
        f = h5py.File(str(f), 'r')
        instance = cls(N=np.array(f['N']).item(),
                       env=env,
                       gamma=np.array(f['gamma']).item(),
                       batch_size=np.array(f['batch_size']).item(),
                       stack_frames=np.array(f['stack_frames']).item())

        instance.Q.update(nn.load_tree(f, 'Q', close=False))
        instance.steps = np.array(f['steps']).item()
        instance.replay_buffer = ReplayBuffer.load(f)
        return instance


@jax.jit
def _schedule_epsilon(max_eps, step):
    def decay(step):
        m = (.1 - max_eps) / 1e6
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




