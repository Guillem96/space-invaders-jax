# -*- coding:utf-8 -*-

import copy
import random
import pickle
from pathlib import Path
from typing import Union

import jax
import jax.numpy as np
import optax

import spaceinv.nn as nn
from spaceinv.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """
    Reinforcement Learning agent implmenting the DQN algorithm described at
    (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)with some slight 
    modifications.

    Paramters
    ---------
    n_actions: int
        Size of the action space
    stack_frames: int
        The number of stacked frames so the agent can gain knowledge about the
        motion
    gamma: float, default .99
        Discount factor
    batch_size: int, default 32
        Number of transitions used to update the Q parameters
    N: int, default 1e5
        Replay experience buffer capacity
    """

    def __init__(self, 
                 n_actions: int,
                 stack_frames: int,
                 gamma: float = .99,
                 batch_size: int = 32,
                 N: int = 1e5) -> None:

        self.n_actions = n_actions
        self.gamma = gamma
        self.stack_frames = stack_frames
        self.batch_size = batch_size

        self.N = N
        self.replay_buffer = ReplayBuffer(N=self.N)

        self.steps = 0
        self.epsilon = 1

        key = jax.random.PRNGKey(0)
        self.Q = _build_q_net(key, stack_frames, n_actions)

        self.training = True

        # Initialize the optimizer
        self.optim = optax.adam(1e-4)
        self.optim_state = self.optim.init(self.Q.parameters)

        # Backward step
        def loss_fn(params, target, states, actions):
            V = self.Q.forward(params, states)
            V = V[np.arange(V.shape[0]), actions]
            return nn.mse_loss(target, V)

        def update_Q(params, target, optim_state, observations, actions):
            grads = self.backward_fn(params, target, observations, actions)
            grads, optim_state = self.optim.update(grads, optim_state, params)
            new_params = optax.apply_updates(params, grads)
            return new_params, optim_state

        self.Q_star = jax.jit(self.Q.forward)
        self.Q_star_params = copy.deepcopy(self.Q.parameters)
        self.backward_fn = jax.grad(jax.jit(loss_fn))
        self.update_Q = jax.jit(update_Q)

    def train(self) -> None:
        """
        Set the agent in training mode:
            - After every experience the agent is going to update the Q 
              parameters
            - Epsilon greedy to select an action
        """
        self.training = True

    def eval(self) -> None:
        """
        Set the agent in evaluation mode:
            - No more training (fixed NN parameters)
            - Take action always use the learnt policy
        """
        self.training = False

    def experience(self, transition: Transition) -> None:
        """
        Updates the agent appending the given transition to the replay buffer
        and updates its weights sampling `batch_size` transitions
        """
        if not self.training:
            return

        self.replay_buffer.experience(transition)
        self._train()
        self.steps += 1

        if transition.is_terminal:
            self.Q_star_params = copy.deepcopy(self.Q.parameters)

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

        V_star = self.Q_star(self.Q_star_params, next_states)
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
        """
        Take an action. When training the action can be selected at random
        (epsilon greedy) or from the learnt policy. When in eval model the 
        action will always com from the "optimal" policy.

        Parameters
        ----------
        state: np.ndarray
            Current state of the game. This is a tensor of shape 
            [`stacked_frames`, 84, 84]
        """
        eps = _schedule_epsilon(self.epsilon, self.steps)

        if self.training and random.random() < eps:
            return random.randint(0, self.n_actions - 1)
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
                      f'stack_frames={self.stack_frames}, '
                      f'training={self.training})')

    def save(self, f: Union[str, Path]) -> None:
        serialized = {
            'Q': self.Q.parameters,
            'optim': self.optim_state,

            'steps': self.steps,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'N': self.N,
            'stack_frames': self.stack_frames,
            'n_actions': self.n_actions,

            'replay_buffer': self.replay_buffer
        }

        pickle.dump(serialized, open(str(f), 'wb'))

    @classmethod
    def load(cls, f: Union[str, Path]) -> 'Agent':

        unserialized = pickle.load(open(str(f), 'rb'))

        instance = cls(N=unserialized['N'],
                       n_actions=unserialized['n_actions'],
                       gamma=unserialized['gamma'],
                       batch_size=unserialized['batch_size'],
                       stack_frames=unserialized['stack_frames'])

        instance.Q.update(unserialized['Q'])
        instance.optim_state = unserialized['optim']

        if 'replay_buffer' in unserialized:
            instance.replay_buffer = unserialized['replay_buffer']
        instance.steps = unserialized['steps']

        return instance


class A2CAgent:
    pass


@jax.jit
def _schedule_epsilon(max_eps: float, step: int) -> float:
    def decay(step):
        m = (.1 - max_eps) / 1e6
        return m * step + 1.

    return jax.lax.cond(step > 1e6, 
                        lambda _: .1,
                        decay,
                        operand=step)


def _build_q_net(key: jax.random.PRNGKey, 
                 n_frames: int, n_actions: int) -> nn.Layer:

    l_keys = jax.random.split(key, 4)

    feature_extractor = [nn.conv_2d(l_keys[0],
                                    in_channels=n_frames, out_channels=16,
                                    kernel_size=8, stride=4,
                                    activation=jax.nn.relu),

                         nn.conv_2d(l_keys[1],
                                    in_channels=16, out_channels=32,
                                    kernel_size=4, stride=2,
                                    activation=jax.nn.relu)]

    classifier = [nn.flatten(),
                  nn.linear(l_keys[2],
                            in_features=2592, out_features=256,
                            activation=jax.nn.relu),
                  nn.linear(l_keys[3],
                            in_features=256, out_features=n_actions)]

    return nn.sequential(*feature_extractor, *classifier)

