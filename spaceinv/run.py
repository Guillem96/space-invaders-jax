# -*- coding:utf-8 -*-

import sys
import functools

import click
import matplotlib.pyplot as plt

import gym
import cv2

from spaceinv.agent import Agent, Transition


@click.command()
@click.option('--n-episodes', type=int, default=5,
              help='Number of training episodes')
@click.option('--keyboard/--no-keyboard', default=False,
              help='Want to play with keyboard?')
@click.option('--render/--no-render', default=True)
@click.option('--stack-frames', type=int, default=4)
def run(n_episodes: int, 
        keyboard: bool, 
        render: bool,
        stack_frames: int) -> None:

    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=2)
    env = gym.wrappers.FrameStack(env, stack_frames)

    rl_agent = Agent(env, stack_frames=stack_frames) 
    keyboard_input_fn = functools.partial(_keyboard_input, env=env)
    take_action_fn = (rl_agent.take_action
                      if not keyboard 
                      else keyboard_input_fn)

    rewards = []

    for e in range(n_episodes):
        state = env.reset()
        episode_reward = 0

        print(f'Episode [{e}]')

        for t in range(1000):
            _render(env, wait=not keyboard, render=render)

            action = take_action_fn(state=state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            t = Transition(state=state, 
                           action=action,
                           is_terminal=done,
                           reward=reward,
                           next_state=next_state)
            rl_agent.experience(t)

            if done:
                print(f'Episode finished after {t} timesteps')
                break

            state = next_state
            print('.', end='', flush=True)

        rewards.append(episode_reward)

        if len(rewards) > 5:
            y = _moving_average(rewards, 5)
            x = range(y.shape[0])
            plt.plot(x, y)
            plt.xlabel('Episodes')
            plt.ylabel('Reward')
            plt.savefig('test.png')

    env.close()


def _moving_average(a, window=3):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _keyboard_input(env, **kwargs):
    key = cv2.waitKey(30) & 0xFF
    if key == 255:
        return 0
    elif key == ord('q') or key == 27:
        sys.exit(0)
    else:
        mapping = env.get_keys_to_action()
        for k, v in mapping.items():
            if key in k:
                return v

        return 0


def _render(env, wait:bool = False, render: bool = True):
    if render:
        im = env.render(mode='rgb_array')
        h, w, _ = im.shape
        ar = h / w
        new_w = 400
        new_h = int(ar * new_w)
        im = cv2.resize(im[..., ::-1], (new_w, new_h))
        cv2.imshow('Space Invader Frame', im)

        if wait:
            if cv2.waitKey(10) == 27:
                sys.exit(0)

