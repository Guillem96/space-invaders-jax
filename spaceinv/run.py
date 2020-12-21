# -*- coding:utf-8 -*-

import sys
import time
import functools
from pathlib import Path
from typing import Callable, Sequence

import click
import matplotlib.pyplot as plt

import jax
import gym
import cv2
import numpy as np

from spaceinv.agent import Agent
from spaceinv.replay_buffer import Transition

jax.config.update('jax_platform_name', 'cpu')


@click.command()
@click.option('--n-episodes', type=int, default=5,
              help='Number of training episodes')
@click.option('--keyboard/--no-keyboard', default=False,
              help='Want to play with keyboard?')
@click.option('--render/--no-render', default=True)
@click.option('--eval/--no-eval', default=False)
@click.option('--stack-frames', type=int, default=3)

@click.option('--save', default='checkpoints', 
              type=click.Path(file_okay=False))
@click.option('--rewards-plot', default='reports/rewards.png',
              type=click.Path(dir_okay=False))
@click.option('--resume', default=None,
              type=click.Path(dir_okay=False, exists=True))
def run(n_episodes: int, 
        keyboard: bool, 
        render: bool,
        eval: bool,
        stack_frames: int,
        save: str,
        rewards_plot: str,
        resume: str) -> None:

    save = Path(save)
    save.mkdir(exist_ok=True, parents=True)

    rewards_plot = Path(rewards_plot)
    rewards_plot.parent.mkdir(exist_ok=True, parents=True)

    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=stack_frames)
    env = gym.wrappers.FrameStack(env, stack_frames, lz4_compress=True)

    if resume is not None:
        rl_agent = Agent.load(resume)
        print(rl_agent)
    else:
        rl_agent = Agent(n_actions=env.action_space.n, 
                         stack_frames=stack_frames)
        rl_agent.save(save / 'initial_state.pkl')
        rl_agent = rl_agent.load(save / 'initial_state.pkl')

    keyboard_input_fn = functools.partial(_keyboard_input, env=env)
    take_action_fn = (rl_agent.take_action
                      if not keyboard 
                      else keyboard_input_fn)

    if eval:
        rl_agent.eval()
    else:
        rl_agent.train()

    beta = .9
    ema_rewards = [0.]

    for e in range(n_episodes):

        print(f'Episode [{e}] ', end='', flush=True)
        init_time = time.time()
        episode_reward = _run_episode(env=env, agent=rl_agent,
                                      initial_state=env.reset(),
                                      take_action_fn=take_action_fn,
                                      is_keyboard_input=keyboard,
                                      render=render)
        elapsed = int(time.time() - init_time)
        print('reward:', episode_reward, f' elapsed: {elapsed}s')

        if e % 100 == 0 and not eval:
            print('Checkpointing agent...')
            print(rl_agent)
            rl_agent.save(save / f'last_checkpoint.pkl')

        ema_rewards.append(beta * ema_rewards[-1] + 
                           (1 - beta) * episode_reward)
        _plot_ema_rewards(ema_rewards, e, rewards_plot)

    env.close()


def _run_episode(agent: Agent, 
                 env: gym.Env,
                 initial_state: np.ndarray,
                 take_action_fn: Callable[[np.ndarray], int],
                 is_keyboard_input: bool,
                 render: bool) -> float:

    episode_reward = 0
    state = initial_state

    for t in range(1000):
        _render(env, wait=not is_keyboard_input, render=render)

        action = take_action_fn(state=np.array(state))
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        t = Transition(state=state, 
                       action=action,
                       is_terminal=done,
                       reward=reward,
                       next_state=next_state)
        agent.experience(t)

        if done:
            break

        state = next_state

    return episode_reward


def _plot_ema_rewards(ema_rewards: Sequence[float], 
                      episode: int, f: Path) -> None:
    _, ax = plt.subplots()
    plt.title('Reward Exponential Moving Average')
    ax.plot(range(len(ema_rewards)), ema_rewards)
    ax.text(.05, .95, f'Last episode: {episode}', transform=ax.transAxes)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.savefig(str(f))
    plt.close()


###############################################################################

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

