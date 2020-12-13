# -*- coding:utf-8 -*-

import sys
import functools

import click

import gym
import cv2


@click.command()
@click.option('--n-episodes', type=int, default=5,
              help='Number of training episodes')
@click.option('--keyboard/--no-keyboard', default=False,
              help='Want to play with keyboard?')
@click.option('--render/--no-render', default=True)
def run(n_episodes: int, keyboard: bool, render: bool) -> None:

    env = gym.make('SpaceInvaders-v0')
    keyboard_input_fn = functools.partial(_keyboard_input, env=env)
    take_action_fn = env.action_space.sample if not keyboard else keyboard_input_fn

    for e in range(n_episodes):
        env.reset()
        for t in range(500):
            _render(env, render=render)

            action = take_action_fn()
            observation, reward, done, info = env.step(action)

            if done:
                print(f'Episode finished after {t} timesteps')
                break

    env.close()


def _keyboard_input(env):
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


def _render(env, render: bool = True):
    if render:
        im = env.render(mode='rgb_array')
        h, w, _ = im.shape
        ar = h / w
        new_w = 400
        new_h = int(ar * new_w)
        im = cv2.resize(im[..., ::-1], (new_w, new_h))
        cv2.imshow('Space Invader Frame', im)
