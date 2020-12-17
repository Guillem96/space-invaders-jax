# -*- coding: utf-8 -*-

import jax
import numpy as np

import optax

import spaceinv.nn as nn


def main():
    # Reproducibility with jax random module
    key = jax.random.PRNGKey(0)
    key, *convs_keys = jax.random.split(key, 5)
    key, clf1_key, clf2_key = jax.random.split(key, 3)
    conv1_key, conv2_key, conv3_key, conv4_key = convs_keys

    # Create the model
    model = nn.sequential(
        nn.conv_2d(conv1_key,
                   in_channels=3, out_channels=64, kernel_size=5,
                   activation=jax.nn.relu),
        nn.max_pool_2d(),

        nn.conv_2d(conv2_key,
                   in_channels=64, out_channels=256, kernel_size=3,
                   padding='same', activation=jax.nn.relu),
        nn.max_pool_2d(),

        nn.conv_2d(conv3_key,
                   in_channels=256, out_channels=256, kernel_size=3,
                   padding='same', activation=jax.nn.relu),
        nn.max_pool_2d(),

        nn.conv_2d(conv4_key,
                   in_channels=256, out_channels=512, kernel_size=3,
                   padding='same', activation=jax.nn.relu),
        nn.global_avg_pooling(),

        nn.linear(clf1_key, 
                  in_features=512, out_features=512, activation=jax.nn.relu),
        nn.dropout(.5),
        nn.linear(clf2_key, 
                  in_features=512, out_features=1, activation=jax.nn.sigmoid))

    optim = optax.adam(1e-3)
    optim_state = optim.init(model.parameters)
    test_tree_seq(model.parameters)
    # test_tree_seq(optim_state[0])

    # Test layer serialization
    layer = nn.conv_2d(conv1_key,
                       in_channels=3, out_channels=64, kernel_size=5,
                       activation=jax.nn.relu)
    test_tree_dict(layer.parameters)


def test_tree_seq(tree):
    nn.save_tree(tree, 'test.h5', 'test_net')
    loaded = nn.load_tree('test.h5', 'test_net')

    assert len(loaded) == len(tree)

    for i in range(len(tree)):
        assert sorted(list(tree[i])) == sorted(list(loaded[i]))
        for k in tree[i]:
            assert np.all(loaded[i][k] == tree[i][k])


def test_tree_dict(tree):
    nn.save_tree(tree, 'test.h5', 'test_net')
    loaded = nn.load_tree('test.h5', 'test_net')
    assert sorted(list(tree)) == sorted(list(loaded))
    for k in tree:
        assert np.all(loaded[k] == tree[k])




if __name__ == '__main__':
    main()

