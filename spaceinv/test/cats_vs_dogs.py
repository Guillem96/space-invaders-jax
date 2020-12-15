# -*- coding:utf-8 -*-

import functools
from typing import Tuple

import tqdm.auto as tqdm

import jax
import jax.numpy as np
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

import spaceinv.nn as nn


# Training parameters (this is a bit ugly, but works for testing :))
batch_size = 32
im_size = 150
epochs = 5


def resize_im(im: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    im = tf.image.resize(im, [im_size, im_size]) / 255.
    return im, label


def main():
    # Load Cats and dogs dataset using Tensorflow Datasets
    ds_len = 23262 # Statistic provided by the tensorflow page
    train_ds_len = int(ds_len * .8) # 80% to train set

    # Download the data
    ds = tfds.load('cats_vs_dogs', 
                   split='train',
                   as_supervised=True,
                   shuffle_files=True)

    train_ds = ds.take(train_ds_len).shuffle(1024)
    train_ds = train_ds.map(resize_im)
    train_ds = tfds.as_numpy(train_ds.batch(batch_size))
    
    valid_ds = ds.skip(train_ds_len)
    valid_ds = valid_ds.map(resize_im)
    valid_ds = tfds.as_numpy(valid_ds.batch(batch_size))
    
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

    # Create and intizalize the adam optimizer
    optim = optax.adam(learning_rate=1e-4)
    state = optim.init(model.parameters)

    def loss_fn(parameters, key, images, labels):
        predictions = model.forward(parameters, images, key=key)
        loss = nn.bce_loss(labels, predictions) / batch_size
        return loss

    @jax.jit
    def train_step(key, params, state, images, labels):
        backward_fn = jax.value_and_grad(jax.jit(loss_fn))
        loss, grads = backward_fn(params, key, images, labels)
        grads, state = optim.update(grads, state, params)
        new_params = optax.apply_updates(params, grads)
        return loss, new_params, state

    # Training
    for e in range(epochs):
        pbar = tqdm.tqdm(train_ds, 
                         desc=f'Epoch[{e}]',
                         total=train_ds_len // batch_size)

        running_loss = 0.
        for i, (im, label) in enumerate(pbar):
            key, subkey = jax.random.split(key)
            loss, new_params, state = train_step(key=key,
                                                 params=model.parameters, 
                                                 state=state,
                                                 images=im, 
                                                 labels=label)
            model = model.update(new_params)

            running_loss += loss
            mean_loss = running_loss / (i + 1)
            pbar.set_postfix({'loss': f'{mean_loss:.5f}'})

    # Run validation
    running_acc = 0.
    running_loss = 0.
    valid_forward = functools.partial(model.forward, training=False)

    for i, (im, label) in enumerate(tqdm.tqdm(valid_ds)):
        probas = jax.jit(valid_forward)(model.parameters, im)
        loss = nn.bce_loss(label, probas) / batch_size

        predictions = probas.reshape(-1) > .5
        acc = np.mean(predictions.astype('int32') == label)

        running_acc += acc
        running_loss += loss

    acc = running_acc / i
    print('Accuracy:', acc, 'Loss:', running_loss / i)


if __name__ == '__main__':
    main()

