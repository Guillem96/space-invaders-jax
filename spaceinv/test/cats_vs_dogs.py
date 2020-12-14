# -*- coding:utf-8 -*-

from typing import Tuple

import tqdm.auto as tqdm

import jax
import jax.numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import spaceinv.nn as nn


# Training parameters (this is a bit ugly, but works for testing :))
batch_size = 32
im_size = 224
epochs = 5


def resize_im(im: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    im = tf.image.resize(im, [im_size, im_size]) / 255.
    return im, label


def main():
    ds_len = 23262
    train_ds_len = int(ds_len * .8)
    key = jax.random.PRNGKey(0)

    ds = tfds.load('cats_vs_dogs', 
                   split='train',
                   as_supervised=True,
                   shuffle_files=True)

    train_ds = ds.take(train_ds_len).shuffle(1024)
    valid_ds = ds.skip(train_ds_len)

    train_ds = train_ds.map(resize_im)
    valid_ds = valid_ds.map(resize_im)

    train_ds = tfds.as_numpy(train_ds.batch(batch_size))
    valid_ds = tfds.as_numpy(valid_ds.batch(batch_size))

    key, *convs_keys = jax.random.split(key, 5)
    key, clf_key = jax.random.split(key)

    conv1_key, conv2_key, conv3_key, conv4_key = convs_keys

    model = nn.sequential(
        nn.conv_2d(conv1_key, 
                   in_channels=3, 
                   out_channels=64, 
                   kernel_size=5, 
                   padding='valid', 
                   activation=jax.nn.relu),
        nn.max_pool_2d(),

        nn.conv_2d(conv2_key,
                   in_channels=64,
                   out_channels=256,
                   kernel_size=3,
                   padding='same',
                   activation=jax.nn.relu),
        nn.max_pool_2d(),

        nn.conv_2d(conv3_key,
                   in_channels=256,
                   out_channels=512,
                   kernel_size=3,
                   padding='same',
                   activation=jax.nn.relu),
        nn.max_pool_2d(),
        
        nn.conv_2d(conv4_key,
                   in_channels=512,
                   out_channels=1024,
                   kernel_size=3,
                   padding='same'),
        nn.global_avg_pooling(),

        nn.linear(clf_key, 
                  in_features=1024, 
                  out_features=1, 
                  activation=jax.nn.sigmoid))

    def loss_fn(parameters, images, labels):
        predictions = model.forward(parameters, images)
        loss = nn.bce_loss(labels, predictions) / batch_size
        return loss

    backward_fn = jax.value_and_grad(jax.jit(loss_fn))
    optim = jax.jit(nn.simple_optimizer(lr=1e-3))

    for e in range(epochs):
        pbar = tqdm.tqdm(train_ds, 
                         desc=f'Epoch[{e}]',
                         total=train_ds_len // batch_size)

        running_loss = 0.
        for i, (im, label) in enumerate(pbar):
            value, grad = backward_fn(model.parameters, im, label)

            new_params = jax.tree_multimap(optim, model.parameters, grad)
            model = model.update(new_params)

            running_loss += value
            loss = running_loss / (i + 1)
            pbar.set_postfix({'loss': f'{loss:.5f}'})

    running_acc = 0.
    running_loss = 0.

    for i, (im, label) in enumerate(tqdm.tqdm(valid_ds)):
        predictions = jax.jit(model.forward)(model.parameters, im)
        loss = nn.bce_loss(label, predictions) / batch_size
        running_loss += loss
        predictions = predictions > .5

        n_correct = np.sum(predictions == label)
        acc = n_correct / predictions.shape[0]
        running_acc += acc

    acc = running_acc / i
    print('Accuracy:', acc, 'Loss oss:', running_loss / i)


if __name__ == '__main__':
    main()

