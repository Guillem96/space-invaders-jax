# -*- coding:utf-8 -*-

from typing import (Callable, Mapping, 
                    NamedTuple, Tuple, 
                    Union, Sequence)

import jax
import jax.numpy as np


Parameter = Mapping[str, np.ndarray]
Kernel = Union[Tuple[int, int], int]

class Layer(NamedTuple):
    parameters: Parameter
    forward: Callable[[Tuple[Parameter, np.ndarray]], np.ndarray]
    
    # This is just for inference, in a training stage jax needs 
    # the parameters as the first argument
    def __call__(self, x: np.ndarray ,**kwargs) -> np.ndarray:
        return self.forward(self.parameters, x, **kwargs)


def linear_layer(key: jax.random.PRNGKey,
                 in_features: int, out_features: int,
                 kernel_initializer=jax.nn.initializers.glorot_normal(),
                 bias_initializer=jax.nn.initializers.zeros,
                 activation=lambda x: x) -> Layer:

    k_key, b_key = jax.random.split(key) 

    parameters = {
        'kernel': kernel_initializer(
                        k_key, 
                        shape=(in_features, out_features)),
        'bias': bias_initializer(b_key, shape=(out_features,))
    }

    def forward(parameters, x, **kwargs):
        out = np.dot(x, parameters['kernel']) + parameters['bias']
        return activation(out)

    return Layer(parameters, forward)


def flatten_layer(start_at: int = 1) -> Layer:
    
    def forward(params, x, **kwargs):
        x_shape = x.shape
        assert start_at < len(x_shape) - 1
        return x.reshape(*x_shape[:start_at], -1)

    return Layer({}, forward)


def max_pool_2d_layer(kernel_size: Kernel = 2, 
                      stride: Kernel = 2,
                      padding: str = 'SAME') -> Layer:
    kernel_size = _ensure_tuple(kernel_size)
    stride = _ensure_tuple(stride)

    def forward(params, x, **kwargs):
        return jax.lax.reduce_window(x, -np.inf, jax.lax.max, 
                                     (1, *kernel_size, 1), 
                                     (1, *stride, 1), 
                                     padding)

    return Layer({}, forward)


def conv_2d_layer(key: jax.random.PRNGKey,
                  in_channels: int,
                  out_channels: int,
                  kernel_size: Kernel,
                  stride: Kernel,
                  padding: str,
                  activation=lambda x: x,
                  kernel_initializer=jax.nn.initializers.glorot_normal(),
                  bias_initializer=jax.nn.initializers.zeros) -> Layer:

    kernel_size = _ensure_tuple(kernel_size)
    stride = _ensure_tuple(stride)

    k_key, b_key = jax.random.split(key) 
    
    parameters = {
        'kernel': kernel_initializer(
                        k_key, 
                        shape=(*kernel_size, in_channels, out_channels)),
        'bias': bias_initializer(b_key, shape=(out_channels,))
    }
    
    def forward(parameters, im, **kwargs):
        dn = jax.lax.conv_dimension_numbers(
                im.shape, 
                parameters['kernel'].shape, 
                ('NHWC', 'HWIO', 'NHWC'))
        
        out = jax.lax.conv_general_dilated(
                im, 
                parameters['kernel'], 
                stride, 
                padding, 
                (1, 1),
                (1, 1),
                dn) + parameters['bias']

        return out

    return Layer(parameters, forward)


def sequential(*layers: Sequence[Layer]) -> Layer:
    parameters = [o.parameters for o in layers]

    def forward(parameters, x, **kwargs):
        for p, l in zip(parameters, layers):
            x = l.forward(p, x, **kwargs)
        return x

    return Layer(parameters, forward)


def _ensure_tuple(k: Kernel) -> Tuple[int, int]:
    if (isinstance(k, tuple) and 
            len(k) == 2 and 
            all(isinstance(int, o) for o in k)):
        return k
    elif isinstance(k, int):
        return k, k
    else:
        raise ValueError('Kernel type must be either a tuple of '
                         '2 elements or an int')


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key, im_key, conv_key, lin_key = jax.random.split(key, 4)

    rand_image = jax.random.uniform(im_key, shape=(1, 224, 224, 3))

    model = sequential(
            conv_2d_layer(conv_key,
                          in_channels=3,
                          out_channels=64, 
                          kernel_size=2,
                          stride=2, padding='VALID'),
            max_pool_2d_layer(16, 16),
            flatten_layer(),
            linear_layer(lin_key, 3136, 10, activation=jax.nn.softmax))


    out = model(rand_image)
    print(out.shape)
    print(out)

