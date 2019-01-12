import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    if dim is None:
        return tf.placeholder(dtype=tf.float32, shape=(None,))
    elif np.isscalar(dim):
        return tf.placeholder(dtype=tf.float32, shape=(None,dim))
    else:
        return tf.placeholder(dtype=tf.float32, shape=(None, *dim))

def placeholder_from_space(space):
    if space is None:
        return placeholder(None)
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


def mlp_ensemble_with_prior(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, num_ensemble=5, prior_scale=1.):
    
    prior_outputs = []
    for k in range(num_ensemble):
        tx = x
        for h in hidden_sizes[:-1]:
            tx = tf.layers.dense(tx, units=h, activation=activation)
        prior_outputs.append(tf.layers.dense(tx, units=hidden_sizes[-1], activation=output_activation))
    prior_outputs = tf.stack(prior_outputs, axis=1)
    prior_outputs = tf.stop_gradient(prior_outputs)

    model_outputs = []
    for k in range(num_ensemble):
        tx = x
        for h in hidden_sizes[:-1]:
            tx = tf.layers.dense(tx, units=h, activation=activation)
        model_outputs.append(tf.layers.dense(tx, units=hidden_sizes[-1], activation=output_activation))
    model_outputs = tf.stack(model_outputs, axis=1)

    real_outputs = model_outputs + prior_scale * prior_outputs
    sample = tf.multinomial(tf.log([[10.] * num_ensemble]), 1)[0][0]
    real_outputs = real_outputs[:, sample, :]
    return real_outputs

def get_vars(scope, exceptscope='MAGIC'):
    return [x for x in tf.global_variables() if (scope in x.name) and (not (exceptscope in x.name))]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def nature_cnn(unscaled_images):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    x = tf.nn.relu(tf.layers.conv2d(scaled_images, 16, [4, 4], strides=(2, 2), padding='VALID', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
        kernel_initializer=tf.contrib.layers.xavier_initializer()))

    x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='VALID', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
        kernel_initializer=tf.contrib.layers.xavier_initializer()))

    x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='VALID', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
        kernel_initializer=tf.contrib.layers.xavier_initializer()))

    print(x)
    return tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, alpha, beta, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=None, action_space=None, observation_space=None, num_ensemble=5, prior_scale=1.):
    
    with tf.variable_scope('cnn'):
        if len(x.shape) > 2: #Images
            x = nature_cnn(x)

    def vf_mlp(x, a, n, p):
        x = mlp_ensemble_with_prior(x, list(hidden_sizes)+[action_space.n], activation, None, num_ensemble=n, prior_scale=p)
        if a is None:
            return x
        x = tf.reduce_sum(x * tf.one_hot(a, action_space.n), axis=1)
        return x

    #Policy
    with tf.variable_scope('testq'):
        testq1_toselect = vf_mlp(x, None, 1, 0.)
        testq2_toselect = vf_mlp(x, None, 1, 0.)

    with tf.variable_scope('loreq'):
        # all_lore = []
        # for i in range(6):
        #     all_lore.append(vf_mlp(x, None, 1, 3))
        # all_lore = tf.stack(all_lore, axis=1)
        # interval = tf.reduce_max(all_lore, axis=1) - tf.reduce_min(all_lore, axis = 1)
        # aver = tf.reduce_mean(all_lore, axis=1)
        loreq_toselect = vf_mlp(x, None, 10, 20.)

    with tf.variable_scope('loitq'):
        loitq_toselect = vf_mlp(x, None, 1, 0.)

    toselect = (tf.minimum(testq1_toselect, testq2_toselect) + alpha * loreq_toselect + beta * loitq_toselect) / (1 + alpha + beta)
    pi = tf.argmax(toselect, 1)

    testpi = tf.argmax(tf.minimum(testq1_toselect, testq2_toselect), 1)
    lorepi = tf.argmax(loreq_toselect, 1)
    loitpi = tf.argmax(loitq_toselect, 1)

    with tf.variable_scope('testq', reuse=True):
        testq1 = vf_mlp(x, a, 1, 0.)
        testq2 = vf_mlp(x, a, 1, 0.)

    with tf.variable_scope('loreq', reuse=True):
        loreq = vf_mlp(x, a, 10, 16.)

    with tf.variable_scope('loitq', reuse=True):
        loitq = vf_mlp(x, a, 1, 0.)

    return pi, testpi, lorepi, loitpi, testq1, testq2, loreq, loitq
