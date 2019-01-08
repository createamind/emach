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

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

def nature_cnn(unscaled_images):
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.

    x = tf.nn.relu(tf.layers.conv2d(scaled_images, 16, [8, 8], strides=(4, 4), padding='VALID', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
        kernel_initializer=tf.contrib.layers.xavier_initializer()))

    x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='VALID', 
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
        kernel_initializer=tf.contrib.layers.xavier_initializer()))

    # x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='VALID', 
    #     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
    #     kernel_initializer=tf.contrib.layers.xavier_initializer()))

    # x = tf.nn.relu(tf.layers.conv2d(x, 32, [4, 4], strides=(2, 2), padding='VALID', 
    #     kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
    #     kernel_initializer=tf.contrib.layers.xavier_initializer()))

    print(x)
    return tf.reshape(x, [-1, x.shape[1] * x.shape[2] * x.shape[3]])

    # activ = tf.nn.relu
    # h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    # h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    # h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), **conv_kwargs))
    # h3 = conv_to_fc(h3)
    # return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    # if len(x.shape) > 2: #Images
    #     x = nature_cnn(x)
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi


# def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#     if len(x.shape) > 2: #Images
#         x = nature_cnn(x)
#     act_dim = action_space.n
#     logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
#     logp_all = tf.nn.log_softmax(logits)
#     # tmd = tf.Print(tmd, [tmd], summarize=1000)
#     # logits = tf.Print(logits, [logits])
#     # logp_all = tf.Print(logp_all, [logp_all])
#     mu = tf.argmax(logits, 1)
#     # pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
#     pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
#     # pi = tf.Print(pi, [mu, pi])
#     # logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
#     logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
#     return mu, pi, logp_pi


# def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#     act_dim = a.shape.as_list()[-1]
#     mu = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
#     log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
#     std = tf.exp(log_std)
#     pi = mu + tf.random_normal(tf.shape(mu)) * std
#     logp = gaussian_likelihood(a, mu, log_std)
#     logp_pi = gaussian_likelihood(pi, mu, log_std)
#     return pi, logp, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(alpha, x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=None, action_space=None, observation_space=None):
    
    with tf.variable_scope('cnn'):
        if len(x.shape) > 2: #Images
            x = nature_cnn(x)

    def vf_mlp(x, a, all_values=False):
        # if len(x.shape) > 2: #Images
        #     x = nature_cnn(x)
        if isinstance(action_space, Box):
            x = tf.concat([x,a], axis=-1)
            return tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        elif isinstance(action_space, Discrete):
            x = mlp(x, list(hidden_sizes)+[action_space.n], activation, None)
            if all_values:
                return x
            x = tf.reduce_sum(x * tf.one_hot(a, action_space.n), axis=1)
            return x

    with tf.variable_scope('q1'):
        q1 = vf_mlp(x, a)

    # with tf.variable_scope('pi'):
    #     if isinstance(action_space, Box):
    #         mu, pi, logp_pi = mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation)
    #         mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
    #         # make sure actions are in correct range
    #         action_scale = action_space.high[0]
    #         mu *= action_scale
    #         pi *= action_scale

    #     elif isinstance(action_space, Discrete):
    #         mu, pi, logp_pi = mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space)


    # policy
    if isinstance(action_space, Box):
        with tf.variable_scope('pi'):
            mu, pi, logp_pi = mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation)
            mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
            # make sure actions are in correct range
            action_scale = action_space.high[0]
            mu *= action_scale
            pi *= action_scale

    elif isinstance(action_space, Discrete):
        with tf.variable_scope('q1', reuse=True):
            all_qs = vf_mlp(x, None, all_values=True)
            logp_all = tf.nn.log_softmax(all_qs * alpha)
            # logp_all = tf.Print(logp_all, [all_qs, logp_all])
            mu = tf.argmax(logp_all, 1)
            pi = tf.squeeze(tf.multinomial(logp_all, 1), axis=1)
            logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=action_space.n) * logp_all, axis=1)

    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(x, mu)
    with tf.variable_scope('q2'):
        q2 = vf_mlp(x, a)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(x, mu)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi