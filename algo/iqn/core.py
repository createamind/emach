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

    return real_outputs

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

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    return mu, pi, log_std

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=None, policy=None, action_space=None, observation_space=None, num_ensemble=5, prior_scale=2., alpha=0.2):
    
    with tf.variable_scope('cnn'):
        if len(x.shape) > 2: #Images
            x = nature_cnn(x)

    def vf_mlp(x, a, num_ensemble, prior_scale):
        if isinstance(action_space, Box):
            x = tf.concat([x, a], axis=-1)
            all_outputs = mlp_ensemble_with_prior(x, list(hidden_sizes)+[1], activation, None, num_ensemble=num_ensemble, prior_scale=prior_scale)
        elif isinstance(action_space, Discrete):
            all_outputs = mlp_ensemble_with_prior(x, list(hidden_sizes)+[action_space.n], activation, None, num_ensemble=num_ensemble, prior_scale=prior_scale)
        else:
            assert(0)
        
        sample = tf.multinomial(tf.log([[10.] * num_ensemble]), 1)[0][0]
        real_outputs = all_outputs[:, sample, :]

        if isinstance(action_space, Box):
            return tf.squeeze(real_outputs, axis=1), all_outputs
        elif isinstance(action_space, Discrete):
            if a is None:
                return real_outputs, all_outputs
            return tf.reduce_sum(real_outputs * tf.one_hot(a, action_space.n), axis=1), \
            tf.reduce_sum(all_outputs * tf.tile(tf.expand_dims(tf.one_hot(a, action_space.n), 1), [1, num_ensemble, 1]), axis=2)
        else:
            assert(0)

    with tf.variable_scope('q1'):
        q1, _ = vf_mlp(x, a, 1, 0.)
    with tf.variable_scope('q2') as q2scope:
        q2, _ = vf_mlp(x, a, num_ensemble, prior_scale)

    def bad_policy(x, pi, action_space, q2scope):
        with tf.variable_scope(q2scope, reuse=True):
            _, all_values = vf_mlp(x, pi, num_ensemble, prior_scale)
            # print(all_values)
            # print(num_ensemble)
            # all_values = tf.Print(all_values, [all_values], summarize=10)
            # all_values = tf.squeeze(all_values, axis=2)
            max_values = tf.reduce_max(all_values, axis=1)
            min_values = tf.reduce_min(all_values, axis=1)
            cha_values = max_values - min_values
            # mean_values = tf.reduce_mean(all_values, axis=1)
            # return tf.squeeze(tf.multinomial(tf.log(all_values * 0 + 10), 1) < 1, axis=-1)
            tmp = (cha_values / (tf.abs(max_values) + tf.abs(min_values)))
            # tmp = tf.Print(tmp, [tmp], summarize=10)
            return tmp < alpha

    if isinstance(action_space, Box):
        with tf.variable_scope('pi'):
            mu, pi, log_std = mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation)
            pi = mu + 0
            
            isbad = tf.cast(bad_policy(x, pi, action_space, q2scope), tf.float32)
            bad_percent = tf.reduce_mean(isbad, axis=0)

            random_pi = tf.random_uniform(tf.reduce_max(pi, axis=0).shape, minval=-1, maxval=1)
            random_pi = tf.expand_dims(random_pi, axis=0)

            pi = isbad * random_pi + (1 - isbad) * pi

            logp_pi = gaussian_likelihood(pi, mu, log_std)
            mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)
            action_scale = action_space.high[0]
            mu *= action_scale
            pi *= action_scale

    elif isinstance(action_space, Discrete):
        with tf.variable_scope('q1', reuse=True):
            all_qs, _ = vf_mlp(x, None, 1, 0.)
            logp_all = tf.nn.log_softmax(all_qs)

            mu = tf.argmax(all_qs, 1)
            pi = mu + 0

            isbad = tf.cast(bad_policy(x, pi, action_space, q2scope), tf.int64)
            bad_percent = tf.reduce_mean(tf.cast(isbad, tf.float32), axis=0)
            # isbad = tf.Print(isbad, [isbad], summarize=10)
            random_pi = tf.squeeze(tf.multinomial(logp_all * 0 + 10, 1), axis=1)
            # random_pi = tf.multinomial(tf.log([[10.] * action_space.n]), 1)[0][0]
            pi = isbad * random_pi + (1 - isbad) * pi

            logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=action_space.n) * logp_all, axis=1)
    else:
        assert(0)

    with tf.variable_scope('q1', reuse=True):
        q1_pi, _ = vf_mlp(x, pi, 1, 0.)
    with tf.variable_scope('q2', reuse=True):
        q2_pi, _ = vf_mlp(x, pi, num_ensemble, prior_scale)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, bad_percent