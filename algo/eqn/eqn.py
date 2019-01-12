import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time
import functools
from algo.eqn import core
from algo.eqn.core import get_vars
from logger import EpochLogger
from gym.spaces import Box, Discrete
import random
import sys
import copy
from collections import deque
if not (sys.version_info[0] < 3):
    print = functools.partial(print, flush=True)

EPS = 1e-5

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for EQN agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def clear(self):
        self.ptr, self.size = 0, 0


def eqn(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e4), gamma=0.99, 
        polyak=0.9, lr=1e-3, alpha='auto', batch_size=100, start_steps=10000,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, replay_iters=5, num_ensemble=5, prior_scale=1.):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    env.seed(seed)
    test_env.seed(seed)

    obs_space = env.observation_space
    act_space = env.action_space
    print(obs_space)
    print(act_space)
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['num_ensemble'] = num_ensemble
    ac_kwargs['prior_scale'] = prior_scale

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph, alpha_ph, beta_ph = core.placeholders_from_spaces(obs_space, act_space, obs_space, None, None, None, None)

    with tf.variable_scope('main'):
        pi, testpi, lorepi, loitpi, testq1, testq2, loreq, loitq = actor_critic(x_ph, a_ph, alpha_ph, beta_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, testq1_target, testq2_target, _, _ = actor_critic(x2_ph, a_ph, alpha_ph, beta_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_space.shape, act_dim=act_space.shape, size=replay_size)
    best_replay_buffer = ReplayBuffer(obs_dim=obs_space.shape, act_dim=act_space.shape, size=replay_size)
    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/testq', 'main/loreq', 'main/loitq', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'testq: %d, \t loreq: %d, \t loitq: %d, \t total: %d\n')%var_counts)


    min_q_value = tf.minimum(testq1_target, testq2_target)
    standard_q_value = r_ph + gamma * (1 - d_ph) * tf.stop_gradient(min_q_value)

    # TestQ

    testq1_loss = 0.5 * tf.reduce_mean((standard_q_value - testq1) ** 2)
    testq2_loss = 0.5 * tf.reduce_mean((standard_q_value - testq2) ** 2)
    testq_loss = testq1_loss + testq2_loss

    #loreQ
    loreq_loss = 0.5 * tf.reduce_mean((standard_q_value - loreq) ** 2)

    #loitQ
    loitq_loss = 0.5 * tf.reduce_mean((standard_q_value - loitq) ** 2)


    testq_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_testq_op = testq_optimizer.minimize(testq_loss, var_list=get_vars('main/testq') + get_vars('main/cnn'))


    loreq_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies([train_testq_op]):
        train_loreq_op = loreq_optimizer.minimize(loreq_loss, var_list=get_vars('main/loreq'))

    loitq_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    with tf.control_dependencies([train_loreq_op]):
        train_loitq_op = loitq_optimizer.minimize(loitq_loss, var_list=get_vars('main/loitq'))

    # Polyak averaging for target variables
    with tf.control_dependencies([train_loitq_op]):
        all_target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    with tf.control_dependencies([train_loreq_op]):
        only_target_update = tf.group([tf.assign(v_targ, polyak * v_targ + (1 - polyak) * v_main)
                                  for v_main, v_targ in zip(get_vars('main/loitq'), get_vars('target/loitq'))])

    all_step_ops = [testq_loss, loreq_loss, loitq_loss, testq1, testq2, loreq, loitq,
                train_testq_op, train_loreq_op, train_loitq_op, all_target_update]

    only_step_ops = [testq_loss, loreq_loss, loitq_loss, testq1, testq2, loreq, loitq,
                train_loitq_op, only_target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, #, 'alpha': alpha_ph, 'beta': beta_ph
                                outputs={'pi': pi, 'testq1': testq1, 'testq2': testq2, 'loreq': loreq, 'loitq': loitq})

    def get_action(o, lorealpha, loitbeta, test=False):
        act_op = [pi, testpi, lorepi, loitpi]
        results = sess.run(act_op, feed_dict={x_ph: np.expand_dims(o.astype(float), axis=0), alpha_ph: [lorealpha], beta_ph: [loitbeta]})
        if not test:
            logger.store(
                ChooseTest=(1 if results[0][0] == results[1][0] else 0),
                ChooseExplore=(1 if results[0][0] == results[2][0] else 0), 
                ChooseExploit=(1 if results[0][0] == results[3][0] else 0))
        # if np.random.randint(1, 100) == 1:
        #     print('under lorealpha = %.2f, loitbeta = %.2f: The model choose %d from [%d, %d, %d]' % (lorealpha, loitbeta, results[0][0], results[1][0], results[2][0], results[3][0]))
        return results[0][0]
    
    def test_agent(n = 10):
        global sess, pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, 0, 0, True))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len, ep_times = env.reset(), 0, False, 0, 0, 0
    total_steps = steps_per_epoch * epochs

    best_ret = deque()
    best_ret.append(-1e10)
    lorealpha = 0.01
    loitbeta = 0.01

    tmp_buffer = []
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        if t > start_steps:
            a = get_action(o, lorealpha, loitbeta, False)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d

        tmp_buffer.append([copy.deepcopy(o), copy.deepcopy(a), copy.deepcopy(r), copy.deepcopy(o2), copy.deepcopy(d)])
        replay_buffer.store(o, a, r, o2, d)
        o = o2

        if t % 30 == 0:
            print('.', end="")


        if d or (ep_len == max_ep_len or (t > 0 and t % steps_per_epoch == 0)):
            if ep_ret > best_ret[-1]:
                for o, a, r, o2, d in tmp_buffer:
                    best_replay_buffer.store(copy.deepcopy(o), copy.deepcopy(a), copy.deepcopy(r), copy.deepcopy(o2), copy.deepcopy(d))
            tmp_buffer = []
            for j in range(replay_iters * ep_len // batch_size):
                if j % 10 == 0:
                    print('*', end="")

                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'], #np.clip(batch['rews'], -1, 1),
                             d_ph: batch['done'],
                             alpha_ph: np.zeros_like(batch['done']),
                             beta_ph: np.zeros_like(batch['done'])
                            }

                outs = sess.run(all_step_ops, feed_dict)

                logger.store(LossTestQ=outs[0], LossExplorationQ=outs[1], LossExploitationQ=outs[2],
                                TestQ1Vals=outs[3], TestQ2Vals=outs[4], LoreQVals=outs[5], LoitQVals=outs[6],
                                LoreAlpha=lorealpha, LoitBeta=loitbeta)


            for j in range(replay_iters * ep_len // batch_size):
                if j % 10 == 0:
                    print('s', end="")

                batch = best_replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'], #np.clip(batch['rews'], -1, 1),
                             d_ph: batch['done'],
                             alpha_ph: np.zeros_like(batch['done']),
                             beta_ph: np.zeros_like(batch['done'])
                            }

                outs = sess.run(only_step_ops, feed_dict)

            if ep_ret > best_ret[-1]:
                best_ret.append(ep_ret)
                loitbeta = 100.
                lorealpha = 0.2
            else:
                loitbeta = max(0.2, loitbeta / (10.))
                if loitbeta <= 0.2 + EPS:
                    if lorealpha < 0.5:
                        lorealpha *= (2.)
                    elif lorealpha < 2:
                        lorealpha = lorealpha + (0.2) 
                    elif lorealpha < 4:
                        lorealpha = min(10, lorealpha + (0.4) / lorealpha)
                    else:
                        if len(best_ret) > 2:
                            best_ret.pop()

        
            logger.store(EpRet=ep_ret, EpLen=ep_len, BestRet=best_ret[-1])
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            ep_times += 1


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()
 
            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch

            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpTimes', ep_times)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('BestRet', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('TestQ1Vals', with_min_and_max=True) 
            logger.log_tabular('TestQ2Vals', with_min_and_max=True) 
            logger.log_tabular('LoreQVals', with_min_and_max=True)
            logger.log_tabular('LoitQVals', with_min_and_max=True)
            logger.log_tabular('LossTestQ', average_only=True)
            logger.log_tabular('LossExplorationQ', average_only=True)
            logger.log_tabular('LossExploitationQ', average_only=True)
            logger.log_tabular('LoreAlpha', average_only=True)
            logger.log_tabular('LoitBeta', average_only=True)
            logger.log_tabular('ChooseTest', average_only=True)
            logger.log_tabular('ChooseExplore', average_only=True)
            logger.log_tabular('ChooseExploit', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
