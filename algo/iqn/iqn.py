import numpy as np
import tensorflow as tf
from numbers import Number
import gym
import time
import functools
from algo.iqn import core
from algo.iqn.core import get_vars
from logger import EpochLogger
from gym.spaces import Box, Discrete
import sys
if not (sys.version_info[0] < 3):
    print = functools.partial(print, flush=True)


class ReplayBuffer:

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

def iqn(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=5000, epochs=100, replay_size=int(1e4), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1, replay_iters=5, num_ensemble=5, prior_scale=2.):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    obs_space = env.observation_space
    act_space = env.action_space
    print(obs_space)
    print(act_space)
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    ac_kwargs['observation_space'] = env.observation_space
    ac_kwargs['num_ensemble'] = num_ensemble
    ac_kwargs['prior_scale'] = prior_scale
    ac_kwargs['alpha'] = alpha

    # alpha = tf.get_variable('alpha', dtype=tf.float32, initializer=0.5)

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders_from_spaces(obs_space, act_space, obs_space, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, bad_percent = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, logp_pi_, _, _,q1_pi_, q2_pi_, bad_percent= actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_space.shape, act_dim=act_space.shape, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    # min_q_pi = tf.minimum(q1_pi_, q2_pi_)
    min_q_pi = q1_pi_

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(min_q_pi - 0 * logp_pi_)
    q_backup = r_ph + gamma * (1 - d_ph) * v_backup

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(0 * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
    value_loss = q1_loss + q2_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    if isinstance(act_space, Box):
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    else:
        train_pi_op = tf.no_op()

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_params = get_vars('main/q') + get_vars('cnn')
    with tf.control_dependencies([train_pi_op]):
        train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
    with tf.control_dependencies([train_value_op]):
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(alpha), bad_percent, 
            train_pi_op, train_value_op, target_update]
            
    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    import multiprocessing
    num_cpus = min(multiprocessing.cpu_count(), 8)
    config = tf.ConfigProto(
        allow_soft_placement = True, 
        device_count={ "CPU": num_cpus },
        inter_op_parallelism_threads=num_cpus,
        intra_op_parallelism_threads=max(1, num_cpus // 4),
    )
    config.gpu_options.allow_growth = True
 
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: np.expand_dims(o.astype(float), axis=0)})[0]

    def test_agent(n=10):
        global sess, mu, pi, q1, q2, q1_pi, q2_pi
        for j in range(n):
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            # if 'ram' in env.unwrapped.spec.id:            
            #     o = o.astype(float) / 128 - 1
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True))
                if 'ram' in env.unwrapped.spec.id:                
                    o = o.astype(float) / 128 - 1
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    if 'ram' in env.unwrapped.spec.id:    
        o = o.astype(float) / 128 - 1
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > start_steps:
            a = get_action(o)
            # print(a)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        if 'ram' in env.unwrapped.spec.id:        
            o2 = o2.astype(float) / 128 - 1
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if t % 30 == 0:
            print('.', end="")
        # End of episode. Training (ep_len times).
        if d or (ep_len == max_ep_len or (t > 0 and t % steps_per_epoch == 0)):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            for j in range(replay_iters * ep_len // batch_size):
                if j % 10 == 0:
                    print('*', end="")
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'], #np.clip(batch['rews'], -1, 1),
                             d_ph: batch['done'],
                            }
                # step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, alpha, train_pi_op, train_value_op, target_update]
                outs = sess.run(step_ops, feed_dict)
                logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                            Q1Vals=outs[3], Q2Vals=outs[4],
                            LogPi=outs[5], Alpha=outs[6], RefusePercent=outs[7])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            # if 'ram' in env.unwrapped.spec.id:            
            #     o = o.astype(float) / 128 - 1


        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Alpha',average_only=True)
            logger.log_tabular('RefusePercent',with_min_and_max=True)
            logger.log_tabular('Q1Vals', with_min_and_max=True) 
            logger.log_tabular('Q2Vals', with_min_and_max=True) 
            # logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ1', average_only=True)
            logger.log_tabular('LossQ2', average_only=True)
            # logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()