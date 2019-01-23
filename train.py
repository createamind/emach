import algo
import gym
from atari_wrappers import wrap_deepmind, is_atari_image, is_breakout_ram, wrap_breakout

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--steps-per-epoch', type=int, default=5000)
    parser.add_argument('--start-steps', type=int, default=10000)
    parser.add_argument('--max-ep-len', type=int, default=1000)
    parser.add_argument('--exp-name', type=str, default='sac')
    parser.add_argument('--algo', type=str, default='sac')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--replay-iters', type=int, default=100)
    parser.add_argument('--prior-scale', type=float, default=2.)
    parser.add_argument('--num-ensemble', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.01)
    args = parser.parse_args()

    from logger import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)


    def wrap_env_creator(env_creator):
        def run():
            env = env_creator()
            if is_atari_image(env):
                print('Welcome Atari!')
                env = wrap_deepmind(env)
            if is_breakout_ram(env):
                print('Welcome Breakout RAM!')
                env = wrap_breakout(env)
            return env
        return run

    algos = {
        'sac': algo.sac,
        'sac1': algo.sac1,
        'ddpg': algo.ddpg,
        'td3': algo.td3,
        'sac1ex': algo.sac1ex,
        'sac1ex_mb': algo.sac1ex_mb,
        'sac1ex_rpf': algo.sac1ex_rpf,
        'eqn': algo.eqn,
        'iqn': algo.iqn
    }
    if args.algo in algos:
        algos[args.algo](wrap_env_creator(lambda : gym.make(args.env)),
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), batch_size=args.batch_size,
            gamma=args.gamma, seed=args.seed, epochs=args.epochs, replay_iters=args.replay_iters,
            steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps, max_ep_len=args.max_ep_len,
            logger_kwargs=logger_kwargs, prior_scale=args.prior_scale, num_ensemble=args.num_ensemble, alpha=args.alpha)
    else:
        raise NotImplementedError