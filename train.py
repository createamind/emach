import algo
import gym

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps-per-epoch', type=int, default=5000)
    parser.add_argument('--start-steps', type=int, default=10000)
    parser.add_argument('--max-ep-len', type=int, default=1000)
    parser.add_argument('--exp-name', type=str, default='sac')
    parser.add_argument('--algo', type=str, default='sac')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--replay-iters', type=int, default=5)
    parser.add_argument('--prior-scale', type=float, default=1.)
    parser.add_argument('--num-ensemble', type=int, default=5)
    args = parser.parse_args()

    from logger import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    algos = {
        'sac': algo.sac,
        'sac1': algo.sac1,
        'ddpg': algo.ddpg,
        'td3': algo.td3,
        'sac1ex': algo.sac1ex,
        'sac1ex_mb': algo.sac1ex_mb,
        'sac1ex_rpf': algo.sac1ex_rpf
    }
    if args.algo in algos:
        algos[args.algo](lambda : gym.make(args.env),
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), batch_size=args.batch_size,
            gamma=args.gamma, seed=args.seed, epochs=args.epochs, replay_iters=args.replay_iters,
            steps_per_epoch=args.steps_per_epoch, start_steps=args.start_steps, max_ep_len=args.max_ep_len,
            logger_kwargs=logger_kwargs, prior_scale=args.prior_scale, num_ensemble=args.num_ensemble)
    else:
        raise NotImplementedError