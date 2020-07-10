import ray
from ray.tune import tune
import ray.rllib.agents.ppo as ppo

from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from virtualstock.drrl.RRLEnv import RRLEnv
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="PPO")
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--as-test", action="store_true")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--stop-reward", type=float, default=2)
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus or None)

    ModelCatalog.register_custom_model(
        "rnn", TorchRNNModel if args.torch else RNNModel)
    
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_gpus'] = 0
    config['num_workers'] = 1
    config['env'] = RRLEnv
    config['model'] = {
                "custom_model": "rnn",
                "max_seq_len": 20,
    }
    config['framework'] = "torch" if args.torch else "tf"
    stop = {
        "episode_reward_mean": args.stop_reward,
    }
    
    analysis = tune.run(ppo.PPOTrainer, config=config, stop=stop, checkpoint_at_end=True)
    checkpoint = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean")
    print(checkpoint)
    agent = ppo.PPOTrainer(config=config, env=RRLEnv)
    agent.restore(checkpoint)
    env = RRLEnv()
    obs = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

