import ray
from ray.tune import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn as dqn

import numpy as np
from ray.rllib.examples.models.rnn_model import RNNModel, TorchRNNModel
from ray.rllib.models import ModelCatalog
from virtualstock.drrl.RRLEnv import RRLEnv
import argparse
ModelCatalog.register_custom_model("rnn",  RNNModel)


class TradeModel:
    def __init__(self, model='ppo', env=RRLEnv, stop=None):
        self.env = RRLEnv
        if model == 'ppo':
            self.trainer = ppo.PPOTrainer
            self.config = ppo.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 1
            self.config['model'] = {
                "custom_model": "rnn",
                "max_seq_len": 20,
            }
        elif model == 'sac':
            self.trainer = sac.SACTrainer
            self.config = sac.DEFAULT_CONFIG.copy()
            
        elif model == 'rainbow':
            self.trainer = dqn.DQNTrainer
            self.config = dqn.DEFAULT_CONFIG.copy()
            self.config["n_step"] = 10
            self.config["noisy"] = True
            self.config["num_atoms"] = 4
            self.config["v_min"] = -10.0
            self.config["v_max"] = 10.0
            
        else:
            raise NotImplementedError
        
        self.config['env'] = self.env
        self.config['num_gpus'] = 1
        self.config['framework'] = "tf"
        self.stop = stop
        ray.init()

    def train(self):
        tune.run(self.trainer, config=self.config, stop=self.stop, checkpoint_freq=10)

    def evaluate(self, checkpoint):
        agent = ppo.PPOTrainer(config=self.config, env=self.env)
        agent.restore(checkpoint)        
        obs = self.env.reset()
        done = False
        episode_reward = 0.0
        h = np.zeros(shape=[64])
        c = np.zeros(shape=[64])
        _state = [h, c]
        while not done:
            ret = agent.compute_action(obs, state=_state)
            action = ret[0]
            _state = ret[1]
            obs, reward, done, info = self.env.step(action)
            print(info)
            episode_reward += reward
        print(episode_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="ppo")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--stop-reward", type=float, default=2)
    parser.add_argument("--stop-iters", type=int, default=100000)
    args = parser.parse_args()
    stop = {
        "episode_reward_mean": args.stop_reward,
        "training_iteration": args.stop_iters,
    }
    model = TradeModel(model=args.run, env=RRLEnv, stop=stop)
    model.train()

    

