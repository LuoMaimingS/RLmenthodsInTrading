import ray
from ray.tune import tune
from ray.tune.logger import pretty_print
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn as dqn
import os
import numpy as np
from ray.rllib.examples.models.rnn_model import RNNModel
from ray.rllib.models import ModelCatalog
# from virtualstock.drrl.RRLEnv import RRLEnv
from envs.financial_env import FinancialEnv
import argparse
import matplotlib.pyplot as plt


ModelCatalog.register_custom_model("rnn",  RNNModel)


class TradeModel:
    def __init__(self, model='ppo', env=FinancialEnv, stop=None, net_type="rnn"):
        self.env = env
        self.model = model
        self.net_type = net_type
        if model == 'ppo':
            self.trainer = ppo.PPOTrainer
            self.config = ppo.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 1
            self.config['model'] = {
                "custom_model": "rnn",
                "max_seq_len": 20,
            }
        elif model == 'a3c':
            self.trainer = a3c.A3CTrainer
            self.config = a3c.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 1
            self.config['model'] = {"use_lstm": True}
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
        # ray.init(redis_password='foobared', redis_max_clients=1)
        ray.init()

    def train(self):
        tune.run(self.trainer, config=self.config, stop=self.stop, checkpoint_freq=10)

    def evaluate(self, checkpoint):
        self.config['in_evaluation'] = True
        agent = self.trainer(config=self.config, env=self.env)
        chkpath = os.path.join(os.path.abspath('.'), 'checkpoint/')
        print(chkpath)
        agent.restore(chkpath+checkpoint)
        self.env = FinancialEnv()
        obs = self.env.reset()
        done = False
        episode_reward = 0.0
        reward_list = []
        if self.net_type == "rnn":
            h = np.zeros(shape=[64])
            c = np.zeros(shape=[64])
            _state = [h, c]
        else:
            _state = None
        i = 0
        while True:
            acs = []
            done = False
            while not done:
                if _state is None:
                    action = agent.compute_action(obs)
                else:
                    ret = agent.compute_action(obs, state=_state)
                    action = ret[0]
                    _state = ret[1]
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                acs.append(action - 1)
            reward_list.append(episode_reward)
            i += 1
            print("day {}, reward : {}".format(i, episode_reward))
            obs = self.env.jmp_to_next_day()
            if self.env.cur_pos == 0:
                break
        np.savez(self.model+"-runsr.npz", profit=np.array(reward_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="ppo")
    parser.add_argument("--net_type", type=str, default="rnn")
    parser.add_argument("--torch", action="store_true")
    parser.add_argument("--stop-reward", type=float, default=200)
    parser.add_argument("--stop-iters", type=int, default=100000)
    args = parser.parse_args()
    stop = {
        "episode_reward_mean": args.stop_reward,
        "training_iteration": args.stop_iters,
    }
    model = TradeModel(model=args.run, env=FinancialEnv, stop=stop, net_type=args.net_type)
    model.train()
    # model.evaluate(checkpoint=args.run+"/runsr/checkpoint_760/checkpoint-760")
