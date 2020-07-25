import ray
from ray.tune import tune
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.sac as sac
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.impala as impala
from ray.tune.registry import register_env
from models.conv1d_keras import KerasQConv1d
import os
import numpy as np
from ray.rllib.examples.models.rnn_model import RNNModel
from ray.rllib.models import ModelCatalog
from envs.financial_env import FinancialEnv
import argparse
import matplotlib.pyplot as plt


ModelCatalog.register_custom_model("rnn",  RNNModel)
ModelCatalog.register_custom_model("keras_q_model", KerasQConv1d)


def env_creator(env_config):
    print(env_config)
    env = FinancialEnv(state=env_config['state'], reward=env_config['reward'], look_back=env_config['lookback'], state_dims=2)
    return env


register_env("fin_env", env_creator)


class TradeModel:
    def __init__(self, model='ppo', env='fin_env', env_config={}, stop=None, net_type="rnn", training=True):
        self.env = env
        self.env_config = env_config
        self.model = model
        self.net_type = net_type
        if model == 'ppo':
            self.trainer = ppo.PPOTrainer
            self.config = ppo.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 16
            # self.config['lr'] = ray.tune.grid_search([1e-3, 5e-4])
            self.config['model'] = {"use_lstm": True}
        elif model == 'a3c':
            self.trainer = a3c.A3CTrainer
            self.config = a3c.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 30
            self.config['model'] = {"use_lstm": True}
        elif model == 'sac':
            self.trainer = sac.SACTrainer
            self.config = sac.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 16
            
        elif model == 'rainbow':
            self.trainer = dqn.DQNTrainer
            self.config = dqn.DEFAULT_CONFIG.copy()
            self.config["n_step"] = 10
            self.config["noisy"] = False
            self.config["num_atoms"] = 4
            self.config["v_min"] = -10.0
            self.config["v_max"] = 10.0
            self.config['num_workers'] = 16
        elif model == 'apex':
            self.trainer = dqn.ApexTrainer
            self.config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
            # self.config["optimizer"].update({"fcnet_hiddens": [512, 512]})
            self.config['num_workers'] = 30
            self.config['model'] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
        elif model == 'impala':
            self.trainer = impala.ImpalaTrainer
            self.config = impala.DEFAULT_CONFIG.copy()
            self.config['num_workers'] = 30
            self.config['model'] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
            # self.config['model'].update({"use_lstm": True, "lstm_cell_size": 512, "fcnet_hiddens": [512, 512]})
            
        else:
            raise NotImplementedError
        
        self.config['env'] = self.env
        self.config['env_config'] = self.env_config
        self.config['num_gpus'] = 1
        self.config['framework'] = "tf"
        self.stop = stop
        # ray.init(redis_password='foobared', redis_max_clients=1)
        ray.init()

    def train(self):
        print(self.config)
        tune.run(self.trainer, config=self.config, stop=self.stop, checkpoint_freq=20)

    def evaluate(self, checkpoint):
        self.config['in_evaluation'] = True
        self.config['num_workers'] = 1
        agent = self.trainer(config=self.config, env=self.env)
        chkpath = os.path.join(os.path.abspath('.'), 'checkpoint/')
        print(chkpath, self.env_config)
        agent.restore(chkpath+checkpoint)
        self.env = FinancialEnv(state=self.env_config['state'], reward=self.env_config['reward'], look_back=self.env_config['lookback'], state_dims=2)
        obs = self.env.reset()
        done = False
        episode_reward = 0.0
        reward_list = []
        if self.net_type == "rnn":
            h = np.zeros(shape=[512])
            c = np.zeros(shape=[512])
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
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
                acs.append(action - 1)
            reward_list.append(episode_reward)
            i += 1
            print("day {}, reward : {}".format(i, episode_reward))
            obs = self.env.reset()
            if self.env.cur_pos == 0:
                break
        np.savez(self.model+"-tp-state"+self.env_config["state"]+"-train.npz", profit=np.array(reward_list))
        plt.close('all')
        plt.figure(figsize=(15, 6))
        plt.plot(reward_list)
        plt.title("trading model "+self.model+", reward TP")
        plt.xlabel("trading days")
        plt.ylabel("total profit")
        plt.savefig(self.model+"-tp-state"+self.env_config["state"]+"-train.png")      


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="ppo")
    parser.add_argument("--net_type", type=str, default="rnn")
    parser.add_argument("--stop-reward", type=float, default=200)
    parser.add_argument("--stop-iters", type=int, default=100000)
    parser.add_argument("--state", type=str, default="3")
    parser.add_argument("--reward", type=str, default="earning_rate")
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--checknum", type=str, default="4180")
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument('--train', dest='is_train', action='store_true')
    flag_parser.add_argument('--evaluate', dest='is_train', action='store_false')
    parser.set_defaults(is_train=True)
    args = parser.parse_args()
    stop = {
        "episode_reward_mean": args.stop_reward,
        "training_iteration": args.stop_iters,
    }
    env_config = {}
    env_config["state"] = args.state
    env_config["reward"] = args.reward
    env_config["lookback"] = args.lookback
    print(env_config)
    model = TradeModel(model=args.run, env='fin_env', env_config=env_config, stop=stop, net_type=args.net_type, training=args.is_train)
    if args.is_train:
        model.train()
    else:
        model.evaluate(checkpoint=args.run+"/tp/checkpoint_"+args.checknum+"/checkpoint-"+args.checknum)
