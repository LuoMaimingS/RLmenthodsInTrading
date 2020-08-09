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
import multiprocessing
import numpy as np
import tensorflow as tf
from ray.rllib.examples.models.rnn_model import RNNModel
from ray.rllib.models import ModelCatalog
from envs.financial_env import FinancialEnv
import argparse
import matplotlib.pyplot as plt


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ModelCatalog.register_custom_model("rnn",  RNNModel)
ModelCatalog.register_custom_model("keras_q_model", KerasQConv1d)


def env_creator(env_config):
    print(env_config)
    env = FinancialEnv(security=env_config["security"], state=env_config["state"], reward=env_config["reward"], look_back=env_config["lookback"], state_dims=2)
    return env


register_env("fin_env", env_creator)


class TradeModel:
    def __init__(self, model="ppo", env="fin_env", env_config={}, stop=None, net_type="rnn", training=True):
        self.env = env
        self.env_config = env_config
        self.model = model
        self.net_type = net_type
        if model == "ppo":
            self.trainer = ppo.PPOTrainer
            self.config = ppo.DEFAULT_CONFIG.copy()
            self.config["num_workers"] = multiprocessing.cpu_count() - 1
            self.config.update({
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "lr": 5e-4
            })
            # self.config["model"] = {"use_lstm": True}
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
        elif model == "a3c":
            self.trainer = a3c.A3CTrainer
            self.config = a3c.DEFAULT_CONFIG.copy()
            self.config["num_workers"] = multiprocessing.cpu_count() - 1
            # self.config["model"] = {"use_lstm": True}
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
            self.config["rollout_fragment_length"] = 50
            self.config["train_batch_size"] = 512
        elif model == "appo":
            self.trainer = ppo.appo.APPOTrainer
            self.config = ppo.appo.DEFAULT_CONFIG.copy()
            self.config["vtrace"] = True
            self.config["num_workers"] = multiprocessing.cpu_count() - 1
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
        elif model == "sac":
            self.trainer = sac.SACTrainer
            self.config = sac.DEFAULT_CONFIG.copy()
            self.config["num_workers"] = 20
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
            self.config["Q_model"].update({
                "fcnet_activation": "relu",
                "fcnet_hiddens": [512, 512]
            })
            self.config["policy_model"].update({
                "fcnet_activation": "relu",
                "fcnet_hiddens": [512, 512]
            })
            self.config["timesteps_per_iteration"] = 25000
            self.config["learning_starts"] = 50000
            self.config["target_network_update_freq"] = 500000
            self.config["prioritized_replay"] = True
            self.config["buffer_size"] = int(2e6)
            self.config["train_batch_size"] = 512
            self.config["rollout_fragment_length"] = 50
            self.config["exploration_config"] = {"type": "PerWorkerEpsilonGreedy"}
            self.config["optimization"] = {
                "actor_learning_rate": 5e-4,
                "critic_learning_rate": 5e-4,
                "entropy_learning_rate": 5e-4,
            }
        elif model == "apex":
            # faster training with similar timestep efficiency with Rainbow or DQN
            self.trainer = dqn.ApexTrainer
            self.config = dqn.apex.APEX_DEFAULT_CONFIG.copy()
            # self.config["optimizer"].update({"fcnet_hiddens": [512, 512]})
            self.config["num_workers"] = multiprocessing.cpu_count() - 1
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
        elif model == "impala":
            # faster training with similar timestep efficiency with PPO or A3C
            self.trainer = dqn.ApexTrainer
            self.trainer = impala.ImpalaTrainer
            self.config = impala.DEFAULT_CONFIG.copy()
            self.config["num_workers"] = multiprocessing.cpu_count() - 1
            self.config["model"] = {
                "custom_model": "keras_q_model",
                "custom_model_config": {"training": training}
            }
            # self.config["model"].update({"use_lstm": True, "lstm_cell_size": 512, "fcnet_hiddens": [512, 512]})
            
        else:
            raise NotImplementedError
        
        self.config["env"] = self.env
        self.config["env_config"] = self.env_config
        if tf.test.is_gpu_available():
            self.config["num_gpus"] = 1
        else:
            self.config["num_gpus"] = 0
        self.config["framework"] = "tf"
        self.stop = stop
        # ray.init(redis_password="foobared", redis_max_clients=1)
        ray.init()

    def train(self):
        print(self.config)
        tune.run(self.trainer, config=self.config, stop=self.stop, checkpoint_freq=20)

    def evaluate(self, checkpoint):
        self.config["in_evaluation"] = True
        self.config["num_workers"] = 1
        agent = self.trainer(config=self.config, env=self.env)
        chkpath = os.path.join(os.path.abspath("."), "checkpoint/")
        print(chkpath, self.env_config)
        agent.restore(chkpath+checkpoint)
        self.env = FinancialEnv(state=self.env_config["state"], reward=self.env_config["reward"], look_back=self.env_config["lookback"], state_dims=2)
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
                print("end...")
                break
        np.savez(self.model+"-state"+self.env_config["state"]+"-evala.ganseql5.npz", profit=np.array(reward_list))
        plt.close("all")
        plt.figure(figsize=(15, 6))
        plt.plot(reward_list)
        plt.title("trading model "+self.model+", reward TP")
        plt.xlabel("trading days")
        plt.ylabel("total profit")
        plt.savefig(self.model+"-state"+self.env_config["state"]+"-evala.ganseql5.png")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="apex")
    parser.add_argument("--net_type", type=str, default="conv1d")
    parser.add_argument("--stop-reward", type=float, default=200)
    parser.add_argument("--stop-iters", type=int, default=100000)
    parser.add_argument("--state", type=str, default="3")
    parser.add_argument("--reward", type=str, default="earning_rate")
    parser.add_argument("--lookback", type=int, default=50)
    parser.add_argument("--checknum", type=str, default="1600")
    flag_parser = parser.add_mutually_exclusive_group(required=False)
    flag_parser.add_argument("--train", dest="is_train", action="store_true")
    flag_parser.add_argument("--evaluate", dest="is_train", action="store_false")
    parser.set_defaults(is_train=True)
    args = parser.parse_args()
    stop = {
        "episode_reward_mean": args.stop_reward,
        "training_iteration": args.stop_iters,
    }
    env_config = {}
    env_config["security"] = "virtual_data_seq2"
    env_config["state"] = args.state
    env_config["reward"] = args.reward
    env_config["lookback"] = args.lookback
    print(env_config)
    model = TradeModel(model=args.run, env="fin_env", env_config=env_config, stop=stop, net_type=args.net_type, training=args.is_train)
    if args.is_train:
        model.train()
    else:
        model.evaluate(checkpoint=args.run+"/seql5/checkpoint_"+args.checknum+"/checkpoint-"+args.checknum)
