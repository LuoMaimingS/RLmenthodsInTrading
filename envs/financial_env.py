import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import datetime
import os
import pandas as pd

ACTIONS_MEANING = {
    0: "SHORT",
    1: "NEUTRAL",
    2: "LONG",
}

STATES = ['0', '1']
# 0: prev 50 minutes delta price

REWARDS = ['TP', 'running_SR', 'log_return']
FEATURES = []

CHECKED_SECURITIES = ['IF9999.CCFX']


class FinancialEnv(gym.Env):
    def __init__(self, config,
                 state=None,
                 reward=None,
                 log_return=False,
                 tax_multiple=1):
        # config 未设置完成，手动赋值
        self.security = 'IF9999.CCFX'
        self.start_date = datetime.datetime.strptime('2010-05-01', '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime('2020-06-28', '%Y-%m-%d')

        assert state in STATES, 'Invalid State Type, Should be one of {}'.format(STATES)
        self.state_type = state
        assert reward in REWARDS, 'Invalid Reward Type, Should be one of {}'.format(REWARDS)
        self.reward_type = reward
        self._set_boundaries()
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = Discrete(3)

        self.log_return = log_return
        self.tax_multi = tax_multiple

        self._load_data()
        self._get_tax_rate()

        self.cur_pos = 0
        self.trading_ticks = 0

        self.capital_base = 100000
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base
        self.cur_return = 0.

        # for SR
        self.A_n = 0.
        self.B_n = 0.

    def reset(self, by_day=True):
        """
        :param by_day:   若为True，重置时跳到下一天的开始，否则回到第一天
        :return:
        """
        if not by_day:
            self.cur_pos = 0
        else:
            self.jmp_to_next_day()
        self.trading_ticks = 0
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base

        # for SR
        self.A_n = 0.
        self.B_n = 0.
        return self.get_ob()

    def step(self, action, by_day=True):
        """
        :param action: 当前动作
        :param by_day: 若为True，则每日最后一个时刻done为True，且不再前进
        :return:
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if isinstance(self.action_space, Discrete): action = action - 1
        done = 0
        if self.cur_pos >= (len(self.indices) - 2):
            done = 1
        elif by_day and self.indices[self.cur_pos].date() != self.indices[self.cur_pos + 1].date():
            done = 1

        self.trading_ticks += 1

        # 仓位未变动，只需要刷新资产和收益
        if action == self.position:
            ob = self.get_ob()
            reward = self.calc_reward()
            self.update_assets()
            # self.log_info()
            # print('reward: {:.2f} done: {}'.format(reward, done))
            self.cur_pos = self.cur_pos + 1 - done
            return ob, reward, done, {'return': self.cur_return}

        # 仓位变化，重计算损益
        cur_price = self.prices[self.cur_pos]
        new_shares = int(self.assets * action / (1.05 * cur_price))
        origin_shares = self.shares
        if new_shares > origin_shares:
            # 买入
            self.cash = self.cash - abs(new_shares - self.shares) * cur_price * (1 + self.buy_rate * self.tax_multi)
        else:
            # 卖出
            self.cash = self.cash + abs(new_shares - self.shares) * cur_price * (1 - self.sell_rate * self.tax_multi)
        self.shares = new_shares
        ob = self.get_ob()
        self.position = action
        reward = self.calc_reward()
        self.update_assets()

        # self.log_info()
        self.cur_pos = self.cur_pos + 1 - done
        return ob, reward, done, {'return': self.cur_return}

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def get_ob(self):
        """
        获得当前时刻的observation信息
        """
        if self.state_type == '0':
            # 前50分钟价差
            head_idx = max(0, self.cur_pos - 50)
            prev_51_prices = np.log(self.prices[head_idx:self.cur_pos + 1])
            delta_50min_prices = np.diff(prev_51_prices)
            if delta_50min_prices.shape[0] != 50:
                delta_50min_prices = np.concatenate(
                    (np.zeros((50 - delta_50min_prices.shape[0],)), delta_50min_prices), axis=0)
            return delta_50min_prices
        else:
            raise NotImplementedError

    def calc_reward(self):
        """
        获得当前时刻的reward信息
        """
        if self.reward_type == 'TP':
            tp_reward = self.cash + self.shares * self.prices[self.cur_pos] - self.assets
            return tp_reward
        elif self.reward_type == 'log_return':
            cur_assets = self.cash + self.shares * self.prices[self.cur_pos]
            origin_return = (cur_assets - self.assets) / self.assets
            log_return = np.log(1 + origin_return)
            return log_return
        elif self.reward_type == 'running_SR':
            n = self.trading_ticks
            r_n = (self.cash + self.shares * self.prices[self.cur_pos] - self.assets) / self.assets
            self.A_n = (1 / n) * r_n + (n - 1) / n * self.A_n
            self.B_n = (1 / n) * np.square(r_n) + (n - 1) / n * self.B_n
            # print(self.A_n, self.B_n, self.trading_ticks)
            if self.trading_ticks == 1 or self.A_n == 0:
                sr_n = 0
            else:
                sr_n = self.A_n / np.sqrt(self.B_n - np.square(self.A_n))
            return sr_n
        else:
            raise NotImplementedError

    def _set_boundaries(self):
        if self.state_type == '0':
            self.high = np.array([5] * 50)
            self.low = np.array([-5] * 50)
        else:
            raise NotImplementedError

    def jmp_to_next_day(self):
        """
        跳转到下一天的开始
        """
        prev_idx = self.cur_pos
        while 1:
            if self.cur_pos >= len(self.indices) - 1:
                self.cur_pos = 0
                break
            self.cur_pos += 1
            if self.indices[prev_idx].date() != self.indices[self.cur_pos].date():
                break
            prev_idx += 1

    def update_assets(self):
        self.assets = self.cash + self.shares * self.prices[self.cur_pos]
        if self.log_return:
            self.cur_return = round(np.log(self.assets / self.capital_base), 4)
        else:
            self.cur_return = round(((self.assets - self.capital_base) / self.capital_base), 4)

    def _load_data(self):
        load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        load_file = os.path.join(load_path, self.security + '.h5')
        if not os.path.exists(load_file):
            raise ValueError('{}，文件不存在'.format(load_file))

        print("读取 {} 数据中......".format(self.security), end='  ')
        raw_data = pd.read_hdf(load_file)
        self.data = raw_data.loc[self.start_date:self.end_date]

        self.prices = list(raw_data.loc[self.start_date:self.end_date]['close'])
        self.indices = self.data.index.tolist()
        self.total_minutes = len(self.indices)
        self.total_days = int(self.total_minutes / 240)
        self.base_price = self.data.loc[self.indices[0]]['open']

        ok = 1 if self.security in CHECKED_SECURITIES else self.find_incomplete_day()
        if not ok:
            raise ValueError('原数据中有缺失数据，请确认！')
        print('共 {} 天， {} 分钟.'.format(self.total_days, self.total_minutes))
        print('读取完毕')

    def _get_tax_rate(self):
        if self.security in ['IF9999.CCFX']:
            self.buy_rate = self.sell_rate = 0.000025
        else:
            raise NotImplementedError

    def log_info(self):
        print('{}th {} price: {:.2f} position:{} cash: {:.2f} shares: {:.2f} assets: {:.2f}'.format(
            self.cur_pos, self.indices[self.cur_pos], self.prices[self.cur_pos],
            self.position, self.cash, self.shares, self.assets))

    def find_incomplete_day(self):
        prev_idx = 0
        i = 0
        while i < len(self.indices):
            if self.indices[prev_idx].date() != self.indices[i].date():
                print(prev_idx, i, self.indices[prev_idx], self.indices[i])
                this_day_minutes = i - prev_idx
                prev_idx = i
                if this_day_minutes != 240:
                    print('{} does not has 240 minutes.'.format(self.indices[i].date()))
                    return False
            i += 1
        if i - prev_idx + 1 != 240:
            print('{} does not has 240 minutes.'.format(self.indices[i].date()))
            return False
        return True


if __name__ == '__main__':
    env = FinancialEnv(1, state='0', reward='running_SR')
    rwd = 0
    while True:
        import random
        ac = 2
        ob, r, done, _ = env.step(ac)
        # print(env.assets, env.cur_pos, r)
        rwd += r
        if done:
            print(env.assets, env.cur_pos, rwd)
            rwd = 0
            env.reset()
        # print(r)
