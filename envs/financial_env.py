import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import datetime
import os
import pandas as pd
import tqdm

ACTIONS_MEANING = {
    0: "SHORT",
    1: "NEUTRAL",
    2: "LONG",
}

FEATURES = []

VALID_SECURITIES = ['IF9999.CCFX']


class FinancialEnv(gym.Env):
    def __init__(self, config):
        # config 未设置完成，手动赋值
        self.security = 'IF9999.CCFX'
        self.start_date = datetime.datetime.strptime('2010-05-01', '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime('2020-06-28', '%Y-%m-%d')

        self._set_boundaries()
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)
        self.action_space = Discrete(3)

        self._load_data()

        self.cur_pos = 0

        self.capital_base = 100000
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base
        self.cur_return = 0

    def reset(self, by_day=True):
        """
        :param by_day:   若为True，重置时跳到下一天的开始，否则回到第一天
        :return:
        """
        if not by_day:
            self.cur_pos = 0
        else:
            self.jmp_to_next_day()
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base
        return self.ob

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        elif action == 1:
            self.cur_pos += 1
        done = 1
        return [self.cur_pos], 1.0 if done else -0.1, done, {}

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    @property
    def ob(self):
        """
        获得当前时刻的observation信息
        """
        cur_price = self.prices[self.cur_pos]

        ret = [cur_price]
        return ret

    def _set_boundaries(self):
        self.high = np.array([5] * 50)
        self.low = np.array([-5] * 50)

    def jmp_to_next_day(self):
        """
        跳转到下一天的开始
        """
        prev_idx = self.cur_pos
        while 1:
            self.cur_pos += 1
            if self.indices[prev_idx].date() != self.indices[self.cur_pos].date():
                break
            prev_idx += 1

    def update_assets(self):
        self.cur_return = round((self.assets - self.capital_base) / self.capital_base, 4)

    def _load_data(self):
        load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
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

        ok = 1 if self.security in VALID_SECURITIES else self.find_incomplete_day()
        if not ok:
            raise ValueError('原数据中有缺失数据，请确认！')
        print('共 {} 天， {} 分钟.'.format(self.total_days, self.total_minutes))
        print('读取完毕')

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
    env = FinancialEnv(1)
    for i in range(3):
        env.reset()
        print(env.cur_pos, env.ob)
