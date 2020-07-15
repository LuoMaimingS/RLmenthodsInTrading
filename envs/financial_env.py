import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import numpy as np
import datetime
import os
from copy import deepcopy
import pandas as pd

ACTIONS_MEANING = {
    0: "SHORT",
    1: "NEUTRAL",
    2: "LONG",
}

STATES = ['0', '1', '2', '3']
# 0: prev 50 minutes delta price
# 1: 14 technical indicators
# 2: 14 technical indicators with position
# 3: prev_10 minutes delta price, 14 technical indicators with position

REWARDS = ['TP', 'running_SR', 'log_return']

CHECKED_SECURITIES = ['IF9999.CCFX']


class FinancialEnv(gym.Env):
    def __init__(self, config,
                 state=None,
                 reward=None,
                 look_back=10,
                 log_return=False,
                 tax_multiple=1,
                 short_term=None,
                 long_term=None):
        # config 未设置完成，手动赋值
        self.security = 'IF9999.CCFX'
        self.start_date = datetime.datetime.strptime('2010-05-01', '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime('2020-06-28', '%Y-%m-%d')

        assert state in STATES, 'Invalid State Type, Should be one of {}'.format(STATES)
        self.state_type = state
        assert reward in REWARDS, 'Invalid Reward Type, Should be one of {}'.format(REWARDS)
        self.reward_type = reward
        self.look_back = look_back
        self._set_boundaries()
        self.observation_space = Box(low=self.low, high=self.high, dtype=np.float32)
        self.observation = []
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

        # for indicators
        self.cur_obv = 0
        self.last_obv = 0
        self.past_30_day_close = []
        self.past_30_day_obv = []             # On-Balance Volume
        self.s = [1, 2, 3] if short_term is None else short_term
        self.l = [9, 12] if long_term is None else long_term

    def reset(self, by_day=True):
        """
        :param by_day:   若为True，重置时跳到下一天的开始，否则回到第一天
        :return:
        """
        if not by_day:
            self.cur_pos = 0
        else:
            self.jmp_to_next_day()

        # for indicators
        if self.cur_pos == 0:
            self.last_obv = 0
            self.past_30_day_obv = []
            self.past_30_day_close = []

        self.observation = []
        self.trading_ticks = 0
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base

        # for SR
        self.A_n = 0.
        self.B_n = 0.

        # for indicators
        # Init OBV.
        self.cur_obv += self.bar_vol[self.cur_pos] * np.sign(self.prices[self.cur_pos] - self.bar_opens[self.cur_pos])
        return self.get_ob()

    def step(self, action, by_day=True):
        """
        根据动作，在t时刻做出行动，得到在t时刻的reward，并返回做完该动作后（即t+1时刻）的state
        :param action: 当前动作
        :param by_day: 若为True，则每日最后一个时刻done为True，且不再前进
        :return:
        """
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if isinstance(self.action_space, Discrete): action = action - 1
        done = 0
        self.trading_ticks += 1

        # Update Past 30 Day Info.
        if self.cur_pos <= 2:
            self.last_obv = 0
            self.past_30_day_obv = []
            self.past_30_day_close = []
        elif self.indices[self.cur_pos - 1].date() != self.indices[self.cur_pos].date():
            # 进入新的交易日，将昨日的相关日级指标更新，且保持追踪30天历史
            if len(self.past_30_day_close) == 30:
                del self.past_30_day_close[0]
                del self.past_30_day_obv[0]
                self.past_30_day_close.append(self.prices[self.cur_pos - 1])
                self.past_30_day_obv.append(self.last_obv)
            else:
                assert len(self.past_30_day_close) < 30
                self.past_30_day_close.append(self.prices[self.cur_pos - 1])
                self.past_30_day_obv.append(self.last_obv)

        # Done check. 当前是否为该日的结束
        if self.cur_pos >= (len(self.indices) - 2):
            done = 1
        elif by_day and self.indices[self.cur_pos].date() != self.indices[self.cur_pos + 1].date():
            done = 1

        if done:
            self.last_obv = self.cur_obv
        else:
            self.cur_obv += self.bar_vol[self.cur_pos + 1] * \
                            np.sign(self.prices[self.cur_pos + 1] - self.bar_opens[self.cur_pos + 1])

        # 仓位未变动，只需要刷新资产和收益
        if action == self.position:
            reward = self.calc_reward()
            self.update_assets()
            # self.log_info()
            self.cur_pos = self.cur_pos + 1 - done
            ob = self.get_ob()
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
        self.position = action
        reward = self.calc_reward()
        self.update_assets()

        # self.log_info()
        self.cur_pos = self.cur_pos + 1 - done
        ob = self.get_ob()
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
            if self.cur_pos != 0:
                cur_delta_price = np.log(self.prices[self.cur_pos]) - np.log(self.prices[self.cur_pos - 1])
            else:
                cur_delta_price = 0.
            self.observation.append([cur_delta_price])
            if len(self.observation) < self.look_back:
                return np.concatenate((
                    np.zeros((self.look_back - len(self.observation), 1)), np.array(self.observation)), axis=0)
            elif len(self.observation) > self.look_back:
                del self.observation[0]
                assert len(self.observation) == self.look_back
                return np.array(self.observation)
            else:
                return np.array(self.observation)
        elif self.state_type == '1' or self.state_type == '2' or self.state_type == '3':
            """
            14 Technical Indicators Analysed In -- C. J. Neely, D. E. Rapach, J. Tu, and G. Zhou,
            “Forecasting the equity risk premium: The role of technical indicators,”
            Manage. Sci., vol. 60, no. 7, pp. 1772–1791, 2014.
            """
            signals = []

            # MOVING AVERAGE 长短期信号，对应的短期指标超越对应的长期指标时，出现1
            ma_sl = []
            sl_indices = self.s + self.l
            for sl in sl_indices:
                sigma = self.prices[self.cur_pos]
                for k in range(sl - 1):
                    if len(self.past_30_day_close) >= (k + 1):
                        sigma += self.past_30_day_close[-k - 1]
                    else:
                        sigma += self.prices[self.cur_pos]
                ma_sl.append(sigma / sl)
            ma_s = ma_sl[:len(self.s)]
            ma_l = ma_sl[len(self.s):]
            ma_signals = []
            for ma_s_val in ma_s:
                for ma_l_val in ma_l:
                    ma_signals.append(int(ma_s_val > ma_l_val))
            signals += ma_signals

            # Momentum 信号
            momentum = []
            for l in self.l:
                if l == 0 or l > len(self.past_30_day_close):
                    momentum.append(0)
                else:
                    momentum.append(int(self.prices[self.cur_pos] > self.past_30_day_close[-l]))
            signals += momentum

            # On-Balance Volume 长短期信号
            ma_obv_sl = []
            sl_indices = self.s + self.l
            for sl in sl_indices:
                sigma = self.cur_obv
                for k in range(sl - 1):
                    if len(self.past_30_day_obv) >= (k + 1):
                        sigma += self.past_30_day_obv[-k - 1]
                    else:
                        sigma += self.cur_obv
                ma_obv_sl.append(sigma / sl)
            ma_obv_s = ma_obv_sl[:len(self.s)]
            ma_obv_l = ma_obv_sl[len(self.s):]
            ma_obv_signals = []
            for ma_obv_s_val in ma_obv_s:
                for ma_obv_l_val in ma_obv_l:
                    ma_obv_signals.append(int(ma_obv_s_val > ma_obv_l_val))
            signals += ma_obv_signals

            if self.state_type == '1':
                self.observation.append(deepcopy(signals))
                if len(self.observation) < self.look_back:
                    return np.concatenate((
                        np.zeros((self.look_back - len(self.observation), 14)), np.array(self.observation)), axis=0)
                elif len(self.observation) > self.look_back:
                    del self.observation[0]
                    assert len(self.observation) == self.look_back
                    return np.array(self.observation)
                else:
                    return np.array(self.observation)

            elif self.state_type == '2':
                self.observation.append(signals + [self.position])
                if len(self.observation) < self.look_back:
                    return np.concatenate((
                        np.zeros((self.look_back - len(self.observation), 15)), np.array(self.observation)), axis=0)
                elif len(self.observation) > self.look_back:
                    del self.observation[0]
                    assert len(self.observation) == self.look_back
                    return np.array(self.observation)
                else:
                    return np.array(self.observation)

            elif self.state_type == '3':
                if self.cur_pos != 0:
                    cur_delta_price = np.log(self.prices[self.cur_pos]) - np.log(self.prices[self.cur_pos - 1])
                else:
                    cur_delta_price = 0.
                self.observation.append([cur_delta_price] + signals + [self.position])
                if len(self.observation) < self.look_back:
                    return np.concatenate((
                        np.zeros((self.look_back - len(self.observation), 16)), np.array(self.observation)), axis=0)
                elif len(self.observation) > self.look_back:
                    del self.observation[0]
                    assert len(self.observation) == self.look_back
                    return np.array(self.observation)
                else:
                    return np.array(self.observation)
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
                k_n = np.sqrt(n / (n - 1))
                sr_n = self.A_n / (k_n * np.sqrt(self.B_n - np.square(self.A_n)))
            return sr_n
        else:
            raise NotImplementedError

    def _set_boundaries(self):
        if self.state_type == '0':
            self.high = np.array([[5] * 1] * self.look_back)
            self.low = np.array([[-5] * 1] * self.look_back)
        elif self.state_type == '1':
            self.high = np.array([[1] * 14] * self.look_back)
            self.low = np.array([[0] * 14] * self.look_back)
        elif self.state_type == '2':
            self.high = np.array([[1] * 14 + [1]] * self.look_back)
            self.low = np.array([[0] * 14 + [-1]] * self.look_back)
        elif self.state_type == '3':
            self.high = np.array([[5] + [1] * 14 + [1]] * self.look_back)
            self.low = np.array([[-5] + [0] * 14 + [-1]] * self.look_back)
        else:
            raise NotImplementedError

    def jmp_to_next_day(self):
        """
        跳转到下一天的开始
        """
        prev_idx = self.cur_pos
        while 1:
            if self.cur_pos >= len(self.indices) - 1 or self.cur_pos == 0:
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

        self.bar_opens = list(raw_data.loc[self.start_date:self.end_date]['open'])
        self.bar_vol = list(raw_data.loc[self.start_date:self.end_date]['volume'])
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
    env.reset()
    rwd = 0
    while True:
        import random
        ac = np.random.randint(0, 2)
        ob, r, done, info = env.step(ac)
        # print(env.assets, env.cur_pos, r)
        rwd += r
        if done:
            print(env.assets, env.cur_pos, rwd)
            rwd = 0
            env.reset()
        # print(r)
