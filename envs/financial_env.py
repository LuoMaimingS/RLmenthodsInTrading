import gym
from gym.spaces import Discrete, Box
from gym.utils import seeding
import datetime
import enum
import random

from envs.data_rewrite import *
from envs.env_utils import *


class Actions(enum.Enum):
    Short = 0
    Neutral = 1
    Long = 2


"""
States types and shape, *n* refers to parameter *look back*.
0: previous n minutes delta price && position && close_return, seq: (n + 2, ) conv_1d: (n, 3)
1: 14 technical indicators, seq: (n * 14, ) conv_1d: (n, 14)
2: 14 technical indicators && position && close_return, seq: (n * 14 + 2, ) conv_1d: (n, 16)
3: previous n minutes delta price, 14 technical indicators && position && close_return, seq: (n * 15 + 2) conv_1d: (n, 17)
78:cheat states, next n minutes delta price, seq: (n, ) conv_1d: (n, 1)
"""
STATES = ['0', '1', '2', '3', '78']


"""
Rewards types and calculations
TP: Total Profit
earning rate: Total earning rate
log_return: Log earning rate
running_SR: running sharp ratio
"""
REWARDS = ['TP', 'earning_rate', 'log_return', 'running_SR']

"""
IF9999.CCFX: 在2016年1月1日前（不包括）每天有 270 个 bar，之后每天有 240 个 bar
"""
CHECKED_SECURITIES = ['IF9999.CCFX', 'virtual_data']


class FinancialEnv(gym.Env):
    def __init__(self,
                 security='IF9999.CCFX',
                 state='1',
                 state_dims=1,
                 reward='TP',
                 forward_reward=False,
                 delayed_reward=False,
                 look_back=50,
                 log_return=False,
                 tax_multiple=1,
                 short_term=None,
                 long_term=None,
                 shuffle_reset=False):
        self.security = security
        self.start_date = datetime.datetime.strptime('2013-08-01', '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime('2015-02-01', '%Y-%m-%d')
        """
        self.start_date = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime('2020-05-31', '%Y-%m-%d')
        """
        assert state in STATES, 'Invalid State Type, Should be one of {}'.format(STATES)
        self.state_type = state
        self.state_dims = state_dims
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
        self.prev_running_sr = 0.

        # for indicators
        self.s = [1, 2, 3] if short_term is None else short_term
        self.l = [9, 12] if long_term is None else long_term

        # for different mode.
        self.forward_r = forward_reward
        self.delayed_reward = delayed_reward
        self.cost_price = 0.
        self.shuffle_reset = shuffle_reset

    def reset(self, by_day=True):
        """
        :param by_day:   若为True，重置时跳到下一天的开始，否则回到第一天
        :return:
        """
        if not by_day:
            self.cur_pos = 0
        else:
            self.jmp_to_next_day(shuffle=self.shuffle_reset)

        self.observation = []
        self.trading_ticks = 0
        self.cash = self.capital_base
        self.position = 0
        self.shares = 0
        self.assets = self.capital_base

        # for SR
        self.A_n = 0.
        self.B_n = 0.
        self.prev_running_sr = 0.
        # self.log_info()

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
        reward = 0.
        self.trading_ticks += 1

        # Done check. 当前是否为该日的结束
        if self.cur_pos >= (len(self.indices) - 2):
            done = 1
        else:
            if self.security.startswith('virtual'):
                if self.cur_pos % 240 == 239:
                    done = 1
            else:
                if by_day and self.indices[self.cur_pos].date() != self.indices[self.cur_pos + 1].date():
                    done = 1

        # 仓位未变动，只需要刷新资产和收益
        if action == self.position:
            if not self.forward_r: reward = self.calc_reward()
            self.update_assets()
            # self.log_info()
            self.cur_pos = self.cur_pos + 1 - done
            ob = self.get_ob()
            if self.forward_r: reward = self.calc_reward()
            return ob, reward, done, {'return': self.cur_return}

        # 仓位变化，重计算损益
        cur_price = self.prices[self.cur_pos]
        self.cost_price = cur_price
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
        if not self.forward_r: reward = self.calc_reward()
        self.update_assets()

        # self.log_info()
        self.cur_pos = self.cur_pos + 1 - done
        ob = self.get_ob()
        if self.forward_r: reward = self.calc_reward()
        return ob, reward, done, {'return': self.cur_return}

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def get_ob(self):
        """
        获得当前时刻的observation信息
        States types and shape, *n* refers to parameter *look back*.
        0: previous n minutes delta log price && position && close_return, seq: (n + 2, ) conv_1d: (n, 3)
        1: 14 technical indicators, seq: (n * 14, ) conv_1d: (n, 14)
        2: 14 technical indicators && position && close_return, seq: (n * 14 + 2, ) conv_1d: (n, 16)
        3: previous n minutes delta log price, 14 technical indicators && position && close_return, seq: (n * 15 + 2) conv_1d: (n, 17)
        """
        next_ret = (self.cash + self.shares * self.prices[self.cur_pos] - self.capital_base) / self.capital_base
        next_ret = next_ret * (self.position != 0)
        if self.state_type == '0':
            # 价差
            start_idx = max(0, self.cur_pos - self.look_back)
            log_prices = np.log(self.prices[start_idx:self.cur_pos + 1])
            n = len(log_prices) - 1
            if self.state_dims == 1:
                cur_ob = np.zeros((self.look_back + 2, ))
                cur_ob[-2 - n:-2] = np.diff(log_prices)
                cur_ob[-2] = self.position
                cur_ob[-1] = next_ret
                return cur_ob
            elif self.state_dims == 2:
                cur_ob = np.zeros((self.look_back, 3))
                if n != 0: cur_ob[-n:, 0] = np.diff(log_prices)
                cur_ob[:, 1] = self.position
                cur_ob[:, 2] = next_ret
                return cur_ob

        elif self.state_type == '1' or self.state_type == '2' or self.state_type == '3':
            """
            14 Technical Indicators Analysed In -- C. J. Neely, D. E. Rapach, J. Tu, and G. Zhou,
            “Forecasting the equity risk premium: The role of technical indicators,”
            Manage. Sci., vol. 60, no. 7, pp. 1772–1791, 2014.
            """
            cur_row = self.data.loc[self.indices[self.cur_pos]]
            signals = []

            # MOVING AVERAGE 长短期信号，对应的短期指标超越对应的长期指标时，出现1
            for s in self.s:
                for l in self.l:
                    ma_s = cur_row['ma_%d' % s]
                    ma_l = cur_row['ma_%d' % l]
                    signals.append(int((ma_s - ma_l) > 1e-6))

            # Momentum 信号
            for l in self.l:
                mom_l = cur_row['mom_%d' % l]
                signals.append(int(mom_l > 1e-6))

            # On-Balance Volume 长短期信号
            for s in self.s:
                for l in self.l:
                    obv_s = cur_row['obv_%d' % s]
                    obv_l = cur_row['obv_%d' % l]
                    signals.append(int((obv_s - obv_l) > 1e-6))
            self.observation.append(signals)
            n = len(self.observation)
            if n > self.look_back: del self.observation[0]

            if self.state_type == '1':
                if self.state_dims == 1:
                    cur_ob = np.zeros((self.look_back * 14,))
                    flatten_ob = np.array(self.observation).reshape(-1, )
                    cur_ob[-n * 14:] = flatten_ob
                    return cur_ob
                elif self.state_dims == 2:
                    cur_ob = np.zeros((self.look_back, 14))
                    cur_ob[-n:, :] = np.array(self.observation)
                    return cur_ob

            elif self.state_type == '2':
                if self.state_dims == 1:
                    cur_ob = np.zeros((self.look_back * 14 + 2,))
                    flatten_ob = np.array(self.observation).reshape(-1, )
                    cur_ob[-n * 14 - 2:-2] = flatten_ob
                    cur_ob[-2] = self.position
                    cur_ob[-1] = next_ret
                    return cur_ob
                elif self.state_dims == 2:
                    cur_ob = np.zeros((self.look_back, 16))
                    cur_ob[-n:, :-2] = np.array(self.observation)
                    cur_ob[:, -2] = self.position
                    cur_ob[:, -1] = next_ret
                    return cur_ob

            elif self.state_type == '3':
                if self.cur_pos != 0:
                    self.observation[-1].insert(0, np.log(self.prices[self.cur_pos]) - np.log(self.prices[self.cur_pos - 1]))
                else:
                    self.observation[-1].insert(0, 0.)
                if self.state_dims == 1:
                    cur_ob = np.zeros((self.look_back * 15 + 2,))
                    flatten_ob = np.array(self.observation).reshape(-1, )
                    cur_ob[-n * 15 - 2:-2] = flatten_ob
                    cur_ob[-2] = self.position
                    cur_ob[-1] = next_ret
                    return cur_ob
                elif self.state_dims == 2:
                    cur_ob = np.zeros((self.look_back, 17))
                    cur_ob[-n:, :-2] = np.array(self.observation)
                    cur_ob[:, -2] = self.position
                    cur_ob[:, -1] = next_ret
                    return cur_ob

        elif self.state_type == '78':
            # cheat 价差
            if self.cur_pos <= len(self.indices) - self.look_back - 2:
                next_n_prices = np.log(self.prices[self.cur_pos:self.cur_pos + self.look_back + 1])
                # next_n_prices = self.prices[self.cur_pos:self.cur_pos + self.look_back + 1]
                next_delta_n_prices = np.diff(next_n_prices)
                return np.reshape(next_delta_n_prices, (self.look_back, 1))
            else:
                return np.zeros((self.look_back, 1))

        else:
            raise NotImplementedError

    def calc_reward(self):
        """
        获得当前时刻的reward信息
        """
        if not self.delayed_reward:
            cur_assets = self.cash + self.shares * self.prices[self.cur_pos]
        else:
            cur_assets = self.cash + self.shares * self.cost_price

        if self.reward_type == 'TP':
            tp_reward = cur_assets - self.assets
            return tp_reward
        elif self.reward_type == 'earning_rate':
            er = (cur_assets - self.assets) / self.capital_base * 100
            return er
        elif self.reward_type == 'log_return':
            origin_return = (cur_assets - self.assets) / self.assets
            log_return = np.log(1 + origin_return)
            return log_return
        elif self.reward_type == 'running_SR':
            n = self.trading_ticks
            r_n = (cur_assets - self.assets) / self.assets
            self.A_n = (1 / n) * r_n + (n - 1) / n * self.A_n
            self.B_n = (1 / n) * np.square(r_n) + (n - 1) / n * self.B_n
            if self.trading_ticks == 1 or self.A_n == 0:
                sr_n = 0
            else:
                k_n = np.sqrt(n / (n - 1))
                sr_n = self.A_n / (k_n * np.sqrt(self.B_n - np.square(self.A_n)))
            delta_sr = sr_n - self.prev_running_sr
            self.prev_running_sr = sr_n
            return delta_sr
        else:
            raise NotImplementedError

    def _set_boundaries(self):
        """
        States types and shape, *n* refers to parameter *look back*.
        0: previous n minutes delta log price && position && close_return, seq: (n + 2, ) conv_1d: (n, 3)
        1: 14 technical indicators, seq: (n * 14, ) conv_1d: (n, 14)
        2: 14 technical indicators && position && close_return, seq: (n * 14 + 2, ) conv_1d: (n, 16)
        3: previous n minutes delta log price, 14 technical indicators && position && close_return, seq: (n * 15 + 2) conv_1d: (n, 17)
        """
        n = self.look_back
        if self.state_type == '0':
            if self.state_dims == 1:
                self.high = np.array([5] * (n + 2))
                self.low = np.array([-5] * (n + 2))
            elif self.state_dims == 2:
                self.high = np.array([[5] * 3] * n)
                self.low = np.array([[-5] * 3] * n)
        elif self.state_type == '1':
            if self.state_dims == 1:
                self.high = np.array([5] * (n * 14))
                self.low = np.array([-5] * (n * 14))
            elif self.state_dims == 2:
                self.high = np.array([[5] * 14] * n)
                self.low = np.array([[-5] * 14] * n)
        elif self.state_type == '2':
            if self.state_dims == 1:
                self.high = np.array([5] * (n * 14 + 2))
                self.low = np.array([-5] * (n * 14 + 2))
            elif self.state_dims == 2:
                self.high = np.array([[5] * 16] * n)
                self.low = np.array([[-5] * 16] * n)
        elif self.state_type == '3':
            if self.state_dims == 1:
                self.high = np.array([5] * (n * 15 + 2))
                self.low = np.array([-5] * (n * 15 + 2))
            elif self.state_dims == 2:
                self.high = np.array([[5] * 17] * n)
                self.low = np.array([[-5] * 17] * n)
        elif self.state_type == '78':
            if self.state_dims == 1:
                self.high = np.array([5] * n)
                self.low = np.array([-5] * n)
            elif self.state_dims == 2:
                self.high = np.array([[5]] * n)
                self.low = np.array([[-5]] * n)
        else:
            raise NotImplementedError
        print('Observation Boundary Set, shape: {}'.format(self.high.shape))

    def jmp_to_next_day(self, shuffle=False):
        """
        跳转到下一天的开始
        :param shuffle: 是否使用随机跳转
        :return: 跳转后的observation
        """
        if shuffle:
            self.cur_pos = random.randint(0, self.total_minutes - 1)
        while 1:
            if self.cur_pos >= len(self.indices) - 1 or self.cur_pos == 0:
                self.cur_pos = 0
                break
            self.cur_pos += 1
            if self.security.startswith('virtual'):
                if self.cur_pos % 240 == 0:
                    break
            else:
                if self.indices[self.cur_pos - 1].date() != self.indices[self.cur_pos].date():
                    break
        return self.get_ob()

    def update_assets(self):
        if not self.delayed_reward:
            self.assets = self.cash + self.shares * self.prices[self.cur_pos]
        else:
            # 延迟reward，只有当仓位变动时，资产才会变动
            self.assets = self.cash + self.shares * self.cost_price

        if self.log_return:
            self.cur_return = round(np.log(self.assets / self.capital_base), 4)
        else:
            self.cur_return = round(((self.assets - self.capital_base) / self.capital_base), 4)

    def _load_data(self):
        load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        if len(self.security) > 6 and self.security[:7] != 'virtual':
            if not os.path.exists(os.path.join(load_path, self.security[:6] + '.h5')):
                print('Version Updated, Needing ReWrite Data.')
                data_rewrite(self.security)
            self.security = self.security[:6]

        load_file = os.path.join(load_path, self.security + '.h5')
        if not os.path.exists(load_file):
            raise ValueError('{}，文件不存在'.format(load_file))

        print("读取 {} 数据中......".format(self.security), end='  ')
        raw_data = pd.read_hdf(load_file)
        if self.security.startswith('virtual'):
            self.data = raw_data
            self.prices = list(raw_data['close'])
        else:
            self.data = raw_data.loc[self.start_date:self.end_date]
            self.prices = list(raw_data.loc[self.start_date:self.end_date]['close'])
        self.norm_data = MinMaxScaler(self.data)

        self.indices = self.data.index.tolist()
        self.total_minutes = len(self.indices)
        self.total_days = int(self.total_minutes / 270)
        self.base_price = self.data.loc[self.indices[0]]['open']

        ok = 1 if self.security in CHECKED_SECURITIES or self.security.startswith('virtual') else self.find_incomplete_day()
        if not ok:
            raise ValueError('原数据中有缺失数据，请确认！')
        print('共 {} 天， {} 分钟.'.format(self.total_days, self.total_minutes))
        print('读取完毕')

    def _get_tax_rate(self):
        if self.security in ['IF9999', 'IF9999.CCFX']:
            self.buy_rate = self.sell_rate = 0.000023
            # self.buy_rate = self.sell_rate = 0.000345
        elif self.security.startswith('virtual'):
            self.buy_rate = self.sell_rate = 0.000023
        else:
            raise NotImplementedError

    def log_info(self):
        print('{}th {} price: {:.2f} position:{} cash: {:.2f} shares: {:.2f} assets: {:.2f}'.format(
            self.cur_pos, self.indices[self.cur_pos], self.prices[self.cur_pos],
            self.position, self.cash, self.shares, self.assets))

    def log_info_after_action(self):
        print('{}th {} price: {:.2f} -> {} current position:{} cash: {:.2f} shares: {:.2f} assets: {:.2f}'.format(
            self.cur_pos, self.indices[self.cur_pos], self.prices[self.cur_pos - 1], self.prices[self.cur_pos],
            self.position, self.cash, self.shares, self.assets))

    def find_incomplete_day(self, bars_per_day=270):
        prev_idx = 0
        i = 0
        while i < len(self.indices):
            if self.indices[prev_idx].date() != self.indices[i].date():
                this_day_minutes = i - prev_idx
                prev_idx = i
                if this_day_minutes != bars_per_day:
                    print('{} does not has {} minutes.'.format(self.indices[i].date(), bars_per_day))
                    return False
            i += 1
            if self.indices[prev_idx].date().year >= 2016 and bars_per_day == 270 and self.security == 'IF9999':
                bars_per_day = 240
        if i - prev_idx != bars_per_day:
            print('{} does not has {} minutes.'.format(self.indices[i - 1].date(), bars_per_day))
            return False
        print('{} Checked, *** PLEASE ADD IT TO CHECKED_SECURITIES ***'.format(self.security))
        return True

    def get_batch_data(self, batch_size, seq_len=30, overlap=True):
        batch_data = list()
        random_idx = np.random.permutation(self.total_minutes) if overlap else \
            np.random.permutation(self.total_minutes) // seq_len
        chosen_idx = random_idx[:batch_size]
        keys = ['open', 'high', 'low', 'close', 'volume']
        for i in range(batch_size):
            if overlap:
                temp_seq = self.norm_data.loc[self.indices[chosen_idx[i]: chosen_idx[i] + seq_len]][keys]
            else:
                temp_seq = self.norm_data.loc[self.indices[chosen_idx[i]*seq_len: (chosen_idx[i] + 1)*seq_len]][keys]
            batch_data.append(temp_seq.values.tolist())
        return batch_data


if __name__ == '__main__':
    env = FinancialEnv(security='virtual_data', state='3', state_dims=1, reward='TP', look_back=3, delayed_reward=False)
    ob = env.reset()
    rwd = 0
    # print(env.get_batch_data(10))
    # print(env.cur_pos, env.indices[env.cur_pos], env.prices[env.cur_pos])
    print(ob.reshape(1, -1))
    while True:
        import random
        ac = random.randint(0, 2)
        # ac = 2
        # print(ob, env.indices[env.cur_pos], env.prices[env.cur_pos])
        ob, r, done, info = env.step(ac)
        # print(r)

        rwd += r
        if done:
            print(env.assets, env.cur_pos, env.indices[env.cur_pos], rwd, env.prices[env.cur_pos])
            rwd = 0
            env.reset()