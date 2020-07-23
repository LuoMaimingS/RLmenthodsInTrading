import os
import pandas as pd
import numpy as np

"""
Rewrite DataFrame Keys: ['open', 'close', 'high', 'low', 'volume', 'money'] +       6
                        [ma_1,       ma_2,      ......,        ma_12]               12
                        [momentum_1, momentum_2, ......, momentum_12]               12
                        [obv_1,      obv_2,     ......,       obv_12]               12
                        Total                                                       42
"""


def data_rewrite(security):
    load_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    load_file = os.path.join(load_path, security + '.h5')
    if not os.path.exists(load_file):
        raise ValueError('{}，文件不存在'.format(load_file))
    raw_data = pd.read_hdf(load_file)
    indices = raw_data.index.tolist()
    rows = len(indices)
    store_indices = []
    rewrite_data = np.zeros((rows, 42))

    last_12_close = []
    last_12_obv = []
    cur_obv = 0.
    invalid_rows = 0

    cur_idx = 0
    while cur_idx < rows:
        if cur_idx % 10000 == 0:
            print("\r {} / {}".format(cur_idx, rows), end=' ')
        cur_row = raw_data.loc[indices[cur_idx]]
        if pd.isnull(cur_row[0]):
            invalid_rows += 1
            cur_idx += 1
            continue

        if indices[cur_idx].date() != indices[cur_idx - 1].date():
            if not pd.isnull(raw_data.loc[indices[cur_idx - 1]]['close']):
                last_12_close.append(raw_data.loc[indices[cur_idx - 1]]['close'])
                last_12_obv.append(cur_obv)
                cur_obv = 0
                while len(last_12_close) > 12:
                    del last_12_close[0]
                    del last_12_obv[0]
                assert len(last_12_obv) <= 12

        rewrite_idx = cur_idx - invalid_rows
        cur_obv += cur_row['volume'] * np.sign(cur_row['close'] - cur_row['open'])

        rewrite_data[rewrite_idx, 0] = cur_row['open']
        rewrite_data[rewrite_idx, 1] = cur_row['close']
        rewrite_data[rewrite_idx, 2] = cur_row['high']
        rewrite_data[rewrite_idx, 3] = cur_row['low']
        rewrite_data[rewrite_idx, 4] = cur_row['volume']
        rewrite_data[rewrite_idx, 5] = cur_row['money']

        offset = 6
        # ma_t
        for i in range(12):
            sum_p = cur_row['close']
            for j in range(1, i + 2):
                if len(last_12_close) >= j:
                    sum_p += last_12_close[-j]
                else:
                    sum_p += cur_row['close']
            rewrite_data[rewrite_idx, offset] = round(sum_p / (i + 2), 2)
            offset += 1

        # momentum_t
        for i in range(12):
            if len(last_12_close) >= (i + 1):
                momentum_i = cur_row['close'] - last_12_close[-i - 1]
            else:
                momentum_i = 0.
            rewrite_data[rewrite_idx, offset] = momentum_i
            offset += 1

        # obv_t
        for i in range(12):
            sum_obv = cur_obv
            for j in range(1, i + 2):
                if len(last_12_obv) >= j:
                    sum_obv += last_12_obv[-j]
                else:
                    sum_obv += cur_obv
            rewrite_data[rewrite_idx, offset] = round(sum_obv / (i + 2), 2)
            offset += 1

        assert offset == 42
        store_indices.append(indices[cur_idx])
        cur_idx += 1

    rewrite_data = rewrite_data[:len(store_indices)]
    print(rewrite_data.shape)

    store_columns = ['open', 'close', 'high', 'low', 'volume', 'money']
    for i in range(1, 13):
        store_columns += ['ma_%d' % i]
    for i in range(1, 13):
        store_columns += ['mom_%d' % i]
    for i in range(1, 13):
        store_columns += ['obv_%d' % i]

    df = pd.DataFrame(rewrite_data, index=store_indices, columns=store_columns)
    write_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', security[:6] + '.h5')
    df.to_hdf(write_path, key='data')


if __name__ == '__main__':
    data_rewrite('IF9999.CCFX')
