# RLmenthodsInTrading
该项目基于不同的强化学习算法，定义了数种State和Reward，以研究在中国金融市场中哪些组合可以获得比较高的收益。
## Release 1.1
加入look back参数，默认为10，控制某个时刻向前看的长度
```
env = FinancialEnv(config, state='3', reward='running_SR', look_back=10)
```
影响返回的observation，shape分别为（look_back, 1） / （look_back, 14） / （look_back, 15） /（look_back, 16）。


## Release 1.0
### 环境参数
```
env = FinancialEnv(config, state='3', reward='running_SR')
```
config还未整合，只是个占位符，留待实现，目前对环境没有任何影响
#### State类别
##### 0
50维。前50分钟的log价差，可以隔日，基于Deng的那篇文章，但是没有加入Momentum

Citation：Yue Deng, Feng Bao, Zhiquan Ren, Youyong Kong, Qionghai Dai. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE Trans. on Neural Networks and Learning Systems. 28(3):653-664, 2017.
##### 1
14维。选取了文章中所分析的14个技术指标形成的交易信号，如短期价格（或OBV）均线穿过长期均线，或momentum为正，形成一个交易信号，相应维度为1，无信号时为0。

Citation：C. J. Neely, D. E. Rapach, J. Tu, and G. Zhou, “Forecasting the equity
risk premium: The role of technical indicators,” Manage. Sci., vol. 60,
no. 7, pp. 1772–1791, 2014.
##### 2
15维。1中14维的交易信号加上agent的当前持仓状态δ，δ ∈ {-1, 0, 1}。
##### 3
25维。前10分钟的log价差，14维的交易信号和agent的当前持仓状态。


#### Reward类别
##### TP
Total Profit，同样是Deng的文章中使用过的。做出动作后并得到交易后，agent的资产与上一时刻的差值。
##### log_return
做出动作后并得到交易后，agent与上个时刻相比的收益情况，使用对数收益率，以满足reward可线性相加。
##### running sharp ratio
增量式更新的sharpe ratio，不用维护冗长的收益数组，计算便捷。

Citation：Moody J , Wu L , Liao Y , et al. Performance functions and reinforcement learning for trading systems and portfolios[J]. Journal of Forecasting, 1998, 17(5-6):441-470.

### 环境运行示例
```angular2html
ob, r, done, info = env.step(ac)

|-------|-------|
|       |       |
|       |       |
|       |       |
9:16   9:17    9:18

# agent做决策的时间点为每个分钟bar的末端，如当前是9:17，在该分钟结束时做出ac。
# 假设会立即交易，得到立即的reward，然后环境向前运行1分钟，返回9:18分钟末的state。
```