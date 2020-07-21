# RLmenthodsInTrading
该项目基于不同的强化学习算法，定义了数种State和Reward，以研究在中国金融市场中哪些组合可以获得比较高的收益。

## Release 1.3
经过对一些文章和代码的分析，发现在当前bar的结束时刻做action，reward的计算有两种方式。
* 该action的税费 + 上一个时刻action在当前bar的收益变动，即

  R<sub>t</sub> = ac<sub>t-1</sub> * (p<sub>t</sub> - p<sub>t-1</sub>) - c * ac<sub>t</sub>
* 该action的税费 + 该action在下一个bar的收益变动，即

  R<sub>t</sub> = ac<sub>t</sub> * (p<sub>t+1</sub> - p<sub>t</sub>) - c * ac<sub>t</sub>

而observation基本是统一的，都会返回下一个bar结束时刻的observation。

因此，在环境中加入了foward_reward参数，默认为False，对应第一种reward计算方式；为True时则对应第二种。

此外，在state中添加了一维，表示当前如果有持仓，平仓后的收益率。

## Release 1.2
加入cheat state (state = '78')，以验证各模块有效性。

## Release 1.1
加入look back参数，默认为10，控制某个时刻向前看的长度
```
env = FinancialEnv(state='3', reward='running_SR', look_back=10)
ob, r, done, info = env.step(ac, by_day=False)
```
影响返回的observation，shape分别为（look_back, 1） / （look_back, 14） / （look_back, 16） /（look_back, 17）。

回测时将by_day设为False，即会一直连续交易，与一般回测流程保持一致。

消除了一些特殊情形下，浮点数判断大小中精度误差导致的信号计算错误问题。

## Release 1.0
### 环境参数
```
env = FinancialEnv(state='3', reward='running_SR')
```
#### State类别
##### 0
维度：(n, 1)：前n分钟的log价差，可以隔日，基于Deng的那篇文章，但是没有加入Momentum

Citation：Yue Deng, Feng Bao, Zhiquan Ren, Youyong Kong, Qionghai Dai. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE Trans. on Neural Networks and Learning Systems. 28(3):653-664, 2017.
##### 1
维度：(n, 14)：选取了文章中所分析的14个技术指标形成的交易信号，如短期价格（或OBV）均线穿过长期均线，或momentum为正，形成一个交易信号，相应维度为1，无信号时为0。

Citation：C. J. Neely, D. E. Rapach, J. Tu, and G. Zhou, “Forecasting the equity
risk premium: The role of technical indicators,” Manage. Sci., vol. 60,
no. 7, pp. 1772–1791, 2014.
##### 2
维度：(n, 16)：1中14维的交易信号加上agent的当前持仓状态δ，δ ∈ {-1, 0, 1}，以及将当前持仓平仓后的收益率。
##### 3
维度：(n, 17)：前n分钟的log价差，14维的交易信号，agent的当前持仓状态和平仓后的收益率。


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