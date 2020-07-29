# RLmenthodsInTrading
该项目基于不同的强化学习算法，定义了数种State和Reward，以研究在中国金融市场中哪些组合可以获得比较高的收益。

## Release 2.0
env中加入delayed_reward参数，为True只有在交易实现时才会有reward，是稀疏的reward实现；为False则同原来一样，不影响。

## Release 2.0
优化了运行速度，稍微改变了state的维度，所有state设计下，都可以随机reset。初次加载会自动重写数据
```angular2html
env = FinancialEnv(state='2', state_dims=1, reward='TP', look_back=1, tax_multiple=1)
```
示例参数分别为：state的类别，返回的维度（flatten还是用于con1d），reward类别，回看长度，税费倍率。
#### State 种类
n为回看长度
* 0：n个价差 + 当前仓位 + 平仓收益。
    * state_dims为1时，shape: (n + 2, )。
    * state_dims为2时，shape: (n, 3)

* 1：14个信号。
    * state_dims为1时，shape: (14 * n, )。
    * state_dims为2时，shape: (n, 14)

* 2：14个信号 + 当前仓位 + 平仓收益。
    * state_dims为1时，shape: (14 * n + 2, )。
    * state_dims为2时，shape: (n, 16)

* 3：n个价差 + 14个信号 + 当前仓位 + 平仓收益。
    * state_dims为1时，shape: (15 * n + 2, )。
    * state_dims为2时，shape: (n, 17)
#### Reward 种类
* TP：           纯收益，资产差
* earning_rate： 资产变动百分比
* log_return：   log收益率
* running_SR：   移动夏普率


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