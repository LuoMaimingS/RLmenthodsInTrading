# RLMethods-Trading 记录

## version 1 （0715）

- 网络结构，PPO和A3C用的是LSTM，Rainbow和SAC的网络用的全连接网络
- 环境设置，reward用的total profit，state为前50分钟的价差，训练集为2015-01-01到2019-12-31，验证集为2020-01-01到2020-05-31的代码为IF9999.CCFX的股指期货
- 训练集回测

<img src="images/trading_profit_state0-tain.png" style="zoom:72%;" />

- 验证集回测

<img src="images/trading_profit_state0-eval.png" style="zoom:72%;" />

上图横坐标为交易的天数，纵坐标为earning rate，即收益率。共展示了四种策略（PPO，A3C，SAC，Rainbow）以及两个baselines（Long only和Short Only）



- 总结与下一版本计划

看到在当前版本，没有学到一个能够表现比baseline好的策略，最好学到了与baseline策略相当的策略。

下一步版本加入更多的feature，看是否feature多了之后能够学到更好的策略。

## version 2（0717）

网络结构与环境设置与version 1保持一致，state不再使用前五十分钟的价差，而是手动提取了14种不同的金融特征。

- 训练情况

![](/Users/hank/Desktop/rl-trade/records/images/state1-training.png)

- 训练集回测

<img src="images/trading_profit_state1_train.png" style="zoom:72%;" />

- 测试集回测

<img src="images/trading_profit_state1_eval.png" style="zoom:72%;" />

## version 3（0717）

网络结构与环境设置与version 2一致，只是在verison2的基础上在state里加入了持仓的情况

- 训练情况

![](/Users/hank/Desktop/rl-trade/records/images/state2-training.png)

- 训练集回测

<img src="images/trading_profit_state2_train.png" style="zoom:72%;" />

- 测试集回测

<img src="images/trading_profit_state2_eval.png" style="zoom:72%;" />

- 总结与下一版本计划

可以看到version 2和version 3的表现基本差不多，在训练过程中都表现不稳定，都没有训练获得表现突出的策略。其中，PPO和SAC趋近于不做动作，而Rainbow和SAC就是没有学到策略，回撤很大。

下一步打算统一各个模型的网络结构，都采用CNN尝试是否有所改善。