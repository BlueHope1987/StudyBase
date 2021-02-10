#P230 优化网络模块

import torch
import torch.nn as nn
import torch.optim as optim   # 优化器软件包

from p229_FirstNet import FirstNet #导入本地简单网络类 或复用代码

net = FirstNet(10, 20, 15)
net.train()              # 将FirstNet置为训练模式（启用Dropout）
#net.cuda()               # 如果有GPU，执行此语句将FirstNet的参数放入GPU 

# 随机定义训练数据
# 共30个序列，每个序列长度5，维度是10
x = torch.randn(30, 5, 10)  
y = torch.randn(30, 1)       # 30个真值
# 随机梯度下降SGD优化器，学习率为0.01
optimizer = optim.SGD(net.parameters(), lr=0.01)  
for batch_id in range(10):
    # 获得当前批次的数据，batch_size=3
    x_now = x[batch_id * 3: (batch_id + 1) * 3]
    y_now = y[batch_id * 3 : (batch_id + 1) * 3]
    res = net(x_now)                         # RNN结果res，维度为3 x 1 x 15
    y_hat, _ = torch.max(res, dim=2)      # 最终预测张量y_hat，维度为3 x 1
    # 均方差损失函数
    loss = torch.sum(((y_now - y_hat) ** 2.0)) / 3  
    print('loss =', loss)
    optimizer.zero_grad()                   # 对net里所有张量的导数清零
    loss.backward()                          # 自动实现反向传播
    optimizer.step()                         # 按优化器的规则沿导数反方向移动每个参数

net.eval()            # 训练完成后，将FirstNet置为测试模式（Dropout不置零，不删除神经元）
y_pred = net(x)       # 获得测试模式下的输出