#深度学习框架PyTorch基础   P244

'''
机器学习基础 P205
按照训练数据中有无输出数据的标注分为监督学习(supervised learning)、无监督学习和半监督学习(semi-supervised learning)
监督学习的训练数据中包括输入范例和对应的输出范例 利用观察和统计输入输出之间的关系训练模型 并通过比对训练结果和标注结果之间的差异调整模型参数 提高预测性能 效果一般较好 但非常依赖标注数据的大小 亦花费大量人力和时间成本
无监督学习指训练数据没有对输出进行标注 需要模型自行定义训练标准 以达预期目标 如聚类 难度远大于监督学习 效果一般也逊于后者 可以有效利用海量未标注数据 特别适合NLP相关应用
半监督学习处理上述两者之间 一种方法是未标注数据进行预训练 通过少量标注数据引导模型进一步提高准确度

模型 建立模型是为了找到自变量房子的d种信息x=(x1,x2,...,xd)与因变量y价格的关系
参数 可能的关系有无数种 通过参数β0,β1...,βd来限定范围 y = β0 + β1x1 + β2x2 +...+ βdxd
     二次模型：y=β(0,0)+∑(i,∑(j,β(1,j)xixj))
不同模型的假设范围大小不一 参数个数和复杂程度也不一样 需要将参数调整到合适的值 训练模型的过程是根据输入和输出不断改变模型参数的取值 以得到具有一定准确度的模型 即参数优化(parameter optimization)

训练集 (training set)模型用于训练的数据
测试集 (test set)推广 泛化 在训练集上检验模型性能无法充分区分模型优劣 因此需要同分布却未出现的数据担当测试集 即关心模型在未见过数据上的推广能力 亦称泛化能力
过拟合 (overfitting) 模型本应在训练集上不断提升准确度 却在超过一定的阈值后于训练集上表现越好的模型在测试集上表现越差
验证集 (validation set) 为在训练中观察到过拟合现象 将训练集中分出一部分数据作为验证集 验证集准确度有下滑即可中止优化 选出之前验证集表现最好的模型作结果
'''

print("=====初始化张量代码示例=====")

import torch
#一个大小为2 x 3的实数型张量
a = torch.FloatTensor([[1.2, 3.4, 5], [3, 6, 7.4]])
print(a)
#一个大小为5 x 6的实数型张量，每个元素根据标准正态分布N(0,1)随机采样
b = torch.randn(5, 6) 
print(b)
#将a第0行第2列的元素改为4.0，变为([[1.2, 3.4, 4.0], [3, 6, 7.4]])
a[0, 2] = 4.0 
print(a)

print("=====运算和求导示例=====")

import torch
a = torch.ones(1)         # 一个1维向量，值为1
#a = a.cuda()              # 将a放入GPU，如果本机没有GPU，注释此句
print(a.requires_grad)           # False
a.requires_grad = True    # 设定a需要计算导数
b = torch.ones(1)
x = 3 * a + b             # x是最终结果
print(x.requires_grad)    # True，因为a需要计算导数，所以x需要计算导数
x.backward()              # 计算所有参数的导数
print(a.grad)                   # tensor([ 3.])，导数为3 

print("=====全连接层示例=====")

import torch.nn as nn
# 四层神经网络，输入层大小为30，两个隐藏层大小为50和70，输出层大小为1
linear1 = nn.Linear(30, 50)
linear2 = nn.Linear(50, 70)
linear3 = nn.Linear(70, 1)
# 10组输入数据作为一批次(batch)，每一个输入为30维
x = torch.randn(10, 30)
# 10组输出数据，每一个输出为1维
res = linear3(linear2(linear1(x)))
print(res)

print("=====丢弃示例=====")

layer = nn.Dropout(0.1)  # Dropout层，置零概率为0.1
input = torch.randn(5, 2)
print(input)
output = layer(input)        # 维度仍为5 x 2，每个元素有10%概率为0
print(output)

print("=====CNN示例=====")

# 卷积神经网络，输入通道有1个，输出通道3个，过滤器大小为5
conv = nn.Conv2d(1, 3, 5)
# 10组输入数据作为一批次(batch)，每一个输入为单通道32 x 32矩阵
x = torch.randn(10, 1, 32, 32)  
# y维度为10 x 3 x 28 x 28，表示输出10组数据，每一个输出为3通道28 x 28矩阵 (28=32-5+1)
y = conv(x)
print(y.shape) # torch.Size([10, 3, 28, 28]

print("=====RNN示例=====")

# 双层GRU输入元素维度是10，状态维度是20，batch是第1维
rnn = nn.GRU(10, 20, num_layers=2)    
# 一批次共3个序列，每个序列长度5，维度是10，注意batch是第1维
x = torch.randn(5, 3, 10) 
# 初始状态，共3个序列，2层，维度是20
h0 = torch.randn(2, 3, 20)
# output是所有的RNN状态，大小为5 x 3 x 20；hn大小为2 x 3 x 20，为RNN最后一个状态
output, hn = rnn(x, h0) 
print(output.shape) # torch.Size([5, 3, 20])
print(hn.shape) # torch.Size([2, 3, 20])
