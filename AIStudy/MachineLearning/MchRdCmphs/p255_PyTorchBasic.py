import torch
#一个大小为2 x 3的实数型张量
a = torch.FloatTensor([[1.2, 3.4, 5], [3, 6, 7.4]])
print(a)
#一个大小为5 x 6的实数型张量，每个元素根据标准正态分布N(0,1)随机采样
b = torch.randn(5, 6) 
#将a第0行第2列的元素改为4.0，变为([[1.2, 3.4, 4.0], [3, 6, 7.4]])
a[0, 2] = 4.0 


import torch
a = torch.ones(1)         # 一个1维向量，值为1
a = a.cuda()              # 将a放入GPU，如果本机没有GPU，注释此句
a.requires_grad           # False
a.requires_grad = True    # 设定a需要计算导数
b = torch.ones(1)
x = 3 * a + b             # x是最终结果
print(x.requires_grad)    # True，因为a需要计算导数，所以x需要计算导数
x.backward()              # 计算所有参数的导数
a.grad                    # tensor([ 3.])，导数为3 

import torch.nn as nn
# 四层神经网络，输入层大小为30，两个隐藏层大小为50和70，输出层大小为1
linear1 = nn.Linear(30, 50)
linear2 = nn.Linear(50, 70)
linear3 = nn.Linear(70, 1)
# 10组输入数据作为一批次(batch)，每一个输入为30维
x = torch.randn(10, 30)
# 10组输出数据，每一个输出为1维
res = linear3(linear2(linear1(x)))