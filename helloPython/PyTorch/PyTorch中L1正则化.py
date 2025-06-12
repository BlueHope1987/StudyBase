#conda install torchvision -c pytorch 或 pip3 install torchvision
#conda install pillow

#引用文章 https://mp.weixin.qq.com/s/zL_B7Zb6hN2FklaWk5epSg 提升模型泛化能力：PyTorch的L1、L2、ElasticNet正则化技术深度解析与代码实现



import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class MLP(nn.Module):
    '''
    多层感知器。
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28*1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''前向传播'''
        return self.layers(x)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

if __name__ == '__main__':

    # 设置固定的随机数种子
    torch.manual_seed(42)

    # 准备MNIST数据集
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # 初始化MLP
    mlp = MLP()

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    # 运行训练循环
    for epoch in range(0, 5):  # 训练5个epoch

        # 打印当前epoch
        print(f'Starting epoch {epoch+1}')

        # 遍历DataLoader获取训练数据
        for i, data in enumerate(trainloader, 0):

            # 获取输入数据和标签
            inputs, targets = data

            # 梯度清零
            optimizer.zero_grad()

            # 执行前向传播
            outputs = mlp(inputs)

            # 计算原始损失
            loss = loss_function(outputs, targets)

            # 计算L1正则化损失分量
            l1_weight = 1.0
            l1_parameters = []
            for parameter in mlp.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = l1_weight * mlp.compute_l1_loss(torch.cat(l1_parameters))

            # 将L1损失分量添加到总损失中
            loss += l1

            # 执行反向传播
            loss.backward()

            # 执行优化步骤
            optimizer.step()

            # 打印训练统计信息
            minibatch_loss = loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.5f (of which %.5f L1 loss)' %
                      (i+1, minibatch_loss, l1.item()))

    # 训练完成
    print('Training process has finished.')