#参考L1正则化

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
    self.layers=nn.Sequential(  
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
    
  def compute_l2_loss(self, w):  
      return torch.square(w).sum()  
    
if __name__=='__main__':  
    
  # 设置固定的随机数种子  
  torch.manual_seed(42)  
    
  # 准备MNIST数据集  
  dataset=MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())  
  trainloader=torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)  
    
  # 初始化MLP  
  mlp=MLP()  
    
  # 定义损失函数和优化器  
  loss_function=nn.CrossEntropyLoss()  
  optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-4)  
    
  # 运行训练循环  
  for epoch in range(0, 5): # 训练5个epoch  
      
    # 打印当前epoch  
    print(f'Starting epoch {epoch+1}')  
      
    # 遍历DataLoader获取训练数据  
    for i, data in enumerate(trainloader, 0):  
        
      # 获取输入数据和标签  
      inputs, targets=data  
        
      # 梯度清零  
      optimizer.zero_grad()  
        
      # 执行前向传播  
      outputs=mlp(inputs)  
        
      # 计算原始损失  
      loss=loss_function(outputs, targets)  
        
      # 计算L2正则化损失分量  
      l2_weight=1.0  
      l2_parameters= []  
      for parameter in mlp.parameters():  
          l2_parameters.append(parameter.view(-1))  
      l2=l2_weight*mlp.compute_l2_loss(torch.cat(l2_parameters))  
        
      # 将L2损失分量添加到总损失中  
      loss+=l2  
        
      # 执行反向传播  
      loss.backward()  
        
      # 执行优化步骤  
      optimizer.step()  
        
      # 打印训练统计信息  
      minibatch_loss=loss.item()  
      if i%500==499:  
          print('Loss after mini-batch %5d: %.5f (of which %.5f l2 loss)'%  
                (i+1, minibatch_loss, l2))  
          current_loss=0.0  
  # 训练完成  
    print('Training process has finished.')

#L2正则化在PyTorch中同样可以便捷地实现。与L1正则化不同，L2正则化计算权重值的平方而非绝对值。
#具体而言，我们将\sum_{i=1}^{n} w_i^2添加到损失函数中。
#L2损失的替代实现方法
#基于L2的权重衰减也可以通过在优化器中设置weight_decay参数来实现。
#weight_decay (float, 可选) — 权重衰减 (L2惩罚) (默认值: 0)
#PyTorch 文档 实现示例：
#optimizer=torch.optim.Adam(mlp.parameters(), lr=1e-4, weight_decay=1.0)
