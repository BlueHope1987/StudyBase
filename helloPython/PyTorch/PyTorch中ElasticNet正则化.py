#参考L1正则化
# 以下示例展示了在PyTorch中实现ElasticNet(L1+L2)正则化的具体方法。
# 在这个实现中，MLP类提供了计算L1和L2损失的独立函数。
# 在训练循环中，这两种损失以加权方式应用(权重分别为0.3和0.7)。
# 在输出统计信息时，各损失分量也会显示在控制台中。

#正则化技术是防止模型过拟合的关键手段，通过在损失函数中添加权重惩罚项，能够有效提升模型的泛化能力。
#L1正则化通过权重的绝对值惩罚促进稀疏性，L2正则化通过权重的平方惩罚控制模型复杂度，
#而ElasticNet正则化则结合两者的优势，提供了更灵活的正则化策略。
#在实际应用中，选择合适的正则化方法和权重系数对于获得最佳的模型性能至关重要。

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
    
  def compute_l1_loss(self, w):  
      return torch.abs(w).sum()  
    
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
        
      # 指定L1和L2权重系数  
      l1_weight=0.3  
      l2_weight=0.7  
        
      # 计算L1和L2正则化损失分量  
      parameters= []  
      for parameter in mlp.parameters():  
          parameters.append(parameter.view(-1))  
      l1=l1_weight*mlp.compute_l1_loss(torch.cat(parameters))  
      l2=l2_weight*mlp.compute_l2_loss(torch.cat(parameters))  
        
      # 将L1和L2损失分量添加到总损失中  
      loss+=l1  
      loss+=l2  
        
      # 执行反向传播  
      loss.backward()  
        
      # 执行优化步骤  
      optimizer.step()  
        
      # 打印训练统计信息  
      minibatch_loss=loss.item()  
      if i%500==499:  
          print('Loss after mini-batch %5d: %.5f (of which %.5f L1 loss; %0.5f L2 loss)'%  
                (i+1, minibatch_loss, l1, l2))  
          current_loss=0.0  
  # 训练完成  
    print('Training process has finished.')