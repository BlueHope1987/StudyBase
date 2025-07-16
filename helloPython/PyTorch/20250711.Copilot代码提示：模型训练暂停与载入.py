#模型载入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
# 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)
# 创建模型实例
model = SimpleModel()  
# 定义优化器和学习率调度器
optimizer = optim.SGD(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
# 模拟训练过程
for epoch in range(5):
    # 生成一些随机数据
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 1)
    # 前向传播
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
# 保存模型状态
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}
torch.save(checkpoint, 'model_checkpoint.pth')
# 定义载入模型的函数
def load_model(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch']
# 载入模型
loaded_epoch = load_model(model, optimizer, scheduler, 'model_checkpoint.pth')
print(f"Loaded model from epoch {loaded_epoch}")
# 继续训练或进行推理
for epoch in range(loaded_epoch, loaded_epoch + 5):
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 1)
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
# 模型载入与训练暂停
# 这里可以添加更多的代码来处理模型载入后的逻辑，例如评估模型性能或保存新的检查点等
