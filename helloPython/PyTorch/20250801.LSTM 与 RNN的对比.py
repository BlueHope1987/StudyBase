'''
https://mp.weixin.qq.com/s/jt43FLx6wUYOcn10wmKiUA
通透！LSTM vs RNN ！！

LSTM 和传统的 RNN 都是处理序列数据的神经网络结构。RNN 在理论上可以捕捉长时依赖，但在实际应用中存在梯度消失和梯度爆炸问题，
导致其难以学习长期依赖关系。LSTM 通过引入记忆单元（Memory Cell），有效缓解了这一问题。


RNN 的核心思想是在时间步之间共享权重，并引入隐藏状态h(t)来传递信息。结构包括：
时刻t的隐藏状态 非线性激活函数（如tanh或ReLu） 权重矩阵 偏置
训练时通过反向传播计算损失关于每个参数的梯度。对于长期依赖，梯度涉及多个时间步的连乘 如果特定部分<1，则梯度会指数衰减，导致梯度消失；反之则爆炸。

LSTM 通过引入记忆单元  和门机制（gate mechanism），来控制信息的保存、更新和丢弃，从而解决 RNN 的长依赖问题。
LSTM 单元包含：
遗忘门 输入门 候选记忆 记忆单元 输出门 隐藏状态
给定输入x(t)和上一时刻的隐藏状态h(t-1) 、记忆单元状态c(t-1) ，LSTM 的更新过程如下：
1. 遗忘门（Forget Gate）：决定保留多少旧信息
2. 输入门（Input Gate）：决定写入多少新信息
3. 候选记忆（Candidate）：新信息的候选值
4. 更新记忆单元（Cell State）
5. 输出门（Output Gate）：决定输出多少当前记忆
6. 更新隐藏状态（Hidden State）
梯度稳定：主路径中不经过激活函数（只涉及加法与乘法），保留了长期梯度流动的能力，避免梯度消失。
门控机制：使模型能够根据上下文动态学习应保留、遗忘或加入哪些信息。
长期依赖建模能力强：有效捕捉远距离依赖。

RNN vs LSTM 总结对比
项目        RNN         LSTM
结构复杂度  简单        较复杂（多个门）
记忆能力    短期        长期（记忆单元）
梯度消失    容易发生    显著缓解
训练难度    容易陷入局部最优    更稳定，但计算量大
表达能力    有限    强（可学复杂模式）

RNN 在每个时间步仅依赖h(t-1)，并通过非线性变换传播信息，导致长期信息难以保留。
而 LSTM 的核心创新在于引入了**累加式的记忆状态 **，在其路径上没有非线性变换，使得梯度可以更容易地沿时间反向传播，从而有效建模长期依赖。


完整案例
数据集
生成一个正弦函数叠加高斯噪声作为时间序列，用前 80% 的数据训练，20% 用作测试；
预测任务：根据前 20 个时间步，预测下一个值。

模型结构
RNNModel: 使用一个标准的单层 RNN；
LSTMModel: 使用单层 LSTM；
都接一个全连接层预测输出；
Loss 函数为 MSE，优化器为 Adam。

'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# 设置中文字体（如微软雅黑），确保中文正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 创建一个合成的时间序列数据（正弦波+噪声）
t = np.arange(0, 100, 0.1)
data = np.sin(t) + np.random.normal(scale=0.5, size=len(t))
df = pd.DataFrame(data, columns=['value'])

# 数据归一化
scaler = MinMaxScaler()
df['value'] = scaler.fit_transform(df[['value']])

# 创建数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# 创建训练和测试数据
seq_length = 20
train_size = int(len(df) * 0.8)
train_data = df['value'].values[:train_size]
test_data = df['value'].values[train_size - seq_length:]

train_dataset = TimeSeriesDataset(train_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        out = self.fc(out[:, -1, :])
        return out

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        out = self.fc(out[:, -1, :])
        return out

# 训练函数
def train_model(model, train_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
    return model, losses

# 预测函数
def predict(model, data_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            pred = model(x_batch)
            predictions.append(pred.item())
            actuals.append(y_batch.item())
    return np.array(predictions), np.array(actuals)

# 初始化模型并训练
rnn_model = RNNModel()
lstm_model = LSTMModel()

rnn_model, rnn_losses = train_model(rnn_model, train_loader, num_epochs=20)
lstm_model, lstm_losses = train_model(lstm_model, train_loader, num_epochs=20)

# 模型预测
rnn_preds, rnn_actuals = predict(rnn_model, test_loader)
lstm_preds, lstm_actuals = predict(lstm_model, test_loader)

# 反归一化预测值
rnn_preds_inv = scaler.inverse_transform(rnn_preds.reshape(-1, 1)).flatten()
lstm_preds_inv = scaler.inverse_transform(lstm_preds.reshape(-1, 1)).flatten()
actuals_inv = scaler.inverse_transform(lstm_actuals.reshape(-1, 1)).flatten()

# 绘制图像
plt.figure(figsize=(16, 12))

# 图1: 损失函数曲线
plt.subplot(2, 2, 1)
plt.plot(rnn_losses, label='RNN Loss', color='crimson')
plt.plot(lstm_losses, label='LSTM Loss', color='mediumseagreen')
plt.title('训练损失随Epoch变化图')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 图2: 测试集预测结果比较
plt.subplot(2, 2, 2)
plt.plot(actuals_inv, label='Actual', color='black')
plt.plot(rnn_preds_inv, label='RNN Predicted', color='darkorange')
plt.plot(lstm_preds_inv, label='LSTM Predicted', color='dodgerblue')
plt.title('测试集预测结果比较')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 图3: 预测误差分布直方图
plt.subplot(2, 2, 3)
rnn_error = rnn_preds_inv - actuals_inv
lstm_error = lstm_preds_inv - actuals_inv
plt.hist(rnn_error, bins=30, alpha=0.7, label='RNN Error', color='red')
plt.hist(lstm_error, bins=30, alpha=0.7, label='LSTM Error', color='green')
plt.title('预测误差分布直方图')
plt.xlabel('误差')
plt.ylabel('频率')
plt.legend()
plt.grid(True)

# 图4: 实际 vs 预测散点图
plt.subplot(2, 2, 4)
plt.scatter(actuals_inv, rnn_preds_inv, label='RNN', alpha=0.5, color='purple')
plt.scatter(actuals_inv, lstm_preds_inv, label='LSTM', alpha=0.5, color='cyan')
plt.plot(actuals_inv, actuals_inv, color='black', linestyle='--', label='Ideal')
plt.title('实际值 vs 预测值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

'''
图1：训练损失曲线：展示模型随训练过程损失的下降趋势。LSTM 收敛更快，且最终损失更低，表明其对序列建模更有效。
图2：测试集预测结果对比：可视化真实值与两种模型的预测值对比。LSTM 更紧密拟合真实趋势，RNN 预测波动性较大。
图3：误差分布图：分析两种模型预测误差的分布范围。LSTM 的误差集中度高，波动小；RNN 误差分布更宽，精度低。
图4：实际 vs 预测散点图：检查预测值与真实值的一致性。LSTM 点更集中在理想线附近，说明相关性更高。

优势总结与适用场景
RNN 优势：
结构简单，参数少；
适用于短期依赖序列建模；
训练速度快，适合资源受限场景。

RNN 劣势：
容易陷入梯度消失/爆炸；
长期依赖学习能力弱。

LSTM 优势：
通过门控机制保留长期信息；
性能更稳定，泛化能力强；
适用于文本、语音、金融等长时间序列建模。

LSTM 劣势：
参数多，训练成本高；
相比 RNN 更复杂，不易调参。

LSTM 在捕捉长期依赖关系上明显优于传统 RNN，具有更强的预测准确性和稳定性。尽管 RNN 结构简单、训练快速，但容易出现梯度消失问题，难以处理复杂时间序列。对于需要建模长期趋势的任务，LSTM 是更可靠的选择。
'''