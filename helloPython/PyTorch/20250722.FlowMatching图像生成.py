'''
https://mp.weixin.qq.com/s/x2CDIKId64136zx8Gr-R6g
Flow Matching生成模型：从理论基础到Pytorch代码实现

>>>示例同 20250723.checkerboard_flow_matching.ipynb

与传统扩散模型通过逆向去噪过程生成数据不同，Flow Matching通过学习时间相关的速度场，建立从噪声分布到目标数据分布的直接映射路径。文章将理论推导与代码实现相结合，使用2D演示数据集验证方法的有效性，为深度学习研究者和工程师提供了一个完整的技术参考。

引言
扩散模型在生成建模领域取得了显著成功，能够生成高质量的图像、视频和音频内容。然而，这类模型存在一个关键局限性：生成过程需要执行数百个去噪步骤，导致推理效率极低。
而Flow Matching的核心思想是通过求解常微分方程(ODE)来学习数据生成过程，而非通过逆向扩散过程。
本文将系统阐述Flow Matching的完整实现过程，包括数学理论推导、模型架构设计、训练流程构建以及速度场学习等关键组件。通过本文的学习，读者将掌握Flow Matching的核心原理，获得一个完整的PyTorch实现，并对生成模型在噪声调度和分数函数之外的发展方向有更深入的理解。

扩散模型的局限性分析扩散模型的工作机制
为了更好地理解Flow Matching的优势，我们首先分析广泛应用的扩散模型，特别是去噪扩散概率模型(DDPMs)的工作原理。这类模型通过学习逆向噪声添加过程来实现图像生成。
训练阶段，模型通过逐步向数据添加噪声直至变成纯高斯噪声来构建前向过程。这个过程本质上是观察图像逐渐退化为随机噪声的过程。然后模型学习执行逐步去噪操作，这等价于学习分数函数，即数据分布的梯度场。从数学角度来看，分数函数提供了指向原始清晰图像的方向信息。

推理阶段的计算复杂性
扩散模型在生成新图像时面临的主要挑战在于推理过程的复杂性。生成过程需要从噪声开始，通过相同的逆向过程逐步去噪。这要求将分数函数嵌入到随机微分方程(SDE)中，该方程描述了噪声随时间的逆向演化规律。即使对于完全训练的模型，采样过程仍需要通过分数函数引导执行大量的去噪步骤。

现有加速方法及其局限性
研究社区提出了多种加速扩散采样的方法。去噪扩散隐式模型(DDIM)通过消除微分方程中的随机性，实现了扩散过程的确定性版本。这些扩散ODE方法使用确定性求解器实现更快的逆向计算。尽管如此，这些改进方法仍然需要数十个去噪步骤和相应次数的神经网络前向传播。

Flow Matching的解决思路
Flow Matching提出了一种根本不同的解决方案：不学习如何去噪，而是直接学习速度场，该速度场描述如何在一条平滑路径上将粒子从噪声推向数据分布。这种方法跳过了传统扩散模型的逆向噪声过程，而是训练模型沿着连续路径将样本移向目标数据分布。

问题定义与数学描述
图像生成任务的核心在于建模样本如何从简单的初始分布(如高斯噪声)移动到复杂的数据分布(如自然图像)。与基于去噪的方法通过学习逐步逆向噪声过程不同，Flow Matching直接建模样本在两个分布之间的流动过程。
从直观理解来看，Flow Matching模型学习一条连接噪声和数据的平滑时间相关变换路径。这种解释使我们能够将图像生成视为一个传输问题：如何将一个点从起始位置移动到目标位置。

Flow Matching目标函数
Flow Matching方法从两个分布开始建模：简单先验分布p₀(例如标准高斯噪声)和复杂终端分布p₁(例如自然图像)。
建模过程首先从每个分布中采样一个点，然后用路径连接它们。最常用的连接方式是直线插值：x(t) = (1-t)x₀ + tx₁。这种线性插值方法在x₀和x₁之间建立了最直接的连接路径。虽然也可以使用其他插值方法如球面插值，但线性插值因其简单性和良好的实践效果而被广泛采用。
目标是学习一个时间相关的速度场f(x,t)，该速度场描述轨迹上每个点的瞬时速度。由于我们已经从路径定义中知道了真实速度，神经网络的任务是近似x'(t) = x₁ - x₀。
训练过程就是使模型的预测速度与真实速度匹配。

Flow Matching采样过程
在训练阶段，网络学习了将点从噪声移动到数据的速度场f(x,t)。训练循环中同时访问x₀和x₁，但在生成阶段只有x₀。采样时的目标是取一个噪声点并将其推向p₁分布，最终获得类似自然图像的结果。
采样过程从样本x₀ ~ p₀(例如标准高斯噪声)开始。然后定义从t = 0到t = 1的时间网格，将其均匀分割成一系列步骤。在每个时间步，我们向前求解ODE来更新样本：
速度场f(x,t)在每个步骤中与当前的x和t一起使用，以获得x'(t)的估计。一旦到达t = 1，就得到了一个希望看起来像自然图像的样本。这个过程类似于跟随流场，我们沿着学习的速度路径将样本"推向"数据方向。
'''

'''
在深入Flow Matching实现之前，我们需要定义一对分布进行映射。虽然可以选择任意两个噪声和数据分布，但为了便于理解和可视化，我们选择两个简单的分布进行演示。
'''
import numpy as np
import torch.nn as nn
import torch
from tqdm import tqdm
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 源分布p₀采用2D标准高斯分布：

def sample_source(batch_size):  
     # 从2D标准高斯分布(均值=0，标准差=1)采样  
     return torch.randn(batch_size, 2)

# 目标分布p₁采用2D棋盘数据集，这是一个由棋盘网格中高斯簇组成的演示数据集：

def sample_target(batch_size):  
    # 在范围[-2, 2)内均匀采样x坐标  
    x1 = torch.rand(batch_size) * 4 - 2  

    # 采样y坐标的步骤：  
    # 步骤1：从均匀分布[0, 1)抽取  
    # 步骤2：随机减去0或2(通过torch.randint)  
    # 结果：大致以-2或-1为中心的值  
    x2_ = torch.rand(batch_size) - torch.randint(high=2, size=(batch_size, )) * 2  

    # 根据x1 bin是偶数还是奇数添加垂直偏移  
    # 这创建了棋盘的交替行偏移  
    x2 = x2_ + (torch.floor(x1) % 2)  

    # 将x1和x2堆叠成(batch_size, 2)向量，并缩放整个网格  
    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45  

    return torch.tensor(data, dtype=torch.float32)

'''
神经网络架构设计
我们构建一个小型神经网络来学习时间相关的速度场f(x,t)。考虑到数据集的复杂性适中，我们设计了一个多层感知机(MLP)，该网络以(x,t)作为输入，输出与x相同维度的速度向量。
由于时间t是标量，我们首先使用两个全连接层将其映射到更高维度空间。然后将这个时间嵌入与位置x连接，并通过几个具有SiLU激活函数的全连接层进行处理。
'''

class FlowModel(nn.Module):   
      """学习时间相关速度场f(x, t)的神经网络"""
      def __init__(self, input_dim=2, time_embed_dim=64):   
            super().__init__()   
                
            # 小型MLP将时间标量t嵌入到更高维空间   
            self.time_embed = nn.Sequential(   
                  nn.Linear(1, time_embed_dim),   
                  nn.SiLU(),   # 激活函数：Sigmoid线性单元   
                  nn.Linear(time_embed_dim, time_embed_dim)   
            )   
          
            # 主网络预测速度，给定(x, 嵌入的t)   
            self.net = nn.Sequential(   
                  nn.Linear(input_dim + time_embed_dim, 128),   # 输入：连接的x和t嵌入   
                  nn.SiLU(),   
                  nn.Linear(128, 128),   
                  nn.SiLU(),   
                  nn.Linear(128, 128),   
                  nn.SiLU(),   
                  nn.Linear(128, 128),   
                  nn.SiLU(),   
                  nn.Linear(128, 128),   
                  nn.SiLU(),   
                  nn.Linear(128, 128),   
                  nn.SiLU(),   
                  nn.Linear(128, input_dim)   # 输出：预测速度(与x相同维度)   
            )   
          
      def forward(self, x, t):   
            # 将时间t(形状：[batch_size, 1])嵌入到更高维向量   
            t_embed = self.time_embed(t)   
          
            # 沿最后一个维度连接位置x和时间嵌入   
            xt = torch.cat([x, t_embed], dim=-1)   
          
            # 通过网络预测在(x, t)处的速度   
            return self.net(xt)
      

'''
损失函数设计
训练目标是使模型匹配路径上的真实速度。由于真实速度就是x₁ - x₀，我们通过最小化真实速度与预测速度之间的平方误差来实现这一约束。
'''

def flow_matching_loss(model, x0, x1, t):  
    # 计算每个t时刻轨迹上的插值点  
    xt = (1 - t) * x0 + t * x1  
    
    # 计算真实速度向量(在轨迹上恒定)  
    v_target = x1 - x0  
    
    # 使用模型预测点(x(t), t)处的速度  
    v_pred = model(xt, t)  
    
    # 计算每个样本预测和真实速度之间的平方误差  
    # 然后在整个批次上平均  
    return ((v_pred - v_target) ** 2).mean()

'''
训练流程实现
模型训练过程包括从p₀和p₁分布采样噪声和数据点。在每个训练步骤中，我们在[0,1]范围内选择随机时间，计算Flow Matching损失，并使用Adam优化器更新参数。随着训练进行，模型逐渐学会连接两个分布的速度场。
'''

num_steps = 10000  
batch_size = 512  
losses = []  

model = FlowModel().to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  

for step in tqdm(range(num_steps)):  
    x0 = sample_source(batch_size).to(device)  
    x1 = sample_target(batch_size).to(device)  
    t = torch.rand(batch_size, 1).to(device)  # 随机插值时间 ∈ [0, 1]  
    
    loss = flow_matching_loss(model, x0, x1, t)  
    
    optimizer.zero_grad()  
    loss.backward()  
    optimizer.step()  
    
    losses.append(loss.item())  
    
    if step % 100 == 0:  
         print(f"Step {step} | Loss: {loss.item():.4f}")


##### 附加 图表代码
# Plot after training
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()

def plot_velocity_row(model, t_values=[0.0, 0.25, 0.5, 0.75, 1.0], grid_size=20):
    model.eval()
    with torch.no_grad():
        # Set up grid and figure
        fig, axes = plt.subplots(1, len(t_values), figsize=(4 * len(t_values), 4))
        x = np.linspace(-4, 4, grid_size)
        y = np.linspace(-4, 4, grid_size)
        xx, yy = np.meshgrid(x, y)
        xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
        xt = torch.tensor(xy, dtype=torch.float32).to(device)

        for i, t_val in enumerate(t_values):
            # Repeat t for each grid point
            tt = torch.full((xt.shape[0], 1), t_val, dtype=torch.float32).to(device)

            # Predict velocity vectors
            v = model(xt, tt).cpu().numpy()

            # Plot velocity field as arrows
            ax = axes[i]
            ax.quiver(xx, yy,
                      v[:, 0].reshape(grid_size, grid_size),
                      v[:, 1].reshape(grid_size, grid_size),
                      scale=20)
            ax.set_title(f"t = {t_val}")
            ax.axis("equal")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    plot_velocity_row(model)


'''
采样算法实现
为了从训练好的模型生成新样本，我们从噪声分布中的一个点开始，使用学习的速度场f(x,t)向前推动它。这通过求解从t = 0到t = 1的ODE来完成，正如第3.3节中描述的那样。我们使用scipy.integrate.solve_ivp这一标准ODE求解器来数值积分学习的速度场，产生位于目标分布中的输出。
'''

def sample_flow(model, x0, t_span=(0, 1)):  
    """
    通过学习的流演化x0以产生来自p1的样本。  
    """
    def ode_func(t, x):  
        # 将输入x和时间t转换为适当的torch张量  
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  
        t_tensor = torch.tensor([[t]], dtype=torch.float32).to(device)  

        # 预测速度而不跟踪梯度  
        with torch.no_grad():  
            v = model(x_tensor, t_tensor)  

        # 返回速度作为NumPy数组(形状：[2])  
        return v.squeeze(0).cpu().numpy()  

    # 使用学习的速度场从t=0到t=1求解ODE  
    sol = solve_ivp(ode_func, t_span, x0.cpu().numpy(), t_eval=[t_span[1]])  

    # 返回t=1时的最终状态(即预测的x1)  
    return sol.y[:, -1]



##### 附加 图表代码

samples = []

# Sample 1000 points from the source distribution p₀
x0 = sample_source(1000).to(device)

# Push each point through the learned flow to generate a sample from p₁
for x in x0:
    with torch.no_grad():
        x = x.to(device)
        x1_hat = sample_flow(model, x, t_span=(0, 1))  # Integrate flow from t=0 to t=1
        samples.append(x1_hat)

# Convert list of sampled points to NumPy array for plotting
samples = np.array(samples)

# Scatter plot of generated samples in 2D
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
plt.title("Generated Samples from Flow Matching")
plt.axis("equal")
plt.grid(True)
plt.show()

real = sample_target(1000)
gen = np.array(samples)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(real[:, 0], real[:, 1], alpha=0.3)
plt.title("Target: Checkerboard")

plt.subplot(1, 2, 2)
plt.scatter(gen[:, 0], gen[:, 1], alpha=0.3)
plt.title("Generated from Model")

plt.show()

'''
实验结果与性能分析
数据分布特征
我们为Flow Matching设计了一个简单的合成玩具问题，其中源分布p₀为2D高斯分布，目标分布p₁为棋盘模式。这种设置使我们能够轻松可视化学习的流场和中间诊断结果。
训练收敛性分析
训练损失曲线显示了良好的收敛特性。损失在早期迭代中快速下降，随后进入振荡阶段，这是神经网络训练中的典型现象。
速度场演化过程
学习的速度场在时间维度上表现出平滑的演化特性。在t = 0时，流场从中心向外定向，反映了向高斯分布的移动特征。在较大的t值时，流场开始表现出更接近棋盘结构的特征。
生成质量评估
为了生成新样本，我们使用scipy.integrate.solve_ivp从t = 0到t = 1积分学习的速度场。从定性角度来看，生成的样本在形状和结构上与目标棋盘分布紧密匹配，验证了方法的有效性。

方法局限性与发展方向
现有局限性分析
虽然Flow Matching为分数匹配提供了简单而优雅的替代方案，但仍面临一些重要局限性。
首先是采样保证问题。Flow Matching绕过了对数似然计算，而是将模型拟合到速度场。虽然这种方法简化了训练过程，但不再保证生成的样本严格位于目标分布中。这与传统的基于似然的方法形成对比，后者在理论上提供了更强的分布保证。
其次是推理时的积分成本。要使用这种方法生成新样本，需要在推理时为每个输入样本求解ODE。与前馈模型或具有较少步骤的扩散采样相比，这在计算上可能是昂贵的，特别是在需要高精度积分的情况下。
最后是速度监督的需求。Flow Matching需要访问真实速度信息。虽然这对于简单的合成数据集来说是直接的，但对于现实世界的复杂数据来说，获得准确的速度监督变得极其复杂。

改进方向与扩展
为了解决这些局限性，研究社区已经开发了几种改进方法。
带分数模型的Flow Matching方法开始结合两种模型的优势，使用基于分数的目标来训练Flow Matching模型。这种混合方法结合了两个世界的优点，既保持了Flow Matching的简洁性，又获得了分数模型的理论保证。
神经ODE求解器的发展为减少推理时间提供了新的可能性。更先进的ODE求解器甚至神经近似器可以通过学习通过流场的高效求解方案来减少推理时间，从而允许在推理时进行更快的采样。

总结
本文从理论基础到实践实现，系统地构建了一个完整的Flow Matching模型。通过这个过程，我们深入探讨了Flow Matching背后的核心思想，分析了它与传统扩散模型的本质区别，并重新实现了核心组件，包括玩具2D数据集、向量场估计和ODE积分等关键技术。
Flow Matching作为生成建模的新兴方法，为解决扩散模型的效率问题提供了有前景的解决方案。虽然仍存在一些理论和实践上的挑战，但其简洁的数学框架和良好的实验效果表明了这一方向的巨大潜力。
对于希望深入理解生成模型内部机制、原型化自己的研究想法或满足技术好奇心的研究者和工程师来说，本文提供的完整实现框架将是一个有价值的起点和参考。

源代码：
https://github.com/vickiiimu/checkerboard-FM-tutorial/blob/main/checkerboard_flow_matching.ipynb
'''