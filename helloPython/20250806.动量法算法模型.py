'''
https://mp.weixin.qq.com/s/RY-eTUYd3ISRHwrWBUJKGg
讲透一个强大算法模型，动量法！！

首先，我们可以这样想：打个比方，学模型就像骑自行车下山
· 普通的优化方法（比如梯度下降），就像你骑车下坡，每次都看看坡的斜度（梯度），然后决定该往哪边转、滑多少。
· 但如果坡太陡，或者有小石头，你可能一下滑得太猛、方向还乱晃，结果来回左右横跳，效率低，甚至摔跤（震荡）。
于是，动量法就来了！
动量法像是在你车上加了一个陀螺仪+减震系统：
· 它会记住你过去滑的方向和速度（就像动量的概念：速度 × 质量），
· 然后在下一次滑的时候，结合之前的趋势来调整方向，而不是只看当前的坡有多陡。
· 这样就更平滑地朝目标前进，不容易被小坑坑绊住。
总结一句话，动量法就是在训练模型时“带点惯性思维”，不是每次都完全重新判断，而是借着以前的方向继续往前冲，让学习更快更稳！

传统的梯度下降法在更新参数时只是简单地沿着当前梯度的反方向前进一步
· 当你在山地上滑行，目标是尽快找到山谷（也就是最小化损失）。
· 传统方法就像只看自己脚下的坡度，每次都沿着坡度最陡的方向下滑。
· 而动量法则类似于在下坡时带有“惯性”或者“冲劲”。当你持续向某个方向滑行时，就会积累一定的动能，使得你不会被局部较小的波动（局部最优）所影响，从而更有可能跨过浅坑或者消除颠簸，加速朝正确方向前进。
这也就是说，动量法在每一步更新时不仅考虑当前的梯度，还“记住”过去的梯度方向，这有助于平滑更新过程，尤其在面对“狭长”或“陡峭而曲折”的损失函数时能够获得更稳定和快速的收敛。
传统方法的问题在于：当损失函数的曲率在不同方向上差别较大时（例如在鞍点或陡峭和缓平区域交替出现的情况），可能出现震荡或收敛缓慢的情况。
动量法的更新规则引入了一个“速度变量”（或称动量变量），用于累加之前的梯度信息。相比于单纯的梯度下降，引入动量后更新量被放大了数倍，从而在梯度方向稳定的区域能够加速前进。
另一方面，如果不同方向的梯度交替变化，由于动量系数  使得前几次的更新未能完全“遗忘”，不同方向的梯度会部分相互抵消，从而减少振荡、增加稳定性。
这种方法在深度学习和其他机器学习模型中应用广泛，能够显著改善收敛速度和训练质量。
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 1. 定义目标函数及其梯度
def rosenbrock(X, a=1, b=100):
    """
    Rosenbrock 函数
    输入：
        X: 二维输入向量 [x, y]
        a, b: 函数参数，默认 a=1, b=100
    输出：
        Rosenbrock 函数值
    """
    x, y = X[0], X[1]
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(X, a=1, b=100):
    """
    Rosenbrock 函数梯度
    输出：
        梯度向量 [df/dx, df/dy]
    推导过程：
        对 x 求导： df/dx = -2*(a - x) - 4*b*x*(y - x**2)
        对 y 求导： df/dy = 2*b*(y - x**2)
    """
    x, y = X[0], X[1]
    grad_x = -2*(a - x) - 4*b*x*(y - x**2)
    grad_y = 2*b*(y - x**2)
    return np.array([grad_x, grad_y])

# 2. 标准 Momentum 方法和 Nesterov Momentum 方法的实现
def momentum_optimizer(grad_func, init, lr=0.001, gamma=0.9, iterations=10000, tol=1e-6):
    """
    标准 Momentum 优化器
    参数：
        grad_func: 损失函数梯度函数
        init: 初始参数（二维向量）
        lr: 学习率
        gamma: 动量系数（惯性因子）
        iterations: 最大迭代次数
        tol: 收敛判断阈值
    返回：
        params: 记录每一步参数的变化轨迹，便于可视化
        loss_history: 记录每一步的目标函数值
    """
    params = [init.copy()]
    loss_history = [rosenbrock(init)]
    # 初始化速度变量（动量项）
    v = np.zeros_like(init)
    theta = init.copy()

    for i in range(iterations):
        grad = grad_func(theta)
        # 更新动量：当前速度等于动量系数乘以前一次速度加上学习率乘以当前梯度
        v = gamma * v + lr * grad
        # 更新参数
        theta = theta - v

        params.append(theta.copy())
        loss_history.append(rosenbrock(theta))

        # 收敛判断：如果梯度的模小于 tol，则认为收敛
        if np.linalg.norm(grad) < tol:
            print(f"Standard Momentum converged after {i+1} iterations.")
            break

    return np.array(params), np.array(loss_history)

def nesterov_optimizer(grad_func, init, lr=0.001, gamma=0.9, iterations=10000, tol=1e-6):
    """
    Nesterov Accelerated Gradient (NAG) 优化器
    参数与标准 Momentum 相同，不同在于梯度计算时使用预估位置：theta - gamma * v
    返回：
        params: 记录参数轨迹
        loss_history: 记录每步目标函数值
    """
    params = [init.copy()]
    loss_history = [rosenbrock(init)]
    v = np.zeros_like(init)
    theta = init.copy()

    for i in range(iterations):
        # 预先校正：计算梯度在“未来”位置（theta - gamma * v）处的值
        lookahead = theta - gamma * v
        grad = grad_func(lookahead)
        # 更新动量项
        v = gamma * v + lr * grad
        # 更新参数
        theta = theta - v

        params.append(theta.copy())
        loss_history.append(rosenbrock(theta))

        if np.linalg.norm(grad) < tol:
            print(f"Nesterov Momentum converged after {i+1} iterations.")
            break

    return np.array(params), np.array(loss_history)

# 3. 参数设置和算法执行

# 初始参数（远离最优解的初始点，便于展示两种方法的迭代过程）
init_point = np.array([-1.5, 2.0])

# 超参数设置：
# 学习率 lr 较小保证收敛稳定性，gamma 设置为 0.9 较常见选择
learning_rate = 0.001
momentum_coeff = 0.9
max_iter = 10000
tolerance = 1e-6

# 执行标准 Momentum 优化器
params_momentum, loss_momentum = momentum_optimizer(rosenbrock_grad, init_point,
                                                      lr=learning_rate, gamma=momentum_coeff,
                                                      iterations=max_iter, tol=tolerance)

# 执行 Nesterov Momentum 优化器
params_nesterov, loss_nesterov = nesterov_optimizer(rosenbrock_grad, init_point,
                                                    lr=learning_rate, gamma=momentum_coeff,
                                                    iterations=max_iter, tol=tolerance)

# 4. 可视化：绘制 Rosenbrock 函数的等高线图和优化轨迹
x = np.linspace(-2, 2, 500)
y = np.linspace(-1, 3, 500)
X, Y = np.meshgrid(x, y)
Z = (1 - X)**2 + 100*(Y - X**2)**2

# 创建图形
plt.figure(figsize=(14, 6))

# 图1，等高线图及优化路径
plt.subplot(1, 2, 1)
# 绘制等高线，使用鲜艳的配色方案（例如采用热度图颜色，cmap='jet'）
contour = plt.contourf(X, Y, Z, levels=50, cmap=cm.jet, alpha=0.7)
plt.colorbar(contour)
# 绘制标准 Momentum 的轨迹
plt.plot(params_momentum[:, 0], params_momentum[:, 1], marker='o', markersize=3, color='red', label='Standard Momentum')
# 绘制 Nesterov Momentum 的轨迹
plt.plot(params_nesterov[:, 0], params_nesterov[:, 1], marker='x', markersize=3, color='lime', label='Nesterov Momentum')
plt.xlabel('X Coordinate', fontsize=12, fontweight='bold')
plt.ylabel('Y Coordinate', fontsize=12, fontweight='bold')
plt.title('Optimization Trajectory on Rosenbrock Function', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 图2，目标函数值随迭代次数的变化
plt.subplot(1, 2, 2)
iter_momentum = np.arange(len(loss_momentum))
iter_nesterov = np.arange(len(loss_nesterov))
plt.semilogy(iter_momentum, loss_momentum, color='magenta', label='Standard Momentum')
plt.semilogy(iter_nesterov, loss_nesterov, color='cyan', label='Nesterov Momentum')
plt.xlabel('Iterations', fontsize=12, fontweight='bold')
plt.ylabel('Objective Value (Log scale)', fontsize=12, fontweight='bold')
plt.title('Objective Value vs Iterations', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

'''
图1：Rosenbrock函数等高线图与优化轨迹，有助于直观理解优化过程，观察算法在凹凸不平的目标函数上怎样穿越“狭长谷底”。
图2：目标函数值随迭代次数的变化，直观反映了算法收敛的速度和稳定性，不仅在初始阶段具有较大下降幅度，在后期收敛时也能维持较为平稳的下降趋势。

算法收敛与动态调节
·本示例中超参数（学习率 lr 和动量系数 gamma）的设置对收敛速度影响较大。
·固定学习率虽然能最终收敛，但在某些阶段可能因为步长过大导致振荡；为此在实际应用中常采用动态衰减学习率的方法来控制更新步长。
·如果希望进一步提高收敛效率，可以对学习率进行指数衰减或者基于当前梯度大小进行自适应调整（例如 RMSProp、Adam 等优化算法）。
·此外，Nesterov 动量法本身就是一种改进优化技术，能够在计算梯度时提前探测未来位置，从而使更新更精确。因此，在很多深度学习任务中，Nesterov 动量法被证明有助于加快训练速度，达到更好的性能。

实验结论
·对于 Rosenbrock 这种具有明显非凸结构的目标函数，利用动量（Momentum）方法能够有效绕过局部剧烈振荡区域，提高优化效率。
·通过图形对比，可以直观感受到 Nesterov 加速技术在优化初期更为主动且收敛较快的特点。
·两种方法均达到了全局最优解附近，但在效率和路径选择上存在差异，这为实际问题中的算法选择提供了经验。
·此外，通过增加可视化以及分析算法中各参数变化趋势，可以帮助使用者更好地理解和调参，提高算法在特定场景下的应用性能。

在上述示例中，我们使用的是固定的学习率和动量系数。下面讨论两种常见的算法优化策略：
1.学习率衰减（Learning Rate Decay）
 · 在许多实际问题中，固定的学习率往往在训练初期表现较好，但在接近收敛时容易出现震荡，导致参数在最优值附近徘徊。
 · 可以采用学习率衰减策略，如指数衰减： lr = lr0 * exp(-decay_rate * t)
 · 这种方法有助于在接近最优解时降低步长，使迭代更稳定，有时可以显著提高最终模型的精度。
2.自适应动量系数调整
 · 在某些阶段，固定的动量系数 gamma 可能不适合所有梯度变化情况。比如在初期希望获得较大步长，而在后期则需要更精细的调整。
 · 自适应动量系数可以根据梯度变化情况进行动态调整，从而让算法在不同训练阶段达到更佳的收敛效果。
 · 目前较为流行的自适应优化算法（如 Adam、RMSProp 等）即在不同维度上引入了自适应调整策略，虽然代价更高，但通常能得到更好的收敛性能。
在本代码示例中，如果大家希望进一步优化，可以在动量算法中嵌入上述策略，使其更适应复杂的问题场景。
'''