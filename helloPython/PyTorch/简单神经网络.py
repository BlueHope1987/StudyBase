import torch
from torch import nn
# Create a tensor  

model=nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
input_tensor = torch.randn(1, 784)  # Example input tensor with batch size 1 and 784 features
output_tensor = model(input_tensor)  # Forward pass through the model  
print(output_tensor.shape)  # Output shape should be (1, 10) for 10 classes
# Create a model and perform a forward pass

#引用文章 https://www.acwing.com/blog/content/54799/

#在上面的例子中，nn.Sequential包含了两个nn.Linear层和一个nn.ReLU激活函数。输入数据首先通过第一个线性层（全连接层），
#然后经过ReLU激活函数，最后通过第二个线性层。每个子模块都会按照它们在nn.Sequential构造函数中的顺序被调用。
#nn.Sequential的参数是一个子模块的列表（或其他可迭代对象）。这些子模块可以是任何继承自nn.Module的类实例，
#比如nn.Linear、nn.Conv2d、nn.ReLU、nn.Dropout等。每个子模块都会按照它们在列表中的顺序被调用，并且数据会在它们之间传递。
#需要注意的是，nn.Sequential不会添加任何额外的层（如池化层或批量归一化层），这些层需要单独添加到模型中。
#此外，虽然nn.Sequential使得代码更加简洁，但对于复杂的网络结构（如包含分支或跳跃连接的网络），
#你可能需要使用nn.Module来定义自己的模型类，并在其中定义forward方法。


# 一个简单神经网络 输入层784个神经元，通常对应于一个28*28像素的图像。神经元的开始，输入数据被连接到输入层的每个节点。
# 隐藏层128个神经元，输入层的数据首先通过一个线性变换（nn.Linear(784, 128)）传递到隐藏层，
# 然后经过一个ReLU激活函数（nn.ReLU()）。ReLU函数会对每个节点的输出应用非线性变换，使得神经网络能够学习复杂的模式。
# 输出层：有10个节点，这通常对应于一个分类任务中的10个不同类别。隐藏层的数据通过一个线性变换（nn.Linear(128, 10)）传递到输出层，输出层将产生每个类别的分数或概率。
# 在实际应用中，每个节点（或神经元）都与其前一层的所有节点以及其后一层的所有节点相连（全连接），并通过权重和偏置参数进行转换。这些权重和偏置参数在训练过程中会被优化，以最小化预测误差。输出层10个神经元