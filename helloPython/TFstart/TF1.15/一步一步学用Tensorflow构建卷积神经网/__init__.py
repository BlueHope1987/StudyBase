import tensorflow as tf
import numpy as np


# 一步一步学用Tensorflow构建卷积神经网络 https://www.jianshu.com/p/53d6cc6bbb25
# https://developer.aliyun.com/article/178374
# https://blog.csdn.net/dyingstraw/article/details/80139343
# https://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-tensorflow/

'''
1.1 常量与变量
Tensorflow中最基本的单元是常量、变量和占位符。
tf.constant()和tf.Variable()之间的区别很清楚；一个常量有着恒定不变的值，一旦设置了它，它的值不能被改变。而变量的值可以在设置完成后改变，但变量的数据类型和形状无法改变。

a=tf.constant(2,tf.int16)
b=tf.constant(4,tf.float32)
c=tf.constant(8,tf.float32)
d = tf.Variable(2, tf.int16)
e = tf.Variable(4, tf.float32)
f = tf.Variable(8, tf.float32)
g = tf.constant(np.zeros(shape=(2,2), dtype=np.float32)) #does work
h = tf.zeros([11], tf.int16)
i = tf.ones([2,2], tf.float32)
j = tf.zeros([1000,4,3], tf.float64)
k = tf.Variable(tf.zeros([2,2], tf.float32))
l = tf.Variable(tf.zeros([5,6,5], tf.float32))

除了tf.zeros()和tf.ones()能够创建一个初始值为0或1的张量（见这里 https://www.tensorflow.org/api_guides/python/constant_op ）之外，还有一个tf.random_normal()函数，它能够创建一个包含多个随机值的张量，这些随机值是从正态分布中随机抽取的（默认的分布均值为0.0，标准差为1.0）。
另外还有一个tf.truncated_normal()函数，它创建了一个包含从截断的正态分布中随机抽取的值的张量，其中下上限是标准偏差的两倍。
有了这些知识，我们就可以创建用于神经网络的权重矩阵和偏差向量了。
'''

print("=========================")
weights = tf.Variable(tf.compat.v1.truncated_normal([256 * 256, 10]))
biases = tf.Variable(tf.zeros([10]))
print(weights.get_shape().as_list())
print(biases.get_shape().as_list())

'''
1.2 Tensorflow 中的图与会话
在Tensorflow中，所有不同的变量以及对这些变量的操作都保存在图（Graph）中。在构建了一个包含针对模型的所有计算步骤的图之后，就可以在会话（Session）中运行这个图了。会话可以跨CPU和GPU分配所有的计算。
'''

#似乎有兼容性问题 无法运行
graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2,2], tf.float32))
with tf.compat.v1.Session(graph=graph) as session:
    tf.compat.v1.global_variables_initializer().run()
    print(a)
    print(session.run(a))
    print(session.run(b))

'''
1.3 占位符 与 feed_dicts
我们已经看到了用于创建常量和变量的各种形式。Tensorflow中也有占位符，它不需要初始值，仅用于分配必要的内存空间。 在一个会话中，这些占位符可以通过feed_dict填入（外部）数据。
以下是占位符的使用示例。
'''

list_of_points1_ = [[1,2], [3,4], [5,6], [7,8]]
list_of_points2_ = [[15,16], [13,14], [11,12], [9,10]]
list_of_points1 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points1_])
list_of_points2 = np.array([np.array(elem).reshape(1,2) for elem in list_of_points2_])
graph = tf.Graph()
with graph.as_default():
    #we should use a tf.placeholder() to create a variable whose value you will fill in later (during session.run()).
    #this can be done by 'feeding' the data into the placeholder.
    #below we see an example of a method which uses two placeholder arrays of size [2,1] to calculate the eucledian distance
    point1 = tf.placeholder(tf.float32, shape=(1, 2))
    point2 = tf.placeholder(tf.float32, shape=(1, 2))
    def calculate_eucledian_distance(point1, point2):
        difference = tf.subtract(point1, point2)
        power2 = tf.pow(difference, tf.constant(2.0, shape=(1,2)))
        add = tf.reduce_sum(power2)
        eucledian_distance = tf.sqrt(add)
        return eucledian_distance
    dist = calculate_eucledian_distance(point1, point2)
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    for ii in range(len(list_of_points1)):
        point1_ = list_of_points1[ii]
        point2_ = list_of_points2[ii]
        feed_dict = {point1 : point1_, point2 : point2_}
        distance = session.run([dist], feed_dict=feed_dict)
        print("the distance between {} and {} -> {}".format(point1_, point2_, distance))

'''
2. Tensorflow 中的神经网络
2.1 简介
[imgs\2.1.png]
包含神经网络的图（如上图所示）应包含以下步骤：
输入数据集：训练数据集和标签、测试数据集和标签（以及验证数据集和标签）。
测试和验证数据集可以放在tf.constant()中。而训练数据集被放在tf.placeholder()中，这样它可以在训练期间分批输入（随机梯度下降）。
神经网络模型及其所有的层。这可以是一个简单的完全连接的神经网络，仅由一层组成，或者由5、9、16层组成的更复杂的神经网络。
权重矩阵和偏差矢量以适当的形状进行定义和初始化。（每层一个权重矩阵和偏差矢量）
损失值：模型可以输出分对数矢量（估计的训练标签），并通过将分对数与实际标签进行比较，计算出损失值（具有交叉熵函数的softmax）。损失值表示估计训练标签与实际训练标签的接近程度，并用于更新权重值。
优化器：它用于将计算得到的损失值来更新反向传播算法中的权重和偏差。
2.2 数据加载
下面我们来加载用于训练和测试神经网络的数据集。为此，我们要下载MNIST(http://yann.lecun.com/exdb/mnist/)和CIFAR-10(https://www.cs.toronto.edu/~kriz/cifar.html)数据集。 MNIST数据集包含了6万个手写数字图像，其中每个图像大小为28 x 28 x 1（灰度）。 CIFAR-10数据集也包含了6万个图像（3个通道），大小为32 x 32 x 3，包含10个不同的物体（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。 由于两个数据集中都有10个不同的对象，所以这两个数据集都包含10个标签。
[imgs\2.2.png]
首先，我们来定义一些方便载入数据和格式化数据的方法。
'''
#2. Tensorflow 中的神经网络 数据加载 我们来定义一些方便载入数据和格式化数据的方法。

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)
def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels
def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    #argmax取最大值的索引 比较0内层1外层 最大值索引一致的累加
    #DEBUG检查点 数组维度不一致 已解决
'''
这些方法可用于对标签进行独热码编码、将数据加载到随机数组中、扁平化矩阵（因为完全连接的网络需要一个扁平矩阵作为输入）：
在我们定义了这些必要的函数之后，我们就可以这样加载MNIST和CIFAR-10数据集了
'''
print("========加载MNIST和CIFAR-10数据集==========")
#pip install python-mnist #https://github.com/sorki/python-mnist
from mnist import MNIST
import pickle

mnist_folder = './helloPython/data/mnist/' #修改 项目根目录相对路径
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
mnist_image_size = 28 #debuging 后加
mndata = MNIST(mnist_folder)
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_size*mnist_image_size*1))
print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))
print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
######################################################################################
cifar10_folder = './helloPython/data/cifar10/' #修改 项目根目录相对路径
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size=32 #debuging 后加
with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes') #import pickle
c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size, c10_image_size, c10_image_depth)
c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_
c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size, c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels
print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print('Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)

#创建一个简单的一层神经网络
'''
你可以从Yann LeCun的网站(http://yann.lecun.com/exdb/mnist/)下载MNIST数据集。下载并解压缩之后，可以使用python-mnist(https://github.com/sorki/python-mnist)工具来加载数据。 CIFAR-10数据集可以从这里(https://www.cs.toronto.edu/~kriz/cifar.html)下载。

2.3 创建一个简单的一层神经网络
神经网络最简单的形式是一层线性全连接神经网络（FCNN， Fully Connected Neural Network）。 在数学上它由一个矩阵乘法组成。
最好是在Tensorflow中从这样一个简单的NN开始，然后再去研究更复杂的神经网络。 当我们研究那些更复杂的神经网络的时候，只是图的模型（步骤2）和权重（步骤3）发生了改变，其他步骤仍然保持不变。
我们可以按照如下代码制作一层FCNN：
'''
#参考 https://www.cnblogs.com/imae/p/10629890.html #没几步梯度爆炸什么的 调参什么的？
#代码可能是示意 已在tf1.15下调试良好

image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels
batch_size=32 ##debuging 后加 32->
#the dataset
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels
#number of iterations and learning rate
num_steps = 50 #debug 10001 -> 
display_step = 1 #debug 1000 ->
learning_rate = 0.01 #debug 0.5->
tf.reset_default_graph() #DEBUG:恢复图

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a Tensorflow friendly form.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth)) #兼容性修改 下同
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset, tf.float32)

    #2) Then, the weight matrices and bias vectors are initialized
    #as a default, tf.truncated_normal() is used for the weight matrix and tf.zeros() is used for the bias vector.
    weights = tf.Variable(tf.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32) #truncated_normal 截断的产生正态分布的随机数
    bias = tf.Variable(tf.zeros([num_labels]), tf.float32) #偏置项 创建所有值为0的张量

    #3) define the model:
    #A one layered fccd simply consists of a matrix multiplication
    def model(data, weights, bias):
        return tf.matmul(flatten_tf_array(data), weights) + bias # matmul函数:矩阵相乘
    logits = model(tf_train_dataset, weights, bias)

    #4) calculate the loss, which will be used in the optimization of the weights
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    #reduce_mean降维平均值 loss是代价值，也就是我们要最小化的值

    #5) Choose an optimizer. Many are available.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #GradientDescentOptimizer 构造一个新的梯度下降优化器实例 learning_rate优化器将采用的学习速率 minimize梯度计算更新

    #optimizer = tf.train.AdamOptimizer().minimize(loss) #尝试替代选用Adma算法优化器 没有出众效果
    #检查：为何今天什么代码都没动 准确率上去了？71%~93%
    #其他操作：转移文件路径经测试无关 新建conda tf1.15环境并跑了两个高准确率范例？其后卡死 强制结束 svchost:SysMain (Superfetch)服务内存猛涨？
    #准确率在不同机器显示有巨大差异
    #在tf1.15下表现良好

    #6) The predicted values for the images in the train dataset and test dataset are assigned to the variables train_prediction and test_prediction.
    #It is only necessary if you want to know the accuracy by comparing it with the actual values.
    train_prediction = tf.nn.softmax(logits) #softmax 将一些输入映射为0-1之间的实数，并且归一化保证和为1
    test_prediction = tf.nn.softmax(model(tf_test_dataset, weights, bias))

#tf.summary.merge_all() #自动管理 应对某些可能出现的异常
with tf.Session(graph=graph) as session:
    #writer=tf.summary.FileWriter('tb_study', session.graph) #生成计算图 tensorboard --logdir=tb_study
    tf.global_variables_initializer().run() #初始化模型 global_variables_initializer会将权重设置为随机值
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
        #新增debug
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        #新增结尾
        _, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)
        #Debug:新增 ",feed_dict=feed_dict" 及循环内定义逻辑 似乎结果不对 但必须投喂数据 以应对上述既定的placeholder
        if (step % display_step == 0):
            train_accuracy = accuracy(predictions,batch_labels[:, :]) #Debug: train_labels[:, :] -> batch_labels
            #DEBUG:数组维度不一致异常！！！ 已解决 train_labels与分片的predictions维度不一致 得用batch_labels如是维度 是否可以参照它？
            '''
            [m:n] #切片操作，取a[m]~a[n-1]之间的内容，m\n可以为负，m>n时返回空
            X[:,0]是numpy中数组的一种写法，表示对一个二维数组，取该二维数组第一维中的所有数据，第二维中取第0个数据，直观来说，X[:,0]就是取所有行的第0个数据, X[:,1] 就是取所有行的第1个数据。
            X[n,:]是取第1维中下标为n的元素的所有值。
            X[:,m:n]，即取所有数据的第m到n-1列数据，含左不含右
            '''
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)

'''
在图中，我们加载数据，定义权重矩阵和模型，从分对数矢量中计算损失值，并将其传递给优化器，该优化器将更新迭代“num_steps”次数的权重。
在上述完全连接的NN中，我们使用了梯度下降优化器来优化权重。然而，有很多不同的优化器(http://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-Tensorflow/#https://www.Tensorflow.org/api_guides/python/train%23Optimizers)可用于Tensorflow。 最常用的优化器有GradientDescentOptimizer、AdamOptimizer和AdaGradOptimizer，所以如果你正在构建一个CNN的话，我建议你试试这些。
Sebastian Ruder有一篇不错的博文(http://ruder.io/optimizing-gradient-descent/)介绍了不同优化器之间的区别，通过这篇文章，你可以更详细地了解它们。



2.4 Tensorflow的几个方面
Tensorflow包含许多层，这意味着可以通过不同的抽象级别来完成相同的操作。这里有一个简单的例子，操作
logits = tf.matmul(tf_train_dataset, weights) + biases，
也可以这样来实现
logits = tf.nn.xw_plus_b(train_dataset, weights, biases)。
这是layers API(https://www.tensorflow.org/tutorials/layers)中最明显的一层，它是一个具有高度抽象性的层，可以很容易地创建由许多不同层组成的神经网络。例如，conv_2d() (http://tflearn.org/layers/conv/?spm=a2c4e.11153940.blogcont178374.26.51cf5fe24l3akV#convolution-2d)或fully_connected() (http://tflearn.org/layers/core/#fully-connected)函数用于创建卷积和完全连接的层。通过这些函数，可以将层数、过滤器的大小或深度、激活函数的类型等指定为参数。然后，权重矩阵和偏置矩阵会自动创建，一起创建的还有激活函数和丢弃正则化层(dropout regularization laye)。
例如，通过使用 层API，下面这些代码：

import Tensorflow as tf
 
w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=0.1))
b1 = tf.Variable(tf.zeros([filter_depth]))
 
layer1_conv = tf.nn.conv2d(data, w1, [1, 1, 1, 1], padding='SAME')
layer1_relu = tf.nn.relu(layer1_conv + b1)
layer1_pool = tf.nn.max_pool(layer1_pool, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

可以替换为

from tflearn.layers.conv import conv_2d, max_pool_2d
 
layer1_conv = conv_2d(data, filter_depth, filter_size, activation='relu')
layer1_pool = max_pool_2d(layer1_conv_relu, 2, strides=2)

可以看到，我们不需要定义权重、偏差或激活(https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_)函数。尤其是在你建立一个具有很多层的神经网络的时候，这样可以保持代码的清晰和整洁。
然而，如果你刚刚接触Tensorflow的话，学习如何构建不同种类的神经网络并不合适，因为tflearn做了所有的工作。
因此，我们不会在本文中使用层API，但是一旦你完全理解了如何在Tensorflow中构建神经网络，我还是建议你使用它。
'''

'''
2.5 创建 LeNet5 卷积神经网络
下面我们将开始构建更多层的神经网络。例如LeNet5卷积神经网络。
LeNet5 CNN架构最早是在1998年由Yann Lecun（见论文http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf?spm=a2c6h.12873639.0.0.4dd17e76Xl9YXh&file=lecun-01a.pdf ）提出的。它是最早的CNN之一，专门用于对手写数字进行分类。尽管它在由大小为28 x 28的灰度图像组成的MNIST数据集上运行良好，但是如果用于其他包含更多图片、更大分辨率以及更多类别的数据集时，它的性能会低很多。对于这些较大的数据集，更深的ConvNets（如AlexNet、VGGNet或ResNet）会表现得更好。
但由于LeNet5架构仅由5个层构成，因此，学习如何构建CNN是一个很好的起点。
Lenet5架构如下图所示：
[imgs\2.5.png]
我们可以看到，它由5个层组成：
第1层：卷积层，包含S型激活函数，然后是平均池层。
第2层：卷积层，包含S型激活函数，然后是平均池层。
第3层：一个完全连接的网络（S型激活）
第4层：一个完全连接的网络（S型激活）
第5层：输出层
这意味着我们需要创建5个权重和偏差矩阵，我们的模型将由12行代码组成（5个层 + 2个池 + 4个激活函数 + 1个扁平层）。
由于这个还是有一些代码量的，因此最好在图之外的一个单独函数中定义这些代码。
'''
#由上面的示例拓展到LeNet5网络 LeNet5.py


'''
我们可以看到，LeNet5架构在MNIST数据集上的表现比简单的完全连接的NN更好。

2.6 影响层输出大小的参数
一般来说，神经网络的层数越多越好。我们可以添加更多的层、修改激活函数和池层，修改学习速率，以看看每个步骤是如何影响性能的。由于i层的输入是i-1层的输出，我们需要知道不同的参数是如何影响i-1层的输出大小的。

要了解这一点，可以看看conv2d() (https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)函数。

它有四个参数：

输入图像，维度为[batch size, image_width, image_height, image_depth]的4D张量
权重矩阵，维度为[filter_size, filter_size, image_depth, filter_depth]的4D张量
每个维度的步幅数。
填充（='SAME'/'VALID'）
这四个参数决定了输出图像的大小。

前两个参数分别是包含一批输入图像的4D张量和包含卷积滤波器权重的4D张量。

第三个参数是卷积的步幅，即卷积滤波器在四维的每一个维度中应该跳过多少个位置。这四个维度中的第一个维度表示图像批次中的图像编号，由于我们不想跳过任何图像，因此始终为1。最后一个维度表示图像深度（不是色彩的通道数；灰度为1，RGB为3），由于我们不想跳过任何颜色通道，所以这个也总是为1。第二和第三维度表示X和Y方向上的步幅（图像宽度和高度）。如果要应用步幅，则这些是过滤器应跳过的位置的维度。因此，对于步幅为1，我们必须将步幅参数设置为[1, 1, 1, 1]，如果我们希望步幅为2，则将其设置为[1，2，2，1]。以此类推。

最后一个参数表示Tensorflow是否应该对图像用零进行填充，以确保对于步幅为1的输出尺寸不会改变。如果 padding = 'SAME'，则图像用零填充（并且输出大小不会改变），如果 padding = 'VALID'，则不填充。
下面我们可以看到通过图像（大小为28 x 28）扫描的卷积滤波器（滤波器大小为5 x 5）的两个示例。
在左侧，填充参数设置为“SAME”，图像用零填充，最后4行/列包含在输出图像中。
在右侧，填充参数设置为“VALID”，图像不用零填充，最后4行/列不包括在输出图像中。
[imgs\2.6.gif]
我们可以看到，如果没有用零填充，则不包括最后四个单元格，因为卷积滤波器已经到达（非零填充）图像的末尾。这意味着，对于28 x 28的输入大小，输出大小变为24 x 24 。如果 padding = 'SAME'，则输出大小为28 x 28。
如果在扫描图像时记下过滤器在图像上的位置（为简单起见，只有X方向），那么这一点就变得更加清晰了。如果步幅为1，则X位置为0-5、1-6、2-7，等等。如果步幅为2，则X位置为0-5、2-7、4-9，等等。
如果图像大小为28 x 28，滤镜大小为5 x 5，并且步长1到4，那么我们可以得到下面这个表：
[imgs\2.6.png]
可以看到，对于步幅为1，零填充输出图像大小为28 x 28。如果非零填充，则输出图像大小变为24 x 24。对于步幅为2的过滤器，这几个数字分别为 14 x 14 和 12 x 12，对于步幅为3的过滤器，分别为 10 x 10 和 8 x 8。以此类推。
对于任意一个步幅S，滤波器尺寸K，图像尺寸W和填充尺寸P，输出尺寸将为
O = 1 + (W - K + 2P) / S
如果在Tensorflow中 padding = “SAME”，则分子加起来恒等于1，输出大小仅由步幅S


2.7 调整 LeNet5 的架构
在原始论文中，LeNet5架构使用了S形激活函数和平均池。 然而，现在，使用relu激活函数则更为常见。 所以，我们来稍稍修改一下LeNet5 CNN，看看是否能够提高准确性。我们将称之为类LeNet5架构
主要区别是我们使用了relu激活函数而不是S形激活函数。
除了激活函数，我们还可以改变使用的优化器，看看不同的优化器对精度的影响。
'''

#由上面的示例拓展到类LeNet5网络 likeLeNet5.py

'''
2.8 学习速率和优化器的影响
让我们来看看这些CNN在MNIST和CIFAR-10数据集上的表现。
[imgs\2.8.1.png]
[imgs\2.8.2.png]
在上面的图中，测试集的精度是迭代次数的函数。左侧为一层完全连接的NN，中间为LeNet5 NN，右侧为类LeNet5 NN。
可以看到，LeNet5 CNN在MNIST数据集上表现得非常好。这并不是一个大惊喜，因为它专门就是为分类手写数字而设计的。MNIST数据集很小，并没有太大的挑战性，所以即使是一个完全连接的网络也表现的很好。
然而，在CIFAR-10数据集上，LeNet5 NN的性能显着下降，精度下降到了40％左右。
为了提高精度，我们可以通过应用正则化或学习速率衰减来改变优化器，或者微调神经网络。
[imgs\2.8.3.png]
可以看到，AdagradOptimizer、AdamOptimizer和RMSPropOptimizer的性能比GradientDescentOptimizer更好。这些都是自适应优化器，其性能通常比GradientDescentOptimizer更好，但需要更多的计算能力。
通过L2正则化或指数速率衰减，我们可能会得到更搞的准确性，但是要获得更好的结果，我们需要进一步研究。



3. Tensorflow 中的深度神经网络
到目前为止，我们已经看到了LeNet5 CNN架构。 LeNet5包含两个卷积层，紧接着的是完全连接的层，因此可以称为浅层神经网络。那时候（1998年），GPU还没有被用来进行计算，而且CPU的功能也没有那么强大，所以，在当时，两个卷积层已经算是相当具有创新意义了。
后来，很多其他类型的卷积神经网络被设计出来，你可以在这里(http://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-Tensorflow/#DL_Literature)查看详细信息。
比如，由Alex Krizhevsky开发的非常有名的AlexNet(https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)架构（2012年），7层的ZF Net(http://arxiv.org/pdf/1311.2901v3.pdf)(2013)，以及16层的 VGGNet(http://arxiv.org/pdf/1409.1556v6.pdf)(2014)。
在2015年，Google发布了一个包含初始模块的22层的CNN（GoogLeNet(http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)），而微软亚洲研究院构建了一个152层的CNN，被称为ResNet(https://arxiv.org/pdf/1512.03385v1.pdf)。
现在，根据我们目前已经学到的知识，我们来看一下如何在Tensorflow中创建AlexNet和VGGNet16架构。
3.1 AlexNet
虽然LeNet5是第一个ConvNet，但它被认为是一个浅层神经网络。它在由大小为28 x 28的灰度图像组成的MNIST数据集上运行良好，但是当我们尝试分类更大、分辨率更好、类别更多的图像时，性能就会下降。
第一个深度CNN于2012年推出，称为AlexNet，其创始人为Alex Krizhevsky、Ilya Sutskever和Geoffrey Hinton。与最近的架构相比，AlexNet可以算是简单的了，但在当时它确实非常成功。它以令人难以置信的15.4％的测试错误率赢得了ImageNet比赛（亚军的误差为26.2％），并在全球深度学习和人工智能领域掀起了一场革命(https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/)。
它包括5个卷积层、3个最大池化层、3个完全连接层和2个丢弃层。整体架构如下所示：
[imgs\3.1.png]
第0层：大小为224 x 224 x 3的输入图像
第1层：具有96个滤波器（filter_depth_1 = 96）的卷积层，大小为11×11（filter_size_1 = 11），步长为4。它包含ReLU激活函数。
紧接着的是最大池化层和本地响应归一化层。
第2层：具有大小为5 x 5（filter_size_2 = 5）的256个滤波器（filter_depth_2 = 256）且步幅为1的卷积层。它包含ReLU激活函数。
紧接着的还是最大池化层和本地响应归一化层。
第3层：具有384个滤波器的卷积层（filter_depth_3 = 384），尺寸为3×3（filter_size_3 = 3），步幅为1。它包含ReLU激活函数
第4层：与第3层相同。
第5层：具有大小为3×3（filter_size_4 = 3）的256个滤波器（filter_depth_4 = 256）且步幅为1的卷积层。它包含ReLU激活函数
第6-8层：这些卷积层之后是完全连接层，每个层具有4096个神经元。在原始论文中，他们对1000个类别的数据集进行分类，但是我们将使用具有17个不同类别（的花卉）的oxford17数据集。
请注意，由于这些数据集中的图像太小，因此无法在MNIST或CIFAR-10数据集上使用此CNN（或其他的深度CNN）。正如我们以前看到的，一个池化层（或一个步幅为2的卷积层）将图像大小减小了2倍。 AlexNet具有3个最大池化层和一个步长为4的卷积层。这意味着原始图像尺寸会缩小2^5。 MNIST数据集中的图像将简单地缩小到尺寸小于0。
因此，我们需要加载具有较大图像的数据集，最好是224 x 224 x 3（如原始文件所示）。 17个类别的花卉数据集，又名oxflower17数据集(http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)是最理想的，因为它包含了这个大小的图像：

ox17_image_width = 224
ox17_image_height = 224
ox17_image_depth = 3
ox17_num_labels = 17
 
import tflearn.datasets.oxflower17 as oxflower17
train_dataset_, train_labels_ = oxflower17.load_data(one_hot=True)
train_dataset_ox17, train_labels_ox17 = train_dataset_[:1000,:,:,:], train_labels_[:1000,:]
test_dataset_ox17, test_labels_ox17 = train_dataset_[1000:,:,:,:], train_labels_[1000:,:]
 
print('Training set', train_dataset_ox17.shape, train_labels_ox17.shape)
print('Test set', test_dataset_ox17.shape, test_labels_ox17.shape)

让我们试着在AlexNet中创建权重矩阵和不同的层。正如我们之前看到的，我们需要跟层数一样多的权重矩阵和偏差矢量，并且每个权重矩阵的大小应该与其所属层的过滤器的大小相对应。

ALEX_PATCH_DEPTH_1, ALEX_PATCH_DEPTH_2, ALEX_PATCH_DEPTH_3, ALEX_PATCH_DEPTH_4 = 96, 256, 384, 256
ALEX_PATCH_SIZE_1, ALEX_PATCH_SIZE_2, ALEX_PATCH_SIZE_3, ALEX_PATCH_SIZE_4 = 11, 5, 3, 3
ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096
 
 
def variables_alexnet(patch_size1 = ALEX_PATCH_SIZE_1, patch_size2 = ALEX_PATCH_SIZE_2, 
                      patch_size3 = ALEX_PATCH_SIZE_3, patch_size4 = ALEX_PATCH_SIZE_4, 
                      patch_depth1 = ALEX_PATCH_DEPTH_1, patch_depth2 = ALEX_PATCH_DEPTH_2, 
                      patch_depth3 = ALEX_PATCH_DEPTH_3, patch_depth4 = ALEX_PATCH_DEPTH_4, 
                      num_hidden1 = ALEX_NUM_HIDDEN_1, num_hidden2 = ALEX_NUM_HIDDEN_2,
                      image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
 
    w1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
 
    w2 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))
 
    w3 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth2, patch_depth3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([patch_depth3]))
 
    w4 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[patch_depth3]))
 
    w5 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b5 = tf.Variable(tf.zeros([patch_depth3]))
 
    pool_reductions = 3
    conv_reductions = 2
    no_reductions = pool_reductions + conv_reductions
    w6 = tf.Variable(tf.truncated_normal([(image_width // 2**no_reductions)*(image_height // 2**no_reductions)*patch_depth3, num_hidden1], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
 
    w7 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
 
    w8 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
 
    variables = {
                 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 
                 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8
                }
    return variables
 
 
def model_alexnet(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 4, 4, 1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer1_norm = tf.nn.local_response_normalization(layer1_pool)
 
    layer2_conv = tf.nn.conv2d(layer1_norm, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer2_norm = tf.nn.local_response_normalization(layer2_pool)
 
    layer3_conv = tf.nn.conv2d(layer2_norm, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])
 
    layer4_conv = tf.nn.conv2d(layer3_relu, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])
 
    layer5_conv = tf.nn.conv2d(layer4_relu, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_relu = tf.nn.relu(layer5_conv + variables['b5'])
    layer5_pool = tf.nn.max_pool(layer4_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer5_norm = tf.nn.local_response_normalization(layer5_pool)
 
    flat_layer = flatten_tf_array(layer5_norm)
    layer6_fccd = tf.matmul(flat_layer, variables['w6']) + variables['b6']
    layer6_tanh = tf.tanh(layer6_fccd)
    layer6_drop = tf.nn.dropout(layer6_tanh, 0.5)
 
    layer7_fccd = tf.matmul(layer6_drop, variables['w7']) + variables['b7']
    layer7_tanh = tf.tanh(layer7_fccd)
    layer7_drop = tf.nn.dropout(layer7_tanh, 0.5)
 
    logits = tf.matmul(layer7_drop, variables['w8']) + variables['b8']
    return logits


现在我们可以修改CNN模型来使用AlexNet模型的权重和层次来对图像进行分类。

'''


'''
3.2 VGG Net-16
VGG Net于2014年由牛津大学的Karen Simonyan和Andrew Zisserman创建出来。 它包含了更多的层（16-19层），但是每一层的设计更为简单；所有卷积层都具有3×3以及步长为3的过滤器，并且所有最大池化层的步长都为2。
所以它是一个更深的CNN，但更简单。
它存在不同的配置，16层或19层。 这两种不同配置之间的区别是在第2，第3和第4最大池化层之后对3或4个卷积层的使用（见下文）。
[imgs\3.2.png]
配置为16层（配置D）的结果似乎更好，所以我们试着在Tensorflow中创建它。

#The VGGNET Neural Network 
VGG16_PATCH_SIZE_1, VGG16_PATCH_SIZE_2, VGG16_PATCH_SIZE_3, VGG16_PATCH_SIZE_4 = 3, 3, 3, 3
VGG16_PATCH_DEPTH_1, VGG16_PATCH_DEPTH_2, VGG16_PATCH_DEPTH_3, VGG16_PATCH_DEPTH_4 = 64, 128, 256, 512
VGG16_NUM_HIDDEN_1, VGG16_NUM_HIDDEN_2 = 4096, 1000
 
def variables_vggnet16(patch_size1 = VGG16_PATCH_SIZE_1, patch_size2 = VGG16_PATCH_SIZE_2, 
                       patch_size3 = VGG16_PATCH_SIZE_3, patch_size4 = VGG16_PATCH_SIZE_4, 
                       patch_depth1 = VGG16_PATCH_DEPTH_1, patch_depth2 = VGG16_PATCH_DEPTH_2, 
                       patch_depth3 = VGG16_PATCH_DEPTH_3, patch_depth4 = VGG16_PATCH_DEPTH_4,
                       num_hidden1 = VGG16_NUM_HIDDEN_1, num_hidden2 = VGG16_NUM_HIDDEN_2,
                       image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
    
    w1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
    w2 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, patch_depth1, patch_depth1], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth1]))
 
    w3 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth1, patch_depth2], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [patch_depth2]))
    w4 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth2, patch_depth2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [patch_depth2]))
    
    w5 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth2, patch_depth3], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [patch_depth3]))
    w6 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth3, patch_depth3], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [patch_depth3]))
    w7 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth3, patch_depth3], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape=[patch_depth3]))
 
    w8 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth4], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w9 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b9 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w10 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b10 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    
    w11 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b11 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w12 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b12 = tf.Variable(tf.constant(1.0, shape=[patch_depth4]))
    w13 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b13 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    
    no_pooling_layers = 5
 
    w14 = tf.Variable(tf.truncated_normal([(image_width // (2**no_pooling_layers))*(image_height // (2**no_pooling_layers))*patch_depth4 , num_hidden1], stddev=0.1))
    b14 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
    
    w15 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b15 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
   
    w16 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b16 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10, 
        'w11': w11, 'w12': w12, 'w13': w13, 'w14': w14, 'w15': w15, 'w16': w16, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8, 'b9': b9, 'b10': b10, 
        'b11': b11, 'b12': b12, 'b13': b13, 'b14': b14, 'b15': b15, 'b16': b16
    }
    return variables
 
def model_vggnet16(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer2_conv = tf.nn.conv2d(layer1_actv, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer3_conv = tf.nn.conv2d(layer2_pool, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_actv = tf.nn.relu(layer3_conv + variables['b3'])   
    layer4_conv = tf.nn.conv2d(layer3_actv, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_actv = tf.nn.relu(layer4_conv + variables['b4'])
    layer4_pool = tf.nn.max_pool(layer4_pool, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer5_conv = tf.nn.conv2d(layer4_pool, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_actv = tf.nn.relu(layer5_conv + variables['b5'])
    layer6_conv = tf.nn.conv2d(layer5_actv, variables['w6'], [1, 1, 1, 1], padding='SAME')
    layer6_actv = tf.nn.relu(layer6_conv + variables['b6'])
    layer7_conv = tf.nn.conv2d(layer6_actv, variables['w7'], [1, 1, 1, 1], padding='SAME')
    layer7_actv = tf.nn.relu(layer7_conv + variables['b7'])
    layer7_pool = tf.nn.max_pool(layer7_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer8_conv = tf.nn.conv2d(layer7_pool, variables['w8'], [1, 1, 1, 1], padding='SAME')
    layer8_actv = tf.nn.relu(layer8_conv + variables['b8'])
    layer9_conv = tf.nn.conv2d(layer8_actv, variables['w9'], [1, 1, 1, 1], padding='SAME')
    layer9_actv = tf.nn.relu(layer9_conv + variables['b9'])
    layer10_conv = tf.nn.conv2d(layer9_actv, variables['w10'], [1, 1, 1, 1], padding='SAME')
    layer10_actv = tf.nn.relu(layer10_conv + variables['b10'])
    layer10_pool = tf.nn.max_pool(layer10_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer11_conv = tf.nn.conv2d(layer10_pool, variables['w11'], [1, 1, 1, 1], padding='SAME')
    layer11_actv = tf.nn.relu(layer11_conv + variables['b11'])
    layer12_conv = tf.nn.conv2d(layer11_actv, variables['w12'], [1, 1, 1, 1], padding='SAME')
    layer12_actv = tf.nn.relu(layer12_conv + variables['b12'])
    layer13_conv = tf.nn.conv2d(layer12_actv, variables['w13'], [1, 1, 1, 1], padding='SAME')
    layer13_actv = tf.nn.relu(layer13_conv + variables['b13'])
    layer13_pool = tf.nn.max_pool(layer13_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer  = flatten_tf_array(layer13_pool)
    layer14_fccd = tf.matmul(flat_layer, variables['w14']) + variables['b14']
    layer14_actv = tf.nn.relu(layer14_fccd)
    layer14_drop = tf.nn.dropout(layer14_actv, 0.5)
    
    layer15_fccd = tf.matmul(layer14_drop, variables['w15']) + variables['b15']
    layer15_actv = tf.nn.relu(layer15_fccd)
    layer15_drop = tf.nn.dropout(layer15_actv, 0.5)
    
    logits = tf.matmul(layer15_drop, variables['w16']) + variables['b16']
    return logits


#The VGGNET Neural Network 
VGG16_PATCH_SIZE_1, VGG16_PATCH_SIZE_2, VGG16_PATCH_SIZE_3, VGG16_PATCH_SIZE_4 = 3, 3, 3, 3
VGG16_PATCH_DEPTH_1, VGG16_PATCH_DEPTH_2, VGG16_PATCH_DEPTH_3, VGG16_PATCH_DEPTH_4 = 64, 128, 256, 512
VGG16_NUM_HIDDEN_1, VGG16_NUM_HIDDEN_2 = 4096, 1000
 
def variables_vggnet16(patch_size1 = VGG16_PATCH_SIZE_1, patch_size2 = VGG16_PATCH_SIZE_2, 
                       patch_size3 = VGG16_PATCH_SIZE_3, patch_size4 = VGG16_PATCH_SIZE_4, 
                       patch_depth1 = VGG16_PATCH_DEPTH_1, patch_depth2 = VGG16_PATCH_DEPTH_2, 
                       patch_depth3 = VGG16_PATCH_DEPTH_3, patch_depth4 = VGG16_PATCH_DEPTH_4,
                       num_hidden1 = VGG16_NUM_HIDDEN_1, num_hidden2 = VGG16_NUM_HIDDEN_2,
                       image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
    
    w1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
    w2 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, patch_depth1, patch_depth1], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth1]))
 
    w3 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth1, patch_depth2], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [patch_depth2]))
    w4 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth2, patch_depth2], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [patch_depth2]))
    
    w5 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth2, patch_depth3], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [patch_depth3]))
    w6 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth3, patch_depth3], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [patch_depth3]))
    w7 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth3, patch_depth3], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape=[patch_depth3]))
 
    w8 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth4], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w9 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b9 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w10 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b10 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    
    w11 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b11 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    w12 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b12 = tf.Variable(tf.constant(1.0, shape=[patch_depth4]))
    w13 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth4, patch_depth4], stddev=0.1))
    b13 = tf.Variable(tf.constant(1.0, shape = [patch_depth4]))
    
    no_pooling_layers = 5
 
    w14 = tf.Variable(tf.truncated_normal([(image_width // (2**no_pooling_layers))*(image_height // (2**no_pooling_layers))*patch_depth4 , num_hidden1], stddev=0.1))
    b14 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
    
    w15 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b15 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
   
    w16 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b16 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10, 
        'w11': w11, 'w12': w12, 'w13': w13, 'w14': w14, 'w15': w15, 'w16': w16, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8, 'b9': b9, 'b10': b10, 
        'b11': b11, 'b12': b12, 'b13': b13, 'b14': b14, 'b15': b15, 'b16': b16
    }
    return variables
 
def model_vggnet16(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer2_conv = tf.nn.conv2d(layer1_actv, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer3_conv = tf.nn.conv2d(layer2_pool, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_actv = tf.nn.relu(layer3_conv + variables['b3'])   
    layer4_conv = tf.nn.conv2d(layer3_actv, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_actv = tf.nn.relu(layer4_conv + variables['b4'])
    layer4_pool = tf.nn.max_pool(layer4_pool, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer5_conv = tf.nn.conv2d(layer4_pool, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_actv = tf.nn.relu(layer5_conv + variables['b5'])
    layer6_conv = tf.nn.conv2d(layer5_actv, variables['w6'], [1, 1, 1, 1], padding='SAME')
    layer6_actv = tf.nn.relu(layer6_conv + variables['b6'])
    layer7_conv = tf.nn.conv2d(layer6_actv, variables['w7'], [1, 1, 1, 1], padding='SAME')
    layer7_actv = tf.nn.relu(layer7_conv + variables['b7'])
    layer7_pool = tf.nn.max_pool(layer7_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer8_conv = tf.nn.conv2d(layer7_pool, variables['w8'], [1, 1, 1, 1], padding='SAME')
    layer8_actv = tf.nn.relu(layer8_conv + variables['b8'])
    layer9_conv = tf.nn.conv2d(layer8_actv, variables['w9'], [1, 1, 1, 1], padding='SAME')
    layer9_actv = tf.nn.relu(layer9_conv + variables['b9'])
    layer10_conv = tf.nn.conv2d(layer9_actv, variables['w10'], [1, 1, 1, 1], padding='SAME')
    layer10_actv = tf.nn.relu(layer10_conv + variables['b10'])
    layer10_pool = tf.nn.max_pool(layer10_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer11_conv = tf.nn.conv2d(layer10_pool, variables['w11'], [1, 1, 1, 1], padding='SAME')
    layer11_actv = tf.nn.relu(layer11_conv + variables['b11'])
    layer12_conv = tf.nn.conv2d(layer11_actv, variables['w12'], [1, 1, 1, 1], padding='SAME')
    layer12_actv = tf.nn.relu(layer12_conv + variables['b12'])
    layer13_conv = tf.nn.conv2d(layer12_actv, variables['w13'], [1, 1, 1, 1], padding='SAME')
    layer13_actv = tf.nn.relu(layer13_conv + variables['b13'])
    layer13_pool = tf.nn.max_pool(layer13_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer  = flatten_tf_array(layer13_pool)
    layer14_fccd = tf.matmul(flat_layer, variables['w14']) + variables['b14']
    layer14_actv = tf.nn.relu(layer14_fccd)
    layer14_drop = tf.nn.dropout(layer14_actv, 0.5)
    
    layer15_fccd = tf.matmul(layer14_drop, variables['w15']) + variables['b15']
    layer15_actv = tf.nn.relu(layer15_fccd)
    layer15_drop = tf.nn.dropout(layer15_actv, 0.5)
    
    logits = tf.matmul(layer15_drop, variables['w16']) + variables['b16']
    return logits


3.3 AlexNet 性能
作为比较，看一下对包含了较大图片的oxflower17数据集的LeNet5 CNN性能：
[imgs\3.3.png]
4. 结语
相关代码可以在我的GitHub库(https://github.com/taspinar/sidl)中获得，因此可以随意在自己的数据集上使用它。
在深度学习的世界中还有更多的知识可以去探索：循环神经网络、基于区域的CNN、GAN、加强学习等等。在未来的博客文章中，我将构建这些类型的神经网络，并基于我们已经学到的知识构建更有意思的应用程序。
文章原标题《Building Convolutional Neural Networks with Tensorflow》，作者：Ahmet Taspinar，译者：夏天，审校：主题曲。
文章为简译，更为详细的内容，请查看原文(http://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-Tensorflow/)
'''