import tensorflow as tf
import numpy as np


# 一步一步学用Tensorflow构建卷积神经网络 https://www.jianshu.com/p/53d6cc6bbb25
# https://developer.aliyun.com/article/178374
# https://blog.csdn.net/dyingstraw/article/details/80139343
# https://ataspinar.com/2017/08/15/building-convolutional-neural-networks-with-tensorflow/

################################################################################
#以下TF1代码TF2化 但在家里的电脑TF2环境下不能良好运行 TF1.15环境下运行正常 这里仅作测试 参考 TF1.15\tfstart.py
#LeNet5.py亦如此 TF2环境下卡在训练第一步

print("=========================")
'''
Tensorflow中最基本的单元是常量、变量和占位符。
tf.constant()和tf.Variable()之间的区别很清楚；一个常量有着恒定不变的值，一旦设置了它，它的值不能被改变。而变量的值可以在设置完成后改变，但变量的数据类型和形状无法改变。
'''
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
#l = tf.Variable(tf.zeros([5,6,5], tf.float32)) 变量冲突:单层网络会话中loss值
'''
除了tf.zeros()和tf.ones()能够创建一个初始值为0或1的张量（见这里）之外，还有一个tf.random_normal()函数，它能够创建一个包含多个随机值的张量，这些随机值是从正态分布中随机抽取的（默认的分布均值为0.0，标准差为1.0）。
另外还有一个tf.truncated_normal()函数，它创建了一个包含从截断的正态分布中随机抽取的值的张量，其中下上限是标准偏差的两倍。
有了这些知识，我们就可以创建用于神经网络的权重矩阵和偏差向量了。
'''
weights = tf.Variable(tf.compat.v1.truncated_normal([256 * 256, 10]))
biases = tf.Variable(tf.zeros([10]))
print(weights.get_shape().as_list())
print(biases.get_shape().as_list())
'''
在Tensorflow中，所有不同的变量以及对这些变量的操作都保存在图（Graph）中。在构建了一个包含针对模型的所有计算步骤的图之后，就可以在会话（Session）中运行这个图了。会话可以跨CPU和GPU分配所有的计算。
'''

#似乎有兼容性问题 无法运行
graph = tf.Graph()
with graph.as_default():
    a = tf.Variable(8, tf.float32)
    b = tf.Variable(tf.zeros([2,2], tf.float32))
with tf.compat.v1.Session(graph=graph) as session:
    tf.compat.v1.global_variables_initializer().run()
    print(f)
#    print(session.run(f))#出错！！！
#    print(session.run(k))#出错！！！

'''
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
    point1 = tf.compat.v1.placeholder(tf.float32, shape=(1, 2))
    point2 = tf.compat.v1.placeholder(tf.float32, shape=(1, 2))
    def calculate_eucledian_distance(point1, point2):
        difference = tf.subtract(point1, point2)
        power2 = tf.pow(difference, tf.constant(2.0, shape=(1,2)))
        add = tf.reduce_sum(power2)
        eucledian_distance = tf.sqrt(add)
        return eucledian_distance
    dist = calculate_eucledian_distance(point1, point2)
with tf.compat.v1.Session(graph=graph) as session:
    tf.compat.v1.global_variables_initializer().run()
    for ii in range(len(list_of_points1)):
        point1_ = list_of_points1[ii]
        point2_ = list_of_points2[ii]
        feed_dict = {point1 : point1_, point2 : point2_}
        distance = session.run([dist], feed_dict=feed_dict)
        print("the distance between {} and {} -> {}".format(point1_, point2_, distance))


'''
输入数据集：训练数据集和标签、测试数据集和标签（以及验证数据集和标签）。
测试和验证数据集可以放在tf.constant()中。而训练数据集被放在tf.placeholder()中，这样它可以在训练期间分批输入（随机梯度下降）。
神经网络模型及其所有的层。这可以是一个简单的完全连接的神经网络，仅由一层组成，或者由5、9、16层组成的更复杂的神经网络。
权重矩阵和偏差矢量以适当的形状进行定义和初始化。（每层一个权重矩阵和偏差矢量）
损失值：模型可以输出分对数矢量（估计的训练标签），并通过将分对数与实际标签进行比较，计算出损失值（具有交叉熵函数的softmax）。损失值表示估计训练标签与实际训练标签的接近程度，并用于更新权重值。
优化器：它用于将计算得到的损失值来更新反向传播算法中的权重和偏差。
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
    return tf.compat.v1.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    #argmax取最大值的索引 比较0内层1外层 最大值索引一致的累加
    #DEBUG检查点 数组维度不一致 已解决

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
神经网络最简单的形式是一层线性全连接神经网络（FCNN， Fully Connected Neural Network）。 在数学上它由一个矩阵乘法组成。
最好是在Tensorflow中从这样一个简单的NN开始，然后再去研究更复杂的神经网络。 当我们研究那些更复杂的神经网络的时候，只是图的模型（步骤2）和权重（步骤3）发生了改变，其他步骤仍然保持不变。
我们可以按照如下代码制作一层FCNN：
'''
#参考 https://www.cnblogs.com/imae/p/10629890.html
#没几步梯度爆炸什么的 调参什么的？

#代码可能是示意 无法调试良好稳定运行

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
tf.compat.v1.reset_default_graph() #DEBUG:恢复图
tf.random.set_seed(1) #DEBUG:尝试指定种子以固定结果

graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a Tensorflow friendly form.
    tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth)) #兼容性修改 下同
    tf_train_labels = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.compat.v1.constant(test_dataset, tf.float32)

    #2) Then, the weight matrices and bias vectors are initialized
    #as a default, tf.truncated_normal() is used for the weight matrix and tf.zeros() is used for the bias vector.
    weights = tf.Variable(tf.compat.v1.truncated_normal([image_width * image_height * image_depth, num_labels]), tf.float32) #truncated_normal 截断的产生正态分布的随机数
    bias = tf.Variable(tf.zeros([num_labels]), tf.float32) #偏置项 创建所有值为0的张量

    #3) define the model:
    #A one layered fccd simply consists of a matrix multiplication
    def model(data, weights, bias):
        return tf.compat.v1.matmul(flatten_tf_array(data), weights) + bias # matmul函数:矩阵相乘
    logits = model(tf_train_dataset, weights, bias)

    #4) calculate the loss, which will be used in the optimization of the weights
    loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    #reduce_mean降维平均值 loss是代价值，也就是我们要最小化的值

    #5) Choose an optimizer. Many are available.
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    #GradientDescentOptimizer 构造一个新的梯度下降优化器实例 learning_rate优化器将采用的学习速率 minimize梯度计算更新

    #optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss) #尝试替代选用Adma算法优化器 没有出众效果
    #检查：为何今天什么代码都没动 准确率上去了？71%~93%
    #其他操作：转移文件路径经测试无关 新建conda tf1.15环境并跑了两个高准确率范例？其后卡死 强制结束 svchost:SysMain (Superfetch)服务内存猛涨？
    #准确率在不同机器显示有巨大差异

    #6) The predicted values for the images in the train dataset and test dataset are assigned to the variables train_prediction and test_prediction.
    #It is only necessary if you want to know the accuracy by comparing it with the actual values.
    train_prediction = tf.compat.v1.nn.softmax(logits) #softmax 将一些输入映射为0-1之间的实数，并且归一化保证和为1
    test_prediction = tf.compat.v1.nn.softmax(model(tf_test_dataset, weights, bias))

#tf.compat.v1.summary.merge_all() #自动管理 应对某些可能出现的异常
with tf.compat.v1.Session(graph=graph) as session:
    #writer=tf.compat.v1.summary.FileWriter('tb_study', session.graph) #生成计算图 tensorboard --logdir=tb_study
    tf.compat.v1.global_variables_initializer().run() #初始化模型 global_variables_initializer会将权重设置为随机值
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
2.5 创建 LeNet5 卷积神经网络
下面我们将开始构建更多层的神经网络。例如LeNet5卷积神经网络。
LeNet5 CNN架构最早是在1998年由Yann Lecun（见论文http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf?spm=a2c6h.12873639.0.0.4dd17e76Xl9YXh&file=lecun-01a.pdf ）提出的。它是最早的CNN之一，专门用于对手写数字进行分类。尽管它在由大小为28 x 28的灰度图像组成的MNIST数据集上运行良好，但是如果用于其他包含更多图片、更大分辨率以及更多类别的数据集时，它的性能会低很多。对于这些较大的数据集，更深的ConvNets（如AlexNet、VGGNet或ResNet）会表现得更好。
但由于LeNet5架构仅由5个层构成，因此，学习如何构建CNN是一个很好的起点。

第1层：卷积层，包含S型激活函数，然后是平均池层。
第2层：卷积层，包含S型激活函数，然后是平均池层。
第3层：一个完全连接的网络（S型激活）
第4层：一个完全连接的网络（S型激活）
第5层：输出层
这意味着我们需要创建5个权重和偏差矩阵，我们的模型将由12行代码组成（5个层 + 2个池 + 4个激活函数 + 1个扁平层）。
由于这个还是有一些代码量的，因此最好在图之外的一个单独函数中定义这些代码。
'''
#由上面的示例拓展到LeNet5网络 LeNet5.py