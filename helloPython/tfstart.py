'''
配置GPU Tensorflow 2.4
创建 tf2环境 激活 安装cuDNN 7.6 CUDA 10 (默认应对应10.1 但据说存在问题 英伟达官网安装亦可https://developer.nvidia.com/cuda-toolkit-archive)
TensorFlow 与 cuda cudnn版本对应关系 https://blog.csdn.net/flyfor2013/article/details/107982839
新版的cpu和gpu版本是不单独区分的。而且这一版不需要单独安装Keras

conda activate tf2
[可选 cudnn依赖项]conda install cudatoolkit=10.1
conda install cudnn=7.6
conda install tensorflow=2.3
'''
import tensorflow as tf
import numpy as np

#测试
physical_devices = tf.config.list_physical_devices()
print(physical_devices)

print(tf.__version__)
v1 = tf.constant(8.0,dtype=tf.dtypes.float32)
v2 = tf.constant(2.0,dtype=tf.dtypes.float32)
v3 = tf.math.multiply(v1,v2)
print(f"result = {v3}")


#一文说清楚pytorch和tensorFlow的区别究竟在哪里
#https://blog.csdn.net/ibelieve8013/article/details/84261482
'''
实现计算图:
x*y=a a+z=b bΣc
'''
np.random.seed(0)
N,D=3,4
tf.compat.v1.disable_eager_execution() #防placeholder出错
x=tf.compat.v1.placeholder(tf.float32) #tf2的向下兼容 替代tf1的tf.placeholder
y=tf.compat.v1.placeholder(tf.float32)
z=tf.compat.v1.placeholder(tf.float32)
a=x*y
b=a+z
c=tf.reduce_sum(b)
grad_x, grad_y, grad_z=tf.gradients(c,[x,y,z])
with tf.compat.v1.Session() as sess:
    values={
        x:np.random.randn(N,D),
        y:np.random.randn(N,D),
        z:np.random.randn(N,D),
    }
    out=sess.run([c,grad_x,grad_y,grad_z],feed_dict=values)
    c_val,grad_x_val,grad_y_val,grad_z_val=out
print(c_val)
print(grad_x_val)
print(grad_y_val)
print(grad_z_val)





#一篇文章就够了 TensorFlow 2.0 实战 (持续更新) https://www.jianshu.com/p/fa334fd76d2f
# 一步一步学用Tensorflow构建卷积神经网络 https://www.jianshu.com/p/53d6cc6bbb25
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
l = tf.Variable(tf.zeros([5,6,5], tf.float32))
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
