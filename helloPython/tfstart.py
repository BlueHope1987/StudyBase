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
    print(session.run(f))
    print(session.run(k))
