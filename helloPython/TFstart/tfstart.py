'''
TensorFlow 的GitHub
https://github.com/tensorflow/tensorflow

默认需要AVX指令集 若处理器不支持需自行编译或找兼容版本
https://github.com/fo40225/tensorflow-windows-wheel


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



#tf.Variable 的定义：tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)
'''
在TensorFlow的世界里，变量的定义和初始化是分开的，一开始，tf.Variable 得到的是张量，而张量并不是具体的值，而是计算过程。
因为tf.Variable 生成的是一个张量，那么 name 就是一个张量的名字，如果你不主动声明的话，就是默认的 Variable
而如果你要得到，变量的值的话，那么你就需要对张量进行计算，首先对变量进行初始化，使用会话进行计算
对变量初始化之后，就可以直接计算变量，那么run 变量，那么就得到了变量的值
Trainable 属性作用? 就是通过trainable属性控制这个变量是否可以被优化器更新,比如有的变量并不是一个常数，而是一个 正态分布，那么优化器，就可以对这个变量进行更新和优化
'''
a=tf.Variable(5)
print(a)
 
b=tf.Variable(10)
print(b)
 
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))