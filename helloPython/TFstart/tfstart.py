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
