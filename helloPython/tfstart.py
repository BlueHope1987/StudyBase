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
print("=========================")

#tf基本数据类型 不可测试？
with tf.device("cpu"):
    a=tf.range(4)
a.device # '/job:localhost/replica:0/task:0/device:CPU:0'
aa=a.gpu() #出错
a.numpy() # array([0, 1, 2, 3], dtype=int32)
a.ndim # 1  (0的话就是标量)
a.shape # TensorShape([4])
a.name # AttributeError: Tensor.name is meaningless when eager execution is enabled. 
tf.rank(tf.ones([3,4,2])) # <tf.Tensor: id=466672, shape=(), dtype=int32, numpy=3>
tf.is_tensor(a) # True
a.dtype # tf.int32

#数据类型转换
a=np.arange(5)
a.dtype # dtype('int64')
aa=tf.convert_to_tensor(a) # <tf.Tensor: id=466678, shape=(5,), dtype=int64, numpy=array([0, 1, 2, 3, 4])>
aa=tf.convert_to_tensor(a, dtype=tf.int32) # <tf.Tensor: id=466683, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4], dtype=int32)>
tf.cast(aa, tf.float32)
b=tf.constant([0,1])
tf.cast(b, tf.bool) # <tf.Tensor: id=466697, shape=(2,), dtype=bool, numpy=array([False,  True])>
a.tf.ones([])
a.numpy()
int(a) #标量可以直接这样类型转换
float(a)

#可训练数据类型
a=tf.range(5)
b=tf.Variable(a)
b.dtype # tf.int32
b.name # 'Variable:0' 其实没啥用
b.trainable #True