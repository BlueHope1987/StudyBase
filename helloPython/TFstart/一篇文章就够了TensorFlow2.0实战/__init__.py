'''
https://www.jianshu.com/p/fa334fd76d2f
一篇文章就够了 TensorFlow 2.0 实战 (持续更新)

生态系统
TensorFlow 2.0
 @tf.function转换成计算图
TensorFlow Lite
TensorFlow.JS
TensorFlow Extended
TensorFlow Prob
TPU Cloud


1. 数据类型
数据载体
list支持不同的数据类型，效率低
np.array相同类型的载体，效率高，但是不支持GPU，不支持自动求导
tf.Tensortensorflow中存储大量连续数据的载体

基本数据类型
tf.int32: tf.constant(1)
tf.float32: tf.constant(1.)
tf.float64: tf.constant(1., dtype=tf.double)
tf.bool: tf.constant([True, False])
tf.string: tf.constant('hello')

'''
import  os
import  tensorflow as tf
import  numpy as np

print("======数据基本属性======")
with tf.device("cpu"):
    a=tf.range(4)
print("1 ", a.device) # '/job:localhost/replica:0/task:0/device:CPU:0'
# aa=a.gpu() 
print("2 ", a.numpy()) # array([0, 1, 2, 3], dtype=int32)
print("3 ", a.ndim) # 1  (0的话就是标量) rank和ndim的区别在于返回的类型不同
print("4 ", a.shape) # TensorShape([4])
#print("5 ", a.name) # AttributeError: Tensor.name is meaningless when eager execution is enabled. name属性在tensorflow2没有意义，因为变量名本身就是name
print("6 ", tf.rank(tf.ones([3,4,2])) )# <tf.Tensor: id=466672, shape=(), dtype=int32, numpy=3>
print("7 ", tf.is_tensor(a)) # True
print("8 ", a.dtype) # tf.int32

print("======数据类型转换======")
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

print("======可训练数据类型======")
a=tf.range(5)
b=tf.Variable(a)
b.dtype # tf.int32
b.name # 'Variable:0' 其实没啥用
b.trainable #True

'''
2. 创建Tensor
tf.convert_to_tensor(data)
tf.zeros(shape)
tf.ones(1)生成一个一维tensor，包含一个1
tf.ones([])生成一个标量1
tf.ones([2])生成一个一维tensor,包含两个1
tf.ones_like(a)相当于tf.ones(a.shape)
tf.fill([3,4], 9) 全部填充9
tf.random.normal([3,4], mean=1, stddev=1)
tf.random.truncated_normal([3,4], mean=0, stddev=1) 带截断的正态分布，（大于某个值重新采样），比如在经过sigmoid激活后，如果用不带截断的，容易出现梯度消失问题。
tf.random.uniform([3,4], minval=0, maxval=100, dtype=tf.int32) 平均分布
'''

idx=tf.range(5)
idx=tf.random.shuffle(idx)
a=tf.random.normal([10,784])
b=tf.random.uniform([10])
a=tf.gather(a, idx) # a中随机取5行
b=tf.gather(b, idx) # b中随机取5个


'''
三维tensor举例
x:[b,seq_len,word_dim] x:[b,5,5]
[imgs\2.1.png]
自然语言处理，b个句子，每个句子有5个单词，每个单词由5维向量表示
四维tensor：图像[b, h, w, c]
五维tensor：meta-learning [task_b, b, h, w, c] (多任务)
'''

#以下是自由活动时间
out=tf.random.uniform([4,10]) # 模拟4张图片的输出，每个输出对应10个分类
y=tf.range(4)
y=tf.one_hot(y, depth=10) # 模拟4张图片的真实分类
loss=tf.keras.losses.mse(y, out) 
loss=tf.reduce_mean(loss) # 计算loss

#一个简单的x@w+b
from tensorflow.keras import layers
net=layers.Dense(10)
net.build((4,8))  # 4 是batch_size, 前一层有8个units
net.kernel #w  shape=(8, 10)
net.bias #b  shape=(10, )
#记住：W的维度是[input_dim, output_dim], b的维度是[output_dim, ]
#自由活动结束

'''
3. Tensor操作
3.1 索引
基本：a[idx][idx][idx]
numpy风格：a[idx,idx,idx]可读性更强

3.2 切片
与numpy基本一致
a[start:end:positive_step]
a[end:start:negative_step]
a[0, 1, ..., 0] 代表任意多个: 只要能推断出有多少个:就是合法的

selective indexing
tf.gather
场景：对[4, 28, 28, 3]Tensor的第[3, 27, 9 ,13]行（也就是第一个28）顺序采样
使用：tf.gather(a, axis=1, indices=[3,27,9,13])

tf.gather_nd W3Cschool解释
场景：对[4, 28, 28, 3]Tensor第二维的[3, 27]和第三维的[20,8]进行采样
使用：tf.gather_nd(a, indices=[[:,3,20,:],[:,3,8,:],[:,27,20,:],[:,27,8,:]])

更多实例: [imgs\3.1.png] [imgs\3.2.png]

tf.boolean_mask
tf.boolean_mask(a, mask=[True, True, False], axis=3) 相当于只取RG两个通道的数据， a的shape是[4, 28, 28, 3]。mask可以是一个list，作用有点像tf.gather_nd


3.3 维度变换
a.shape, a.ndim
'''
#tf.transpose 比如交换图像的行列，也就是旋转90°
a=tf.random.normal([4, 3, 2, 1])
tf.transpose(a, perm=[0, 1, 3, 2]) #相当于交换最后两维

#tf.reshape
a=tf.random.normal([4, 28, 28, 3])
tf.reshape(a, [4, 784, 3])
tf.reshape(a, [4, -1, 3]) #效果和上面一样
tf.reshape(a, [4, -1]) 

#tf.expand_dims增加维度（dim和axis含义类似）
a=tf.random.normal([4, 35, 8])
tf.expand_dims(a, axis=3)  # 增加的维度是第4(3+1)维 shape是[4, 35, 8, 1]

#tf.squeeze维度压缩，默认去掉所有长度是1的维度，也可以通过axis指定某一个维度

'''
3.4 Broadcasting
Tensor运算的时候首先右对齐，插入维度，并将长度是1的维度扩张成相应的长度

图示 [imgs\3.4.png]

场景：一般情况下，高维度比低维度的概念更高层，如[班级，学生，成绩]，利用broadcasting把小维度推广到大维度。
作用：简洁、节省内存
tf.broadcast_to(a, [2,3,4])

3.5 合并与分割
tf.concat([a, b], axis=0) 在原来的维度上累加，要求其他维度的长度都相等。比如[4,35,8] concat [2,35,8] => [6,35,8]
tf.stack([a, b], axis=0) 在0维度处创建一个维度，长度为2 （因为这里只有a,b两个），要求所有维度的长度都相等
res=tf.unstack(c, axis=3) c的第3维上打散成多个张量，数量是这个维度的长度
tf.split(c, axis=3, num_or_size_splits=[2,3,2]比unstack更灵活
3.6 数据统计
tf.norm(a) 求a的范数，默认是二范数
tf.norm(a, ord=1, axis=1) 第一维看成一个整体，求一范数
tf.reduce_min reduce是为了提醒我们这些操作会降维
tf.reduce_max
tf.reduce_mean
tf.argmax(a) 默认返回axis=0上最大值的下标
tf.argmin(a)
tf.equak(a,b) 逐元素比较
tf.reduce_sum(tf.cast(tf.equal(a,b), dtype=tf.int32) 相当于统计相同元素的个数
tf.unique(a)返回一个数组和一个idx数组（用于反向生成a）

3.7 排序
tf.sort(a, direction='DESCENDING' 对最后一个维度进行排序
tf.argsort(a) 得到升序排列后元素在原数组中的下标
tf.gather(a, tf.argsort(a))
res=tf.math.top_k(a,2) res.indices res.value 用于topK accuracy

3.8 填充与复制
tf.pad(a, [[1,1],[1,1], ...]) 每一维上前后填充的数量
a=tf.random.normal([4,28,28,3])
b=tf.pad(a, [[0, 0], [2, 2], [2, 2], [0, 0]]) # 图片四周各填充两个像素
tf.tile(a, [ ])后面的参数指定每个维度复制的次数，1表示保持不变，2表示复制一次
 [a,b,c] -> [a,b,c,a,b,c]

3.9 张量限幅
tf.maximum(a, 2) 每个元素都会大于2， 简单的relu实现就用这个
tf.minimum(a, 8)
tf.clip_by_value(a, 2, 8)
new_grads, total_norm = tf.clip_by_globel_norm(grads, 15) 等比例放缩，不改变数据的分布，不影响梯度方向，可用于梯度消失，梯度爆炸

3.10 其他高级操作
indices=tf.where(a>0) 返回所有为True的坐标，配合tf.gather_nd(a, indices)使用
tf.where(cond, A, B) 根据cond,从A,B中挑选元素
tf.scatter_nd(indices, updates, shape)
 根据indics,把updates中的元素填充到shape大小的全零tensor中 [imgs\3.10.png]
points_x, points_y = tf.meshgrid(x, y)
points=tf.stack([points_x, points_y], axis=2
'''
#低层级方法实战MNIST tfstart_mnist_LowMethod.py

'''
4. 神经网络与全连接
4.1 数据加载
keras.datasets
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data() => numpy数组
y_onehot = tf.one_hot(y, depth=10)
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
tf.data.Dataset.from_tensor_slices

    db=tf.data.Dataset.from_tensor_slices((x, y)).batch(16).repeat(2) # 相当于数据翻了倍
    itr = iter(db)
    for i in range(10):
        print(next(itr)[0][15][16,16,0])  # batch中最后一张图中的一个像素

db=db.shuffle(10000)

4.2 全连接层
net=tf.keras.layers.Dense(units)
net.build(input_shape=(None, 784)) 根据输入shape创建net的所有变量 w ,b
net(x) #x是真正的输入
model = keras.Sequetial([keras,layers.Dense(2, activation='relu'), [keras,layers.Dense(4, activation='relu') ])
model.summary() 打印网络信息

4.3 输出方式
tf.sigmoid 保证输出在[0,1]中
prob=tf.nn.softmax(logits) 保证所有输出之和=1, logits一般指没有激活函数的最后一层的输出
tf.tanh输出在 [-1, 1]之间

4.4 误差计算
MSE
    tf.reduce_mean(tf.losses.MSE(y, out))
交叉熵 -log(q_i)
    tf.losses.categorical_crossentropy(y, logits, from_logits=True) 大多数情况下，使用from_logits参数，从而不用手动添加softmax
    tf.losses.binary_crossentropy(x, y)

5. 梯度下降、损失函数
导数 => 偏微分 某个坐标方向的导数=> 梯度所有坐标方向导数的集合

5.1 自动求梯度
    with tf.GradientTape() as tape:
        loss= ...
        [w_grad] = tape.gradiet(loss, [w])  # w是指定要求梯度的参数
with tf.GradientTape(persistent=True) as tape: 使得tape.gradient可被多次调用

求二阶导
    with tf.GradientTape() as t1:
        with tf.GradientTape() as t2:
            y = x * w + b
        dy_dw, dy_db = t2.gradient(y, [w, b])

    d2y_dw2 = t1.gradient(dy_dw, w)

5.2 反向传播
单输出感知机

'''

