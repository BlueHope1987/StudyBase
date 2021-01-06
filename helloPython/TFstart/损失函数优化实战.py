#一篇文章就够了 TensorFlow 2.0 实战 (持续更新) https://www.jianshu.com/p/fa334fd76d2f
#损失函数优化实战

import  numpy as np
from    matplotlib import pyplot as plt
from    mpl_toolkits.mplot3d import Axes3D

def loss(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
Z = loss([X, Y])

fig = plt.figure('loss')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(45, -60)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


import  tensorflow as tf
xx = tf.constant([0., 0.])

for step in range(200):

    with tf.GradientTape() as tape:
        tape.watch([xx])
        yy = loss(xx)

    grads = tape.gradient(yy, [xx])[0]  # y 对 x求导
    xx -= 0.01*grads

    if step % 20 == 0:
        print ('step {}: xx = {}, f(xx) = {}'
               .format(step, xx.numpy(), yy.numpy()))