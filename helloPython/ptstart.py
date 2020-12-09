'''
conda activate tf2
conda install pytorch=1.6.0
'''

#一文说清楚pytorch和tensorFlow的区别究竟在哪里
#https://blog.csdn.net/ibelieve8013/article/details/84261482
'''
实现计算图:
x*y=a a+z=b bΣc
'''

import torch
from torch.autograd import Variable
N,D=3,4
x=Variable(torch.randn(N,D),requires_grad=True)
y=Variable(torch.randn(N,D),requires_grad=True)
z=Variable(torch.randn(N,D),requires_grad=True)
a=x*y
b=a+z
c=torch.sum(b)
c.backward()
print(c)
print(x.grad.data)
print(y.grad.data)
print(z.grad.data)