import numpy as np

a=np.array([2,1])
print(a)
type(a)
c=np.array([[1,1],[3,4]])
print(c)
d=np.array([[1],[2]])
print(d)

print("转置变量")
print(d.T) #转置变量用 .T
print(d.T.T)

a=np.array([2.1])
b=np.array([1,2])
print(a+b)

print("4-1-(9)")
print(2*a)

print("4-1-(10)")
b=np.array([1,3])
c=np.array([4,2])
print(b.dot(c))

print("4-1-(11) 向量的模")
a=np.array([1,3])
print(np.linalg.norm(a))

print("4-2-(1)求和符号")
a=np.ones(1000) #[1 1 1 ... 1]
b=np.arange(1,1001) #[1 2 3 ... 1000]
print(a.dot(b))

print("4-2-(2)绘制梯度的图形")
import matplotlib.pyplot as plt
def f(w0,w1):
    return w0**2+w0*w1+3
def df_dw0(w0,w1):
    return 2*w0+2*w1
def df_dw1(w0,w1):
    return 2*w0+0*w1

w_range=2
dw=0.25
w0=np.arange(-w_range,w_range+dw,dw)
w1=np.arange(-w_range,w_range+dw,dw)

ww0, ww1=np.meshgrid(w0,w1)

ff=np.zeros((len(w0),len(w1)))
dff_dw0=np.zeros((len(w0),len(w1)))
dff_dw1=np.zeros((len(w0),len(w1)))
for i0 in range(len(w0)):
    for i1 in range(len(w1)):
        ff[i1,i0]=f(w0[i0],w1[i1])
        dff_dw0[i1,i0]=df_dw0(w0[i0],w1[i1])
        dff_dw1[i1,i0]=df_dw1(w0[i0],w1[i1])

plt.figure(figsize=(9,4))
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)
cont=plt.contour(ww0,ww1,ff,10,colors='k')
cont.clabel(fmt='%d',fontsize=8)
plt.xticks(range(-w_range,w_range+1,1))
plt.yticks(range(-w_range,w_range+1,1))
plt.xlim(-w_range-0.5,w_range+0.5)
plt.ylim(-w_range-0.5,w_range+0.5)

plt.subplot(1,2,2)
plt.quiver(ww0,ww1,dff_dw0,dff_dw1)
plt.xlabel('$w_0$',fontsize=14)
plt.ylabel('$w_1$',fontsize=14)
plt.xticks(range(-w_range,w_range+1,1))
plt.yticks(range(-w_range,w_range+1,1))
plt.xlim(-w_range-0.5,w_range+0.5)
plt.ylim(-w_range-0.5,w_range+0.5)
plt.show()
