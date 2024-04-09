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
