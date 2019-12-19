
#https://github.com/microsoft/ai-edu/blob/master/A-%E6%95%99%E5%AD%A6%E8%AF%BE%E7%A8%8B/math_intro/01_%E4%BB%A3%E6%95%B0.ipynb

import numpy as np
from tabulate import tabulate

x = np.array(range(-10, 11))  # 从 -10 到 10 的21个数据点
y = (3 * x - 4) / 2           # 对应的函数值

print(tabulate(np.column_stack((x,y)), headers=['x', 'y']))




from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
# %matplotlib inline

from matplotlib import pyplot as plt

plt.plot(x, y, color="grey", marker = "o")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()



plt.plot(x, y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()                # 画出坐标轴
plt.axvline()
plt.show()




plt.plot(x, y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()
plt.annotate('x',(1.333, 0)) # 标出截距点
plt.annotate('y',(0,-2))
plt.show()



plt.plot(x, y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()
m = 1.5
xInt = 4 / 3
yInt = -2
mx = [0, xInt]
my = [yInt, yInt + m * xInt]
plt.plot(mx, my, color='red', lw=5)  # 用红色标出
plt.show()


#In [60]

plt.grid()                             # 放大图
plt.axhline()
plt.axvline()
m = 1.5
xInt = 4 / 3
yInt = -2
mx = [0, xInt]
my = [yInt, yInt + m * xInt]
plt.plot(mx, my, color='red', lw=5) 
plt.show()


#In [61]
m = 1.5
yInt = -2

x = np.array(range(-10, 11))
y2 = m * x + yInt                       # 斜率截距形式

plt.plot(x, y2, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()

plt.annotate('y', (0, yInt))
plt.show()




#In [62]:
l1p1 = [16, 0]  # 线1点1
l1p2 = [0, 16]  # 线1点2


l2p1 = [25,0]   # 线2点1
l2p2 = [0,10]   # 线2点2


plt.plot(l1p1,l1p2, color='blue')
plt.plot(l2p1, l2p2, color="orange")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()

plt.show()