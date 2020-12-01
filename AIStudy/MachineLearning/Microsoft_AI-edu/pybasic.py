
#https://github.com/microsoft/ai-edu/blob/master/A-%E6%95%99%E5%AD%A6%E8%AF%BE%E7%A8%8B/math_intro/01_%E4%BB%A3%E6%95%B0.ipynb


#In [54]:
import numpy as np
from tabulate import tabulate

x = np.array(range(-10, 11))  # 从 -10 到 10 的21个数据点
y = (3 * x - 4) / 2           # 对应的函数值

print(tabulate(np.column_stack((x,y)), headers=['x', 'y']))



#In [55]:
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
# %matplotlib inline

#In [56]:
from matplotlib import pyplot as plt

plt.plot(x, y, color="grey", marker = "o")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()


#In [57]:
plt.plot(x, y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()                # 画出坐标轴
plt.axvline()
plt.show()



#In [58]:
plt.plot(x, y, color="grey")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()
plt.annotate('x',(1.333, 0)) # 标出截距点
plt.annotate('y',(0,-2))
plt.show()


#In [59]:
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



#In [63]:
x = 10
y = 6
print ((x + y == 16) & ((10 * x) + (25 * y) == 250))


#In [64]:
x = 5**3
print(x)


#In [65]:
import math

x = math.sqrt(9)            # 平方根
print (x)

cr = round(64 ** (1. / 3))   # 立方根
print(cr)


#In [66]:
print (9**0.5)
print (math.sqrt(9))


#In [67]:
x = math.log(16, 4)
print(x)


#In [68]:
print(math.log10(64))

print (math.log(64))


#In [69]:
x = np.array(range(-10, 11))
y3 = 3 * x ** 3

print(tabulate(np.column_stack((x, y3)), headers=['x', 'y']))


#In [70]:
plt.plot(x, y3, color="magenta")  # y3是曲线
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()
plt.show()



#In [71]:
y4 = 2.0**x
plt.plot(x, y4, color="magenta")
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axhline()
plt.axvline()
plt.show()



#In [72]:
year = np.array(range(1, 21))                      # 年份
balance = 100 * (1.05 ** year)                     # 余额
plt.plot(year, balance, color="green")
plt.xlabel('Year')
plt.ylabel('Balance')
plt.show()


