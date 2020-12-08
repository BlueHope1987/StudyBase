#!/usr/bin/python3

#https://www.runoob.com/python3/python3-basic-syntax.html Python3 基础语法
# 第一个注释
# 第二个注释

'''
第三注释
第四注释
'''
 
"""
第五注释
第六注释
"""
print ("Hello, Python!")

#缩进
if True:
    total = 'aaa' + \
            'bbb' + \
            'ccc' #多行斜杠
    total2 = ['item_one', 'item_two', 'item_three',
            'item_four', 'item_five'] #括号内多行不需要斜杠
    print ("True")
else:
    print ("False")
#字符串与转义
str='Runoob'
print(str)                 # 输出字符串
print(str[0:-1])           # 输出第一个到倒数第二个的所有字符
print(str[0])              # 输出字符串第一个字符
print(str[2:5])            # 输出从第三个开始到第五个的字符
print(str[2:])             # 输出从第三个开始后的所有字符
print(str[1:5:2])          # 输出从第二个开始到第五个且每隔两个的字符
print(str * 2)             # 输出字符串两次
print(str + '你好')         # 连接字符串
 
print('------------------------------')
 
print('hello\nrunoob')      # 使用反斜杠(\)+n转义特殊字符
print(r'hello\nrunoob')     # 在字符串前面添加一个 r，表示原始字符串，不会发生转义



#https://www.runoob.com/python3/python3-class.html  Python3 面向对象

#!/usr/bin/python3
 
class MyClass:
    """一个简单的类实例"""
    i = 12345
    def f(self):
        return 'hello world'
 
# 实例化类
x = MyClass()
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())

class Complex:
    def __init__(self, realpart, imagpart): #构造方法 def定义方法 类方法必须包含参数 self, 且为第一个参数 两个下划线开头，声明该成员为私有
        self.r = realpart #self代表类的实例，而非类
        self.i = imagpart
x = Complex(3.0, -4.5)
print(x.r, x.i)   # 输出结果：3.0 -4.5

#类定义
class people:
    #定义基本属性
    name = ''
    age = 0
    #定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0
    #定义构造方法
    def __init__(self,n,a,w):
        self.name = n
        self.age = a
        self.__weight = w
    def speak(self):
        print("%s 说: 我 %d 岁。" %(self.name,self.age))
 
#单继承示例
class student(people):
    grade = ''
    def __init__(self,n,a,w,g):
        #调用父类的构函
        people.__init__(self,n,a,w)
        self.grade = g
    #覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
 
#另一个类，多重继承之前的准备
class speaker():
    topic = ''
    name = ''
    def __init__(self,n,t):
        self.name = n
        self.topic = t
    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
 
#多重继承
class sample(speaker,student):
    a =''
    def __init__(self,n,a,w,g,t):
        student.__init__(self,n,a,w,g)
        speaker.__init__(self,n,t)
 
test = sample("Tim",25,80,4,"Python")
test.speak()   #方法名同，默认调用的是在括号中排前地父类的方法

class Parent:        # 定义父类
   def myMethod(self):
      print ('调用父类方法')
 
class Child(Parent): # 定义子类
   def myMethod(self):
      print ('调用子类方法')
 
c = Child()          # 子类实例
c.myMethod()         # 子类调用重写方法
super(Child,c).myMethod() #用子类对象调用父类已被覆盖的方法








#https://www.runoob.com/python3/python3-namespace-scope.html 命名空间和作用域

total = 0 # 这是一个全局变量
# 可写函数说明
def sum( arg1, arg2 ):
    #返回2个参数的和."
    total = arg1 + arg2 # total在这里是局部变量.
    print ("函数内是局部变量 : ", total)
    return total
 
#调用sum函数
sum( 10, 20 )
print ("函数外是全局变量 : ", total)

num = 1
def fun1():
    global num  # 需要使用 global 关键字声明
    print(num) 
    num = 123
    print(num)
fun1()
print(num)

def outer():
    num = 10
    def inner():
        nonlocal num   # nonlocal关键字声明
        num = 100
        print(num)
    inner()
    print(num)
outer()

#报错的
a = 10
def test():
#    a = a + 1
    print(a)
test()

a = 10
def test():
    global a
    a = a + 1
    print(a)
test()

a = 10
def test(a):
    a = a + 1
    print(a)
test(a)


#九九乘法表 https://www.runoob.com/python3/python3-99-table.html
input("回车继续...") #输入输出: https://www.runoob.com/python3/python3-inputoutput.html

for i in range(1,10): #内置range(start, stop[, step]) 函数可创建一个整数列表，一般用在 for 循环中。
    for j in range(1,i+1):
        print('{}×{}={}\t'.format(j,i,i*j),end='')
    print()

#Python 十进制转二进制、八进制、十六进制 https://www.runoob.com/python3/python3-conversion-binary-octal-hexadecimal.html

dec = int(input("输入数字："))
print("十进制数为：", dec)
print("转换为二进制为：", bin(dec))
print("转换为八进制为：", oct(dec))
print("转换为十六进制为：", hex(dec))




#正则表达式 https://www.runoob.com/python3/python3-reg-expressions.html
import re
print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配
line = "Cats are smarter than dogs"
# .* 表示任意匹配除换行符（\n、\r）之外的任何单个或多个字符
#正则表达式中，group（）用来提出分组截获的字符串，（）用来分组
#正则表达式中的三组括号把匹配结果分成三组group() 同group（0）就是匹配正则表达式整体结果 group(1) 列出第一个括号匹配部分，group(2) 列出第二个括号匹配部分，group(3) 列出第三个括号匹配部分。
#re.match(pattern, string, flags=0) 尝试从字符串的起始位置匹配一个模式
#flags re.I大小写不敏感 L本地化识别 M多行匹配 S使.匹配包括换行在内的所有字符 U根据Unicode字符集解析字符 X更灵活的格式
matchObj = re.match( r'(.*) are (.*?) .*', line, re.M|re.I)
if matchObj:
   print ("matchObj.group() : ", matchObj.group())
   print ("matchObj.group(1) : ", matchObj.group(1))
   print ("matchObj.group(2) : ", matchObj.group(2))
else:
   print ("No match!!")

#re.search(pattern, string, flags=0)扫描整个字符串并返回第一个成功的匹配
searchObj = re.search( r'(.*) are (.*?) .*', line, re.M|re.I)
if searchObj:
   print ("searchObj.group() : ", searchObj.group())
   print ("searchObj.group(1) : ", searchObj.group(1))
   print ("searchObj.group(2) : ", searchObj.group(2))
else:
   print ("Nothing found!!")

#re.match与re.search的区别
matchObj = re.match( r'dogs', line, re.M|re.I)
if matchObj:
   print ("match --> matchObj.group() : ", matchObj.group())
else:
   print ("No match!!")
matchObj = re.search( r'dogs', line, re.M|re.I)
if matchObj:
   print ("search --> matchObj.group() : ", matchObj.group())
else:
   print ("No match!!")

#检索替换
phone = "2004-959-559 # 这是一个电话号码"
num = re.sub(r'#.*$', "", phone) # 删除注释 字符串前面加上r表示原生字符串（rawstring）
print ("电话号码 : ", num)
num = re.sub(r'\D', "", phone)# 移除非数字的内容
print ("电话号码 : ", num)