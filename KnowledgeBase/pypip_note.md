PIP 更换国内安装源

pip国内的一些镜像

  阿里云 https://mirrors.aliyun.com/pypi/simple/ 
  中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/ 
  豆瓣(douban) http://pypi.douban.com/simple/ 
  清华大学 https://pypi.tuna.tsinghua.edu.cn/simple/ 
  中国科学技术大学 http://pypi.mirrors.ustc.edu.cn/simple/

修改源方法：

临时使用： 
可以在使用pip的时候在后面加上-i参数，指定pip源 
eg: pip install scrapy -i https://pypi.tuna.tsinghua.edu.cn/simple

本地命令
pip install XXXX -i https://mirrors.aliyun.com/pypi/simple/

从文件清单安装
pip install -r train_requirements.txt

永久修改： 
linux: 
修改 ~/.pip/pip.conf (没有就创建一个)， 内容如下：

[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
1
2
windows: 
直接在user目录中创建一个pip目录，如：C:\Users\xx\pip，在pip 目录下新建文件pip.ini，内容如下

或者按照网友的建议：win+R 打开用户目录%HOMEPATH%，在此目录下创建 pip 文件夹，在 pip 目录下创建 pip.ini 文件, 内容如下

[global]

timeout = 6000

index-url = https://pypi.tuna.tsinghua.edu.cn/simple

trusted-host = pypi.tuna.tsinghua.edu.cn

批量卸载
pip uninstall <包名> <包名> <包名> <包名> -y
-y 不再逐一确认


淘宝镜像
https://npm.taobao.org/mirrors/python/