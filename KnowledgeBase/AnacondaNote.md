Python 和 Anaconda

Anaconda 是一个用于科学计算的 Python 发行版，支持 Linux, Mac, Windows, 包含了众多流行的科学计算、数据分析的 Python 包。
Anaconda 安装包可以到 https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/ 下载。

Conda 是一个开源的软件包管理系统和环境管理系统，用于安装多个版本的软件包及其依赖关系，并在它们之间轻松切换。 Conda 是为 Python 程序创建的，适用于 Linux，OS X 和Windows，也可以打包和分发其他软件。
conda分为anaconda和miniconda。anaconda是包含一些常用包的版本（这里的常用不代表你常用 微笑.jpg），miniconda则是精简版，需要啥装啥。

推荐使用miniconda

常用conda 命令编辑
conda list
列出当前 conda 环境所链接的软件包 
conda create
列出环境
conda info -e
创建一个 conda 环境
conda create -n 环境名 -c 镜像源
conda create -n xxx python=3.8
删除环境
conda remove -n 环境 --all
退出：
conda deactivate

安装库：
conda install cudnn=7.6
conda install tensorflow
conda install torchvision -c pytorch （-c 从通道中获取）
pip3 install torchvision


[安装路径]\Scripts\activate 启动环境
conda activate XXX 激活环境

[Windows]配置默认镜像、新环境路径等 配置文件在X:\Users\<你的用户名>\.condarc
[参考]https://www.cnblogs.com/wqbin/p/11810415.html
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
ssl_verify: true
envs_dirs:
  - D:\xxx\xxx\envs  # 按顺序第一个路径作为默认存储路径，搜索环境和缓存时按先后顺序在各目录中查找
  - C:\Users\xxx\AppData\Local\conda\conda\envs
  - C:\Users\xxx\.conda\envs                        
pkgs_dirs:
  - D:\xxx\anaconda3\pkgs
  - C:\Users\xxx\AppData\Local\conda\conda\pkgs

[环境同步参考]
 [tf2] 
    conda create -n tf2 python=3.8.5 tensorflow=2.3.0 pytorch=1.6.0
    家里的电脑会出问题 所有的tf会话都跑飞 第二步loss值nan 准确度不上升 经反复安装测试 是三者同时进行安装造成的
     [tf2test]已弃用
    conda create -n tf2test python=3.8.0 tensorflow=2.3.0
    没问题 即便换3.8.5 后来再安装pytorch 1.6.0

 [tf1.15]
    conda create -n tf1.15 python=3.7 tensorflow=1.15
    (tf1不支持3.8)