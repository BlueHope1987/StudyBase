https://developer.aliyun.com/article/1366693
https://developer.aliyun.com/article/1366697
https://developer.aliyun.com/article/1366699

## 1. Kubernetes 介绍

### 1.1 K8S 部署方式演变

随着互联网的发展，在应用程序部署方式上主要经历了三个时代：

#### 1.传统部署：互联网早期，会直接将应用程序部署在物理机上
- 优点：简单，不需要其它技术的参与

- 缺点：不能为应用程序定义资源使用边界，很难合理地分配计算资源，而且程序之间容易产生影响

#### 2.虚拟化部署：可以在一台物理机上运行多个虚拟机，每个虚拟机都是独立的一个环境

- 优点：程序环境不会相互产生影响，提供了一定程度的安全性

- 缺点：增加了操作系统，浪费了部分资源

#### 3.容器化部署：与虚拟化类似，但是共享了操作系统

- 优点：  
    - 可以保证每个容器拥有自己的文件系统、CPU、内存、进程空间等  
    - 运行应用程序所需要的资源都被容器包装，并和底层基础架构解耦  
    - 容器化的应用程序可以跨云服务商、跨Linux操作系统发行版进行部署  

容器化部署方式给带来很多的便利，但是也会出现一些问题，比如说：  
- 一个容器故障停机了，怎么样让另外一个容器立刻启动去替补停机的容器  
- 当并发访问量变大的时候，怎么样做到横向扩展容器数量  

这些容器管理的问题统称为容器编排问题，为了解决这些容器编排问题，就产生了一些容器编排的软件：
- Swarm：Docker自己的容器编排工具
- Mesos：Apache的一个资源统一管控的工具，需要和Marathon结合使用
- Kubernetes：Google开源的的容器编排工具

#### 1.2 kubernetes简介

kubernetes 的 logo, 标志是一个舵手，因为英文单词 K 和 S 中间包含 8 个字母，所以简称 K8S， 是一个全新的基于容器技术的分布式架构领先方案，是谷歌严格保密十几年的秘密武器----Borg 系统的一个开源版本，于 2014 年 9 月发布第一个版本，2015年 7 月发布第一个正式版本。

kubernetes 的本质是一组服务器集群，它可以在集群的每个节点上运行特定的程序，来对节点中的容器进行管理。目的是实现资源管理的自动化，主要提供了如下的主要功能：

- 自我修复：一旦某一个容器崩溃，能够在1秒中左右迅速启动新的容器
- 弹性伸缩：可以根据需要，自动对集群中正在运行的容器数量进行调整
- 服务发现：服务可以通过自动发现的形式找到它所依赖的服务
- 负载均衡：如果一个服务起动了多个容器，能够自动实现请求的负载均衡
- 版本回退：如果发现新发布的程序版本有问题，可以立即回退到原来的版本
- 存储编排：可以根据容器自身的需求自动创建存储卷

#### 1.3 kubernetes组件

一个 kubernetes 集群主要是由控制节点 master、工作节点 node构成，每个节点上都会安装不同的组件。
- master：集群的控制平面，负责集群的决策 ( 管理 )  
Master 节点上会安装四个重要组件，分别如下：  
    - ApiServer : 资源操作的唯一入口，接收用户输入的命令，提供认证、授权、API注册和发现等机制
    - Scheduler : 负责集群资源调度，按照预定的调度策略将 Pod 调度到相应的 node 节点上
    - ControllerManager : 负责维护集群的状态，比如程序部署安排、故障检测、自动扩展、滚动更新等
    - Etcd ：负责存储集群中各种资源对象的信息，相当于 K8S 的数据库
- node：集群的数据平面，负责为容器提供运行环境 ( 干活 )  
node 节点上会安装三个重要组件，分别如下：
    - Kubelet : 负责维护容器的生命周期，即通过控制docker，来创建、更新、销毁容器
    - KubeProxy : 负责提供集群内部的服务发现和负载均衡
    - Docker : 负责节点上容器的各种操作
    

下面，以部署一个 nginx 服务来说明 kubernetes 系统各个组件调用关系：

1. 首先要明确，一旦 kubernetes 环境启动之后，master 和 node 都会将自身的信息存储到 etcd 数据库中；

2. 一个 nginx 服务的安装请求会首先被发送到 master 节点的 apiServer 组件；

3. apiServer 组件会调用 scheduler 组件来决定到底应该把这个服务安装到哪个 node 节点上，在此时，它会从 etcd 中读取各个 node 节点的信息，然后按照一定的算法进行选择，并将结果告知 apiServer；

4. apiServer 调用 controller-manager 去调度 Node 节点安装 nginx 服务；

5. kubelet 接收到指令后，会通知 docker，然后由 docker 来启动一个 nginx 的 pod，pod 是 kubernetes 的最小操作单元，容器必须跑在 pod 中；

6. 当Pod启动后，一个 nginx 服务就运行了，如果需要访问 nginx，就需要通过 kube-proxy 来对 pod 产生访问的代理。这样，外界用户就可以访问集群中的 nginx 服务了

### 1.4 kubernetes 概念

- Master：集群控制节点，每个集群需要至少一个 master 节点负责集群的管控

- Node：工作负载节点，由 master 分配容器到这些 node 工作节点上，然后 node 节点上的 docker 负责容器的运行

- Pod：kubernetes 的最小控制单元，容器都是运行在 pod 中的，一个 pod 中可以有 1 个或者多个容器

- Controller：控制器，通过它来实现对 pod 的管理，比如启动 pod、停止 pod、伸缩 pod 的数量等等

- Service：pod 对外服务的统一入口，下面可以维护着同一类的多个 pod

- Label：标签，用于对 pod 进行分类，同一类 pod 会拥有相同的标签

- NameSpace：命名空间，用来隔离 pod 的运行环境


## 2. kubernetes集群环境搭建
### 2.1 部署方式  
Kubernetes 有多种部署方式，目前主流的方式有：minikube、kubeadm、二进制包等。  
在生产环境 Kubernetes 集群主要有两种方式：
#### 1. kubeadm  
Kubeadm 是一个 K8s 部署工具，提供 kubeadm init 和 kubeadm join，用于快速部署 Kubernetes 集群。  
官方地址：  
https://kubernetes.io/docs/reference/setup-tools/kubeadm/kubeadm/  
#### 2. 二进制包   
从 github 下载发行版的二进制包，手动部署每个组件，组成 Kubernetes 集群。  
Kubeadm 降低部署门槛，但屏蔽了很多细节，遇到问题很难排查。如果想更容易可控，推荐使用二进制包部署 Kubernetes 集群，虽然手动部署麻烦点，期间可以学习很多工作原理，也利于后期维护。

### 2.2 kubeadm 部署方式介绍

kubeadm 是官方社区推出的一个用于快速部署 kubernetes 集群的工具，这个工具能通过两条指令完成一个 kubernetes 集群的部署：
- 创建一个 Master 节点 kubeadm init
- 将 Node 节点加入到当前集群中$ kubeadm join < Master 节点的IP 和端口

### 2.3 安装要求

在开始之前，部署 Kubernetes 集群机器需要满足以下几个条件：

- 一台或多台机器，操作系统 CentOS7.x-86_x64  
- 硬件配置：2GB 或更多 RAM，2 个 CPU 或更多 CPU，硬盘 30GB 或更多  
- 集群中所有机器之间网络互通  
- 可以访问外网，需要拉取镜像  
- 禁止 swap 分区  

### 2.4 最终目标

- 在所有节点上安装 Docker 和 kubeadm  
- 部署 Kubernetes Master  
- 部署容器网络插件  
- 部署 Kubernetes Node，将节点加入 Kubernetes 集群中  
- 部署 Dashboard Web 页面，可视化查看 Kubernetes 资源  

### 2.5 准备环境

本文采用三节点进行部署，mini1 代表 Master 节点，mini2、mini3 代表 node 节点。
|角色 |IP地址 |组件 |
|----|----|----|
|mini1|192.168.244.131|docker, kubectl, kubeadm, kubelet|
|mini2|192.168.244.132|docker, kubectl, kubeadm, kubelet|
|mini3|192.168.244.133|docker, kubectl, kubeadm, kubelet|

### 2.6 环境初始化

下面 2.6.1 - 2.6.11 章节的执行步骤在三台机器上全部执行,截图只展示 mini1的内容！！！

#### 2.6.1 检查操作系统的版本
```
# 此方式下安装 kubernetes 集群要求 Centos 版本要在 7.5 或之上
[root@mini1 ~]# cat /etc/redhat-release
CentOS Linux release 7.6.1810 (Core)
```
#### 2.6.2 主机名解析

在 hosts 文件配置 IP 地址映射
```
# 主机名成解析 编辑三台服务器的/etc/hosts文件，添加下面内容
192.168.244.131 mini1
192.168.244.132 mini2
192.168.244.133 mini3
```

##### 2.6.3 时间同步
kubernetes 要求集群中的节点时间必须精确一直，这里使用 chronyd 服务从网络同步时间  
企业中建议配置内部的会见同步服务器  
chronyd 安装教程如下：  
```
# 启动chronyd服务
[root@mini1 ~]# yum install -y chrony
[root@mini1 ~]# systemctl start chronyd
[root@mini1 ~]# systemctl enable chronyd
[root@mini1 ~]# date
```
未完待续 https://developer.aliyun.com/article/1366697