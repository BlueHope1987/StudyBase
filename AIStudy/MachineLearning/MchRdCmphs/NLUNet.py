#P60 自然语言理解 实战：文本分类
from Cnn_Maxpool import CNN_Maxpool
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NLUNet(nn.Module):
    #word_dim为词向量长度，window_size为CNN窗口长度，out_channels为CNN输出通道数，K为类别个数
    def __init__(self,word_dim,window_size,out_channels,K):
        super(NLUNet,self).__init__()
        #CNN和最大池化
        self.cnn_maxpool=CNN_Maxpool(word_dim,window_size,out_channels)
        #输出层为全连接层
        self.linear=nn.Linear(out_channels, K)

    #x:输入tensor,维度为 batch × seq_len × word_dim
    #输出class_score，维度是 batch × K
    def forward(self, x):
        #文本向量，结果维度是 batch × out_channels
        doc_embed=self.cnn_maxpool(x)
        #文本分数，结果维度是 batch × K
        class_score=self.linear(doc_embed)
        return class_score

K=3      #三分类
net=NLUNet(10,3,15,K)
#共30个序列，每个序列长度为5，词向量维度是10
x=torch.randn(30,5,10,requires_grad=True)
#30个真值分类，类别为0-K-1的整数
y=torch.LongTensor(30).random_(0,K)
optimizer=optim.SGD(net.parameters(),lr=0.01)
#res大小为 batch × K
res=net(x)
#Pytorch自带交叉熵函数，包含计算softmax
loss_func=nn.CrossEntropyLoss()
loss=loss_func(res,y)
print('loss1 =', loss) # 测试代码 摘自GitHu
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 测试代码 摘自GitHub
res = net(x)
loss = loss_func(res, y)
print('loss2 =', loss) # loss2应该比loss1小