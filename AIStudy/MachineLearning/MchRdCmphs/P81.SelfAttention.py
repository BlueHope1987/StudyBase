#P81 自注意力计算示例
'''
计算上下文编码时，循环神经网络RNN以线性方式传递单词信息，一个单词的信息随距离增加而衰减。当文章较长时，靠前或靠后部分语句几乎没有进行有效状态传递，但有时需要理解相隔较远的部分。
可使用 自注意力机制 自注意力计算一个向量组和自身的注意力向量。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    #dim为向量维度，hidden_dim为自注意力计算的隐藏层维度
    def __init__(self,dim,hidden_dim):
        super(SelfAttention,self).__init__()
        #参数矩阵W
        self.W=nn.Linear(dim,hidden_dim)

    #x:进行自注意力计算的向量组，batch×n×dim
    def forward(self,x):  #书中缩进有误
        #计算隐藏层，结果维度为batch×n×hidden_dim
        hidden=self.W(x)
        #注意力分数Scores,维度为batch×n×n
        scores=hidden.bmm(hidden.transpose(1,2))
        #对最后一维进行softmax
        alpha=F.softmax(scores,dim=-1)
        #注意力向量，结果维度为batch×n×dim
        attended=alpha.bmm(x)
        return attended

# 测试代码 摘自GitHub
batch = 10
n = 15
dim = 40
hidden_dim = 20
x = torch.randn(batch, n, dim)
self_attention = SelfAttention(dim, hidden_dim)
res = self_attention(x)
print(res.shape) # torch.Size([10, 15, 40])