#P81 自注意力计算示例
'''
交互层
    机器阅读理解的编码层获得问题和文章中单词的语义向量表示，但两部分编码基本独立。为获取答案模型需要交互处理文章和问题中的信息。即在交互层进行语义信息融合。
互注意力
    可用注意力机制计算从文章到问题的注意力向量，基于对文章第i个词pi的理解，对问题单词向量组(q1,q2,...qn)的语意总结，得到一个向量piq。
自注意力
    计算上下文编码时，循环神经网络RNN以线性方式传递单词信息，一个单词的信息随距离增加而衰减。当文章较长时，靠前或靠后部分语句几乎没有进行有效状态传递，但有时需要理解相隔较远的部分。
    可使用 自注意力机制 自注意力计算一个向量组和自身的注意力向量。
    该机制中，文本中所有单词对(pi,pj)，无论位置远近，均直接计算注意力函数值。使信息可以在相隔任意距离的单词间交互，大大提高信息的传递效率。
    每单词计算过程独立，可并行计算；完全舍弃位置信息，可与RNN同时使用。
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