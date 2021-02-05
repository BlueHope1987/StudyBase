#P69 注意力机制 使用内积函数计算注意力
#(attention mechanism)最早在机器视觉领域被提出的，根据当前需要将权重集中在某些词语，随时间推移调整
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
a: 被注意的向量组，batch × m × dim
x: 进行注意力计算的向量组，batch × n × dim
注意力取决于从向量x的角度给a打分
'''
def attention(a,x):
    #内积计算注意力分数，结果维度为batch×n×m
    scores=x.bmm(a.transpose(1,2))
    #对最后一维进行softmax
    alpha=F.softmax(scores,dim=-1)
    #注意力向量，结果维度为batch×n×dim
    attended=alpha.bmm(a)
    return attended #每个x[i,j]都得到了个对应dim维注意力向量attended[i,j]

# 测试代码 摘自GitHub
batch = 10
m = 20
n = 30
dim = 15
a = torch.randn(batch, m, dim)
x = torch.randn(batch, n, dim)
res = attention(a, x)
print(res.shape) # torch.Size([10, 30, 15])

'''
REF:注意力机制的序列到序列模型
(sequence-to-sequence(seq2seq) model with attention mechanism)
应对从单个文本向量很难提取出文本里的不同部分信息的情况
seq2seq模型使用编码器中的RNN处理输入词向量，并将最后一个RNN状态作为文本向量输入解码器，且RNN中输入的每个单词的状态均保留下来。在解码器的每步生成中，用前一步解码器状态计算编码器包含多个信息状态的注意力向量。因结果表示当前解码器观察到的输入信息的总结，因此也被称作“上下文向量”，输入解码器RNN便在每一步都拥有了含注意力分布的输入信息。
'''