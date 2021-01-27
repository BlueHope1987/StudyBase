#P78 多层双向RNN生成文本中每个单词的上下文编码
'''
理解一个单词需要考虑它的上下文
上下文编码(contextual embedding) 随单词上下文不同而发生改变 反应出单词在当前语句中含义
为解决一个词在不同的语句语义不同但其向量表示完全相同的情况
循环神经网络RNN是最常用的上下文编码生成结构 为更有效
'''
import torch
import torch.nn as nn

class Contextual_Embedding(nn.Module):
    #word_dim为词向量维度，state_dim为RNN状态维度，rnn_layer为RNN层数
    def __init__(self,word_dim,state_dim,rnn_layer):
        super(Contextual_Embedding,self).__init__()
        #多层双向GRU，输入维度为word_dim，状态维度为state_dim
        self.rnn=nn.GRU(word_dim,state_dim,num_layers=rnn_layer,bidirectional=True,batch_first=True)

    #输入x为batch组文本，每个文本长度为seq_len，每个词用一个word_dim维向量表示，输入维度为batch×seq_len×word_dim
    #输出res为所有单词的上下文向量表示，维度是batch×seq_len×out_dim
    def forward(self,x):
        #结果维度为batch×seq_len×out_dim，其中out_dim=2×state_dim包括两个方向
        res,_=self.rnn(x)
        return res

# 测试代码 摘自GitHub
batch = 10
seq_len = 20
word_dim = 50
state_dim = 100
rnn_layer = 2
x = torch.randn(batch, seq_len, word_dim)
context_embed = Contextual_Embedding(word_dim, state_dim, rnn_layer)
res = context_embed(x)
print(res.shape) # torch.Size([10, 20, 200])