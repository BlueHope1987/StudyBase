#含参加权和 P58
#含参加权和(parametrized weighted sum)是一种将多个词向量变为一个文本向量的常用方法。与"平均池化"不同，权重根据向量之间的关系确定并优化。是一种自注意力机制。
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSum(nn.Module):
    #输入的词向量维度为word_dim
    def __init__(self,word_dim):
        super(WeightedSum,self).__init__()
        self.b=nn.Linear(word_dim,1) #参数张量 全连接层nn.Linear实现向量内积
    
    #x: 输入tensor，维度为 batch × seq_len × word_dim
    #输出res，维度是batch × word_dim
    def forward(self,x):
        #内积得分，维度是 batch × seq_len × 1
        scores=self.b(x)
        #softmax运算，结果维度是batch × seq_len × 1
        #softmax函数: 可以把一组输入的数字变成概率值并维持原有的大小关系，使之的和等1
        weights=F.softmax(scores,dim=1)
        #用矩阵乘法实现加权和，结果维度是 batch × word_dim × 1
        res=torch.bmm(x.transpose(1,2),weights) #可实现两个批次矩阵的相乘 batch×a×b 与 batch×b×c
        #删除最后一维，结果维度是 batch × word_dim
        res=res.squeeze(2)
        return res

#测试代码 摘自GitHub
batch = 10
seq_len = 20
word_dim = 50
x = torch.randn(batch, seq_len, word_dim)
weighted_sum = WeightedSum(word_dim)
res = weighted_sum(x)
print(res.shape) # torch.Size([10, 50])