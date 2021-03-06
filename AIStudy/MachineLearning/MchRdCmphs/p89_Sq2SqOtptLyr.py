#P89 训练自由式答案生成网络的输出层
'''
自由式答案生成
可以为任何自然语言形式，不需要其中所有单词均来自文章 即自然语言生成的过程 任务输出层常用序列到序列模型 即编码器-解码器模型
编码器 从交互层获得每个单词向量，双向循环神经网络处理文本所有单词
解码器 使用单向循环神经网络依次产生答案的单词 一般开头<s> 词表外词<UNK> 词表每个单词打分softmax得预测概率状态传递 Teacher Forcing(P63)与否使用标准答案或高分词 以此类推
注意力机制 在解码器生成答案第i个位置的单词时 计算输入状态和编码器中所有状态的注意力向量c(i-1) 是编码器中文章单词向量的线性组合 遂称c(i-1)为上下文向量 输入解码器RNN 随其状态每一步变化而变化
'''

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

'''
  a: 被注意的的向量组，batch x m x dim 
  x: 进行注意力计算的向量组，batch x n x dim
'''
from p69_attention import attention #通过导入代替下例函数 （py文件名带点是强烈不推荐的 导入时代表路径）
'''
def attention(a, x):
    # 内积计算注意力分数，结果维度为batch x n x m
    scores = x.bmm(a.transpose(1, 2))
    # 对最后一维进行softmax
    alpha = F.softmax(scores, dim=-1)
    # 注意力向量，结果维度为batch x n x dim
    attended = alpha.bmm(a) 
    return attended
'''


class Seq2SeqOutputLayer(nn.Module):
    # word_dim为交互层输出的问题向量和文章单词向量的维度，以及词表维度
    # embed为编码层使用的词表，即nn.Embedding(vocab_size, word_dim)
    # vocab_size为词汇表大小
    def __init__(self, embed, word_dim, vocab_size):
        super(Seq2SeqOutputLayer, self).__init__()
        # 使用和编码层同样的词表向量
        self.embed = embed
        self.vocab_size = vocab_size
        # 编码器RNN，单层单向GRU，batch是第0维
        self.encoder_rnn = nn.GRU(word_dim, word_dim, batch_first=True)
        # 解码器RNN单元GRU
        self.decoder_rnncell = nn.GRUCell(word_dim, word_dim)
        # 将RNN状态和注意力向量的拼接结果降维成word_dim维
        self.combine_state_attn = nn.Linear(word_dim + word_dim, word_dim)
        # 解码器产生单词分数的全连接层，产生一个位置每个单词的得分
        self.linear = nn.Linear(word_dim, vocab_size, bias=False)
        # 全连接层和词表共享参数
        self.linear.weight = embed.weight
    
    # x: 交互层输出的文章单词向量，维度为batch x x_seq_len x word_dim
    # q: 交互层输出的问题向量，维度为batch x word_dim
    # y_id：真值输出文本的单词编号,维度为batch x y_seq_len
    # 输出预测的每个位置每个单词的得分word_scores，维度是batch x y_seq_len x vocab_size
    def forward(self, x, q, y_id):
        # 得到真值输出文本的词向量，维度为batch x y_seq_len x word_dim
        y = self.embed(y_id) 
        # 编码器RNN，以问题向量q作为初始状态
        # 得到文章每个位置的状态enc_states，结果维度是batch x x_seq_len x word_dim
        # 得到最后一个位置的状态enc_last_state，结果维度是1 x batch x word_dim
        enc_states, enc_last_state = self.encoder_rnn(x, q.unsqueeze(0))
        # 解码器的初始状态为编码器最后一个位置的状态，维度是batch x word_dim
        prev_dec_state = enc_last_state.squeeze(0)
        # 最终输出为每个答案的所有位置各种单词的得分
        scores = torch.zeros(y_id.shape[0], y_id.shape[1], self.vocab_size)
        for t in range(0, y_id.shape[1]):
            # 将前一个状态和真值文本第t个词的向量表示输入解码器RNN，得到新的状态，维度batch x word_dim
            new_state = self.decoder_rnncell(y[:,t,:].squeeze(1), prev_dec_state)     
            # 利用3.4节的attention函数获取注意力向量，结果维度为batch x word_dim
            context = attention(enc_states, new_state.unsqueeze(1)).squeeze(1)
            # 将RNN状态和注意力向量的拼接结果降维成word_dim维, 结果维度为batch x word_dim
            new_state = self.combine_state_attn(torch.cat((new_state, context), dim=1))
            # 生成这个位置每个词表中单词的预测得分
            scores[:, t, :] = self.linear(new_state)
            # 此状态传入下一个GRUCell
            prev_dec_state = new_state
        return scores

# 100个单词
vocab_size = 100
# 单词向量维度20
word_dim = 20
embed = nn.Embedding(vocab_size, word_dim)
# 共30个真值输出文本的词id，每个文本长度8
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
# 此处省略编码层和交互层，用随机化代替
# 交互层最终得到：
# 1) 文章单词向量x，维度30 x x_seq_len x word_dim
# 2) 问题向量q，维度30 x word_dim
x = torch.randn(30, 10, word_dim)
q = torch.randn(30, word_dim)
# 设定网络
net = Seq2SeqOutputLayer(embed, word_dim, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=0.1) 
# 获得每个位置上词表中每个单词的得分word_scores，维度为30 x y_seq_len x vocab_size
word_scores = net(x, q, y_id)
# PyTorch自带交叉熵函数，包含计算softmax
loss_func = nn.CrossEntropyLoss()
# 将word_scores变为二维数组，y_id变为一维数组，计算损失函数值
# word_scores计算出第2、3、4...个词的预测，因此需要和y_id错开一位比较
loss = loss_func(word_scores[:,:-1,:].contiguous().view(-1, vocab_size), y_id[:,1:].contiguous().view(-1))
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()


# 测试代码 摘自GitHub
word_scores = net(x, q, y_id)
loss = loss_func(word_scores[:,:-1,:].contiguous().view(-1, vocab_size), y_id[:,1:].contiguous().view(-1))
print('loss2 =', loss) # loss2应该比loss1小