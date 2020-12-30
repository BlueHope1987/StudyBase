#利用RNN获得一段文本的但向量表示
#P55

import torch
import torch.nn as nn
class BiRNN(nn.Module):
    # word_dim为词向量长度，hidden_size为RNN隐状态维度
    def __init__(self,word_dim,hidden_size):
        super(RNN,self).__init__()
        #双向GRU，输入的张量第一维是batch大小
        self.gru=nn.GRU(word_dim,hidden_size=hidden_size,bidirectional=True,batch_first=True)
    # 输入x为barch组文本，长度为seq_len，词向量长度为word_dim，维度为batch×seq_len×word_dim
    # 输出为文本向量，维度为batch × (2 × hidden_size)
    def forward(self,x):
        batch=x.shape[0]
        # output为每个单词对应的最后一层RNN的隐状态，维度为batch×seq_len×(2×hidden_size)
        # last_hidden为最终的RNN状态，维度为2 × batch × hidden_size
        output, last_hidden=self.gru(x)
        return last_hidden.transpose(0,1).contiguous().view(batch,-1)
