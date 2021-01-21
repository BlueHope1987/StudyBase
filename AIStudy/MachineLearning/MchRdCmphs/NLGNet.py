#P63 自然语言理解 实战：生成文本
#这种生成文本的方式是将状态转化成单词的过程，被称为解码器。必须是从左到右单向RNN，而不能使用双向RNN，因为解码过程是依次产生下一个词。
from Cnn_Maxpool import CNN_Maxpool #导入本地最大池化类 或复用代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NLGNet(nn.Module):
    # word_dim为词向量长度，window_size为CNN窗口长度，rnn_dim为RNN的状态维度，vocab_size为词汇表大小
    def __init__(self, word_dim, window_size, rnn_dim, vocab_size):
        super(NLGNet, self).__init__()
        # 单词编号与词向量对应参数矩阵
        self.embed = nn.Embedding(vocab_size, word_dim)  
        # CNN和最大池化        
        self.cnn_maxpool = CNN_Maxpool(word_dim, window_size, rnn_dim)
        # 单层单向GRU，batch是第0维
        self.rnn = nn.GRU(word_dim, rnn_dim, batch_first=True) 
        # 输出层为全连接层，产生一个位置每个单词的得分
        self.linear = nn.Linear(rnn_dim, vocab_size)     
    
    # x_id：输入文本的词编号,维度为batch x x_seq_len
    # y_id：真值输出文本的词编号,维度为batch x y_seq_len
    # 输出预测的每个位置每个单词的得分word_scores，维度是batch x y_seq_len x vocab_size
    def forward(self, x_id, y_id):
        # 得到输入文本的词向量，维度为batch x x_seq_len x word_dim
        x = self.embed(x_id) 
        # 得到真值输出文本的词向量，维度为batch x y_seq_len x word_dim
        y = self.embed(y_id) 
        # 输入文本向量，结果维度是batch x cnn_channels
        doc_embed = self.cnn_maxpool(x)
        # 输入文本向量作为RNN的初始状态，结果维度是1 x batch x y_seq_len x rnn_dim
        h0 = doc_embed.unsqueeze(0)
        # RNN后得到每个位置的状态，结果维度是batch x y_seq_len x rnn_dim
        rnn_output, _ = self.rnn(y, h0)
        # 每一个位置所有单词的分数，结果维度是batch x y_seq_len x vocab_size
        word_scores = self.linear(rnn_output)   
        return word_scores

vocab_size = 100                        # 100个单词
net = NLGNet(10, 3, 15, vocab_size)     # 设定网络 
# 共30个输入文本的词id，每个文本长度10
x_id = torch.LongTensor(30, 10).random_(0, vocab_size) 
# 共30个真值输出文本的词id，每个文本长度8
y_id = torch.LongTensor(30, 8).random_(0, vocab_size)
optimizer = optim.SGD(net.parameters(), lr=1) 
# 每个位置词表中每个单词的得分word_scores，维度为30 x 8 x vocab_size
word_scores = net(x_id, y_id)
# PyTorch自带交叉熵函数，包含计算softmax
loss_func = nn.CrossEntropyLoss()
# 将word_scores变为二维数组，y_id变为一维数组，计算损失函数值
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss1 =', loss)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 自定义测试代码
print(x_id)
print(y_id)

# 测试代码 摘自GitHub
word_scores = net(x_id, y_id)
loss = loss_func(word_scores[:,:-1,:].reshape(-1, vocab_size), y_id[:, 1:].reshape(-1))
print('loss2 =', loss) # loss2应该比loss1小