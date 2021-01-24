#CNN最大池化  P57
#为得到一个文本向量，将每段文本的L个向量取平均值，即新向量的第j维是这L各向量第j维的平均值。即平均池化
#或最大池化：对着L个向量的每一维求最大值，即新向量的第j维是这L个向量的第j维的最大值。优势：重要单词位置无关，即平移不变性。
import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN_Maxpool(nn.Module):
    #word_dim为词向量长度，window_size为CNN窗口长度，output_dim为CNN输出通道数
    def __init__(self,word_dim,window_size,out_channels):
        super(CNN_Maxpool,self).__init__()
        #1个输入通道，out_channels个输出通道，过滤器大小为window_size×word_dim
        self.cnn=nn.Conv2d(1,out_channels,(window_size,word_dim))

    #输入x为batch组文本，长度为seq_len，词向量长度为word_dim，维度为batch×seq_len×word_dim
    #输出res为所有文本向量，每个向量的维度为out_channels
    def forward(self,x):
        #变成单通道，结果维度为batch×1×seq_len×word_dim
        #tensor.unsqueeze()在指定位置加入一个大小为1的维度表示单通道，tensor.squeeze()删除大小为1的通道
        x_unsqueeze=x.unsqueeze(1)
        #CNN，结果维度为barch×out_channels×new_seq_len×1
        x_cnn=self.cnn(x_unsqueeze)
        #删除最后一维，结果维度为batch×out_channels×new_seq_len
        x_cnn_result=x_cnn.squeeze(3)
        #最大池化，遍历最后一维求最大值，结果维度为batch×out_channels
        res,_=x_cnn_result.max(2)
        return res

#测试代码 摘自GitHub
batch = 10
seq_len = 20
word_dim = 50
window_size = 3
out_channels = 100
x = torch.randn(batch, seq_len, word_dim)
cnn_maxpool = CNN_Maxpool(word_dim, window_size, out_channels)
res = cnn_maxpool(x)
print(res.shape) # torch.Size([10, 100])

'''
字符CNN(Character CNN)
子词：CNN过滤器移动过程中，窗口包含的连续字符
CNN和最大池化的另一用途，可有效应对拼写错误，大部分子词正确不影响输出
'''