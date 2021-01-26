#P77 字符CNN 字符卷积神经网络Char-CNN
#应对字符拼写错误和子词需要
'''
机器阅读理解模型架构：编码层（文章、问题） 交互层（文章×问题） 输出层（回答）
编码层：分词 转化词向量分布式表示 字符编码 上下文编码
  分词：词表、非词表词(OOV)<UNK>
    词表：大小|V| 每个词d维向量表示 获取词向量两种方式
    1、保持词表向量不变，采用预训练词表中的向量（如Word2vec的300维向量），训练过程中不进行改变；
        模型参数少 初期收敛快
    2、词表中向量视为参数，训练中和其他参数一起求导并优化。预训练词表向量初始化或随机初始化。
        可根据实际数据调整词向量值以达更好训练效果 预训练词表比随机初始化于最初几轮结果更优
    命名实体(names entity)表(向量化) N种 表大小N 每种长度d(N)
    词性(part-of-speech)表(向量化) P种 表大小P 每种长度d(P)
    spaCy 文本分析软件包获得命名实体和词性
    精确匹配编码(exact matching) 适用于文章中(标识 出现 存在)的单词
  字符编码 每个字符用一个向量表示
    子词 如 inter- 中间 相互 -ward 方向
    字符CNN K个字符 向量(c1,c2,...,cK) 每向量维度为c CNN利用窗口大小W有f个输出通道的卷及神经网络获得(K-W+1)个f维向量 最大池化求每维度最大值 形成f维向量为结果
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class Char_CNN_Maxpool(nn.Module):
    #char_num为字符表大小，char_dim为字符向量长度，window_size为CNN窗口长度，output_dim为CNN输出通道数
    def __init__(self,char_num,char_dim,window_size,out_channels):
        super(Char_CNN_Maxpool,self).__init__()
        #字符表的向量，共char_num个向量，每个维度为char_dim
        self.char_embed=nn.Embedding(char_num,char_dim)
        #1个输入通道，out_channels个输出通道，过滤器大小为window_size×char_dim
        self.cnn=nn.Conv2d(1,out_channels,(window_size,char_dim))

    #输入char_ids为batch组文本，每个文本长度为seq_len，每个词含word_len个字符编号（0-char_num-1），输入维度为batch×seq_len×word_len
    #输出res为所有单词的字符向量表示，维度是batch×seq_len×out_channels
    def forward(self,char_ids):
        #根据字符编号得到字符向量，结果维度为batch×seq_len×word_len×char_dim
        x=self.char_embed(char_ids)
        #合并前两维并变成单通道，结果维度为(batch×seq_len)×1×word_len×char_dim
        x_unsqueeze=x.view(-1,x.shape[2],x.shape[3]).unsqueeze(1)
        #CNN，结果维度为(batch×seq_len)×out_channels×new_seq_len×1
        x_cnn=self.cnn(x_unsqueeze)
        #删除最后一维，结果维度为(batch×seq_len)×out_channels×new_seq_len
        x_cnn_result=x_cnn.squeeze(3)
        #最大池化，遍历最后一维求最大值，结果维度为(batch×seq_len)×out_channels
        res,_=x_cnn_result.max(2)
        return res.view(x.shape[0],x.shape[1],-1)

# 测试代码 摘自GitHub
batch = 10
seq_len = 20
word_len = 12
char_num = 26
char_dim = 10
window_size = 3
out_channels = 8
char_cnn = Char_CNN_Maxpool(char_num, char_dim, window_size, out_channels)
char_ids = torch.LongTensor(batch, seq_len, word_len).random_(0, char_num - 1)
res = char_cnn(char_ids)
print(res.shape) # torch.Size([10, 20, 8])