#P87 答案区间生成
'''
输出层
    是机器阅读理解模型计算并输出答案的模块，根据任务要求的方式生成答案，并构造合理的损失函数便于模型在训练集优化。
多项选择式答案
    分类问题 对选项中每个单词进行编码，和问题与文章计算注意力向量 得到代表选项语意的向量 综合文章、问题和选项计算选项得分
    自然语言理解范畴 交叉熵作损失函数 输出层通过softmax得到分数对应的概率值
区间式答案
    指答案由文章中一段连续的语句组成。m个词的文章可能的区间式答案m(m-1)/2种 所有单词计算答案区间开头和结尾两个可能性分数
'''
import torch
import numpy as np

#设文本共m个词，prob_s是大小为m的开始位置概率，prob_e是大小为m的结束位置概率，均为一维PyTorch张量
#L为答案区间可以包含的最大的单词数
#输出为概率最高的区间在文本中的开始和结束位置
def get_best_interval(prob_s, prob_e, L):
    #获得m×m的矩阵，其中prob[i,j]=prob_s[i]×prob_e[j]
    prob = torch.ger(prob_s, prob_e) #两个一维输入向量x,y生成矩阵S[i,j]=x[i]*y[j]
    #将prob限定为上三角矩阵，且只保留主对角线及其右上方L-1条对角线的值，其他值清零
    #即如果i>j或j-i+1>L，设置prob[i, j] = 0
    prob.triu_().tril_(L - 1) #获得上三角矩阵 保证开始位置不晚于结束位置 区间长度不大于L
    #转化成为numpy数组
    prob = prob.numpy()
    #获得概率最高的答案区间，开始位置为第best_start个词, 结束位置为第best_end个词
    best_start, best_end = np.unravel_index(np.argmax(prob), prob.shape)
    return best_start, best_end

# 测试代码 摘自GitHub
sent_len = 20
L = 5
prob_s = torch.nn.functional.softmax(torch.randn(sent_len), dim=0)
prob_e = torch.nn.functional.softmax(torch.randn(sent_len), dim=0)
best_start, best_end = get_best_interval(prob_s, prob_e, L)
print(best_start, best_end)