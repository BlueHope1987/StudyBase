# https://www.runoob.com/pytorch/pytorch-text-classification.html

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import Vectors
'''
发生异常: OSError
[WinError 127] 找不到指定的程序。
  File "D:\StudyBase\helloPython\PyTorch\20230704.文本情感分析项目演练.py", line 6, in <module>
    from torchtext.data import Field, TabularDataset, BucketIterator
OSError: [WinError 127] 找不到指定的程序。

Copilot:版本不兼容 
pip show torchtext
pip uninstall torchtext
pip install torchtext==0.6.0 （没有0.8.1版）
升级到最新 pip3 install torch torchtext -i https://pypi.tuna.tsinghua.edu.cn/simple -U

https://stackoverflow.org.cn/questions/69765669
自 0.9 版本以来，字段是 Torchtext 的遗留功能。您链接的那篇文章来自该版本之前。
如果您拥有最新的 torchtext，但正在尝试使用旧版功能，则需要使用 torchtext.legacy。
（经测试0.18.0不可用）
'''
import spacy
import numpy as np

'''
安装依赖
pip install torch torchtext spacy
python -m spacy download en_core_web_sm

en_core_web_sm 是 Spacy 库中的一个预训练模型，专门用于处理英语的自然语言处理（NLP）任务。
Spacy 是一个强大的 Python 库，提供了丰富的 NLP 工具，包括分词、词性标注和命名实体识别

使用IMDB电影评论数据集，包含50,000条带有情感标签(正面/负面)的评论。
'''

'''
## Copilot 添加：处理数据集，随机抽取训练集和测试集 按需运行
# git clone https://github.com/EtherealShen/IMDB/
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_csv('helloPython\_Datasets\IMDB\IMDB-Movie-Data.csv')  # 假设您的数据文件名为 all.csv

# 随机划分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# 保存为新的csv文件
train_df.to_csv('helloPython/_Datasets/imdb/train.csv', index=False)
test_df.to_csv('helloPython/_Datasets/imdb/test.csv', index=False)
##
'''


#数据预处理

# 定义字段处理
TEXT = Field(tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False)

# 加载数据集
train_data, test_data = TabularDataset.splits(
  #  path='./data',
    path='helloPython/_Datasets/imdb/',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)]
)

# 构建词汇表
TEXT.build_vocab(train_data,
                max_size=25000,
                vectors=Vectors(name=r"helloPython\_Datasets\glove.6B.100d.txt")) #vectors="glove.6B.100d" 改写本地

#glove.6B.100d.txt 是一个包含预训练词向量资源的压缩文件。 该词向量是由斯坦福大学训练
# glove.6B词向量是使用全局向量（Vectors for Word Representation）算法进行训练的，它是一种基于词共现统计的词向量训练方法。
# 特指包含100维的词向量，适用于各种任务中。
# 词汇表匹配：您可以通过TEXT.build_vocab方法，将您自定义的词汇表与glove词向量中的词进行匹配，创建出适合您需要的词向量。
# 获取词向量：一旦构建了新的词向量，您可以通过TEXT.vocab.vectors获取到这些词的向量表示，以便在后续任务中使用。

#模型实现

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)
       
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)
    
#模型训练

# 模型参数
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2

# 初始化模型
model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)

# 优化器和损失函数
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

#训练循环

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
   
    model.train()
   
    for batch in iterator:
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        epoch_loss += loss.item()
        epoch_acc += accuracy(predictions, batch.label)
       
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#模型评估

#评估函数

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
   
    model.eval()
   
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            loss = criterion(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += accuracy(predictions, batch.label)
           
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

#准确率计算

def accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

#模型应用

#预测新文本

## 由Copilot添加声明
# 加载Spacy英文模型
nlp = spacy.load('en_core_web_sm')
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_sentiment(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()



#示例预测

positive_review = "This movie was fantastic! I really enjoyed it."
negative_review = "The film was terrible and boring."

print(f"Positive review score: {predict_sentiment(model, positive_review):.4f}")
print(f"Negative review score: {predict_sentiment(model, negative_review):.4f}")

#文心一言生成的五段好评影评 每段影评均突出不同电影类型（奇幻、剧情、喜剧、爱情、动作）的亮点，语言充满热情且细节丰富，适合用于表达真挚好评。
print(f"Pegative review score: {predict_sentiment(model, "This film is a masterpiece of visual storytelling! From the breathtaking cinematography to the emotionally resonant score, every frame feels meticulously crafted. The characters are layered and unforgettable, and the plot twists kept me on the edge of my seat until the final credits. A must-watch for anyone who believes in the power of cinema to transport us to another world."):.4f}")
print(f"Pegative review score: {predict_sentiment(model, "A hauntingly beautiful film that lingers long after the lights come up. The performances are raw and authentic, especially the lead actress who delivers a career-defining role. The director’s use of silence and subtle symbolism creates an atmosphere of tension and melancholy that’s both mesmerizing and deeply moving. I left the theater feeling utterly spellbound."):.4f}")
print(f"Pegative review score: {predict_sentiment(model, "Hilarious, heartwarm, and surprisingly profound! This comedy manages to balance side-splitting humor with genuine emotional stakes. The chemistry between the cast is electric, and the script is filled with witty dialogue that never feels forced. It’s the kind of film that makes you laugh, cry, and then immediately want to rewatch it with friends. A modern classic!"):.4f}")
print(f"Pegative review score: {predict_sentiment(model, "A sweeping epic that redefines romance on screen! The cinematography is lush, the costumes are exquisite, and the chemistry between the leads is palpable. What sets this film apart is its willingness to explore the complexities of love—joy, sacrifice, and resilience—without resorting to clichés. I was utterly captivated from start to finish. Bravo to the entire team!"):.4f}")
print(f"Pegative review score: {predict_sentiment(model, "An adrenaline-fueled thrill ride that delivers on every level! The action sequences are inventive and jaw-dropping, but what truly elevates this film is its heart. The protagonist’s journey from vulnerability to strength is inspiring, and the supporting cast adds depth to the story. This isn’t just a superhero movie—it’s a celebration of courage and hope. I can’t wait to see what comes next!"):.4f}")

#文心一言生成的五段差评影评 每段影评均针对不同问题（剧情混乱、节奏拖沓、特效依赖、类型混搭失败、对原作不尊重）展开犀利批评，语言直接且充满讽刺，适合用于表达强烈不满。
print(f"Negative review score: {predict_sentiment(model, "This film is a chaotic mess of clichés and poor decisions! The pacing drags endlessly, the characters are one-dimensional, and the plot twists feel forced and predictable. The supposed 'thrills' fall flat, and the dialogue is so cringe-worthy it made me laugh—for all the wrong reasons. Save your time and money; this is a dumpster fire of a movie."):.4f}")
print(f"Negative review score: {predict_sentiment(model, "I’ve never felt so bored in a theater. This pretentious 'art-house' film is two and a half hours of slow-motion shots, mumbled dialogue, and zero narrative coherence. The director seems more interested in showing off their 'vision' than telling a story that matters. By the end, I was actively rooting for the credits to roll. Avoid at all costs."):.4f}")
print(f"Negative review score: {predict_sentiment(model, "What a disappointment! This blockbuster relies entirely on over-the-top CGI and deafening explosions to distract from its paper-thin plot and cardboard characters. The action scenes are so poorly edited they’re impossible to follow, and the 'humor' feels like it was written by a middle schooler. Even the most die-hard fans of the genre will leave feeling cheated."):.4f}")
print(f"Negative review score: {predict_sentiment(model, "This film tries to mix romance and horror but ends up failing spectacularly at both. The jokes are stale, the horror elements are laughably tame, and the chemistry between the leads is nonexistent. It’s as if the writers threw random scenes into a blender and hoped for the best. The result? A cringeworthy disaster that’s not even 'so bad it’s good'—just bad."):.4f}")
print(f"Negative review score: {predict_sentiment(model, "As a fan of the original series, this reboot is an insult to everything that made the franchise great. The story is riddled with plot holes, the new characters are irritating, and the special effects look like they were pulled from a low-budget video game. The filmmakers clearly didn’t care about the source material, and it shows. This isn’t just a bad movie—it’s a betrayal of its audience."):.4f}")

