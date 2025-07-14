# https://www.runoob.com/pytorch/pytorch-text-classification.html

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.data import Field, TabularDataset, BucketIterator,LabelField
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
## Copilot 添加：处理数据集，随机抽取训练集和测试集 未分列不可用
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
print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
##
'''
'''
## Copilot 添加：处理数据集，随机抽取训练集和测试集 考虑选列 按需运行一次
import pandas as pd

df = pd.read_csv('helloPython/_Datasets/IMDB/IMDB-Movie-Data.csv')

# 以Description为文本，Rating为情感标签（>=7为正面，<7为负面）
df = df[['Description', 'Rating']].dropna()
df['label'] = (df['Rating'] >= 7).astype(int)
df = df.rename(columns={'Description': 'text'})

# 只保留text和label两列
df = df[['text', 'label']]

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_df.to_csv('helloPython/_Datasets/imdb/train.csv', index=False)
test_df.to_csv('helloPython/_Datasets/imdb/test.csv', index=False)
print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
'''


#数据预处理

# 定义字段处理
TEXT = Field(tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            include_lengths=True)
# LABEL = Field(sequential=False, use_vocab=False)
LABEL = LabelField(dtype=torch.float) # Copilot: 使用 LabelField 替代 Field 处理标签 LabelField会自动把'pos'/'neg'转为数字，适配你的模型 LABEL 会自动把 'pos' 映射为 1.0，'neg' 映射为 0.0，兼容你的二分类损失函数。 防止报错ValueError: could not convert string to float: 'pos'

# 用 torchtext 提供的imdb数据集
# datasets.IMDB.splits() 方法会自动下载数据集并返回训练集和测试集
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL) #第三方下载http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz 解压至项目根\.data\imdb\
print(f"Train: {len(train_data)}, Test: {len(test_data)}")
print(vars(train_data.examples[0]))

'''
# 加载数据集
train_data, test_data = TabularDataset.splits(
  #  path='./data',
    path='helloPython/_Datasets/imdb/',
    train='train.csv',
    test='test.csv',
    format='csv',
    fields=[('text', TEXT), ('label', LABEL)],
    skip_header=True  # 关键参数，跳过表头
)
'''
# 构建词汇表
TEXT.build_vocab(train_data,
                max_size=25000,
                vectors=Vectors(name=r"helloPython\_Datasets\glove.6B.100d.txt")) #vectors="glove.6B.100d" 改写本地
LABEL.build_vocab(train_data)
#glove.6B.100d.txt 是一个包含预训练词向量资源的压缩文件。 该词向量是由斯坦福大学训练
# glove.6B词向量是使用全局向量（Vectors for Word Representation）算法进行训练的，它是一种基于词共现统计的词向量训练方法。
# 特指包含100维的词向量，适用于各种任务中。
# 词汇表匹配：您可以通过TEXT.build_vocab方法，将您自定义的词汇表与glove词向量中的词进行匹配，创建出适合您需要的词向量。
# 获取词向量：一旦构建了新的词向量，您可以通过TEXT.vocab.vectors获取到这些词的向量表示，以便在后续任务中使用。
# git clone https://gitcode.com/open-source-toolkit/fd914
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
        self.dropout = nn.Dropout(0.6) #0.5 提高以防止过拟
       
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
HIDDEN_DIM = 128 #256 降低以避免过拟
OUTPUT_DIM = 1
N_LAYERS = 2

# 初始化模型
model = SentimentLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS)
model.embedding.weight.data.copy_(TEXT.vocab.vectors)

# 优化器和损失函数
#optimizer = optim.Adam(model.parameters())
optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)  # L2正则 以防过拟
criterion = nn.BCEWithLogitsLoss()

#训练循环

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
   
    model.train()
    #更频繁的打印进度方法
    for i, batch in enumerate(iterator):
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy(predictions, batch.label)
        # 每100个batch打印一次进度
        # if (i + 1) % 100 == 0:
        if (i + 1) % 1 == 0:
            print(f"  Batch {i+1}/{len(iterator)} processed")
            print(f"  Loss: {epoch_loss / (i + 1):.4f}, Accuracy: {epoch_acc / (i + 1):.4f}")

            # 绘制训练曲线 Copilot代码提示添加
            plt.clf()  # 清除当前图形
            plt.plot(range(i+1), [epoch_loss / (j + 1) for j in range(i+1)], label='Train Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title('Training Loss Progress')
            plt.legend()
            plt.pause(0.001)  # 暂停以更新图形

        if stop_event.is_set():
            print("检测到停止请求，正在停止批次...")
            break
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

    '''
    #原方法
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
    '''
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

# Copilot补充训练代码
# 构建数据迭代器
BATCH_SIZE = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text),
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# 如果有预训练模型，载入模型参数
try:
    model.load_state_dict(torch.load('helloPython/_Datasets/IMDB/best_model.pt'))
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("No pre-trained model found, starting from scratch.")


# 将模型和损失函数移动到设备
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
criterion = criterion.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# 训练过程
N_EPOCHS = 20
best_test_loss = float('inf')
test_loss = float('inf')
patience = 10
counter = 0

# 绘制训练曲线 Copilot代码提示添加
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()  # 开启交互模式
train_losses = []
test_losses = []
train_acc = []
test_acc = []
plt.show()  # 显示图形窗口


import threading
import sys
from queue import Queue

stop_event = threading.Event()
message_queue = Queue()

# 新增：键盘监听线程函数
def keyboard_listener():
    input()  # 等待用户输入
    stop_event.set()  # 设置停止标志
    message_queue.put("训练停止请求已发送")

# 启动键盘监听线程
keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
keyboard_thread.start()

print("按 Enter 键停止训练...")

stop_event.set() # ****程序控制：跳过训练循环直接进入推理****

# 训练循环
for epoch in range(N_EPOCHS):
    if stop_event.is_set():
        print("检测到停止请求，正在停止迭代...")
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            counter = 0
            torch.save(model.state_dict(), 'helloPython/_Datasets/IMDB/best_model.pt')
            print("模型已保存，进行后续过程。")
        break

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')

    # 绘制训练曲线 Copilot代码提示添加
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    # 绘制训练曲线
    plt.clf()  # 清除当前图形
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(train_acc, label='Train Acc')
    plt.plot(test_acc, label='Test Acc')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.pause(0.001)  # 暂停以更新图形
    plt.show()
    
    # 保存最佳模型
    # Early stopping 早停以避免过拟
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        counter = 0
        torch.save(model.state_dict(), 'helloPython/_Datasets/IMDB/best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break
'''
训练损失（train loss）和测试损失（test loss）是评估神经网络性能的两个关键指标，其变化趋势可反映模型训练状态及潜在问题。
正常学习状态
当‌train loss和test loss均持续下降‌，表明模型仍在有效学习，训练过程正常。 ‌
过拟合风险
若‌train loss持续下降但test loss趋于平稳或上升‌，通常意味着模型过拟合训练数据，需通过简化模型结构、增加数据多样性（如数据增强）或调整训练策略（如减少批次大小、引入dropout）来改善。 ‌
数据集质量问题
若‌train loss趋于平稳但test loss持续下降‌，可能因数据集标注错误或分布不均衡导致，需检查数据完整性及划分方式。 ‌
训练瓶颈
当‌train loss和test loss均趋于平稳‌，可能因学习率过低或批次大小过小导致训练停滞，需调整学习率或增加批次数量。 ‌
结构或参数问题
若‌两者同时上升‌，通常由网络结构设计缺陷（如层数过多）、参数配置不当或数据未清洗引起，需优化网络架构或参数设置。
'''



'''
# 参考训练过程代码 https://www.cnblogs.com/Fgociallo/p/18886660
#正式训练

BATCH_SIZE = 64 # 先确定好一个批次的样本数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 自动检测是否可用 GPU（cuda），否则使用 CPU
# 创建迭代器
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE, 
    device=device)
# 通过这行代码可以看到各样本的第 2 个词, 详情就不细说了
[TEXT.vocab.itos[i] for i in next(iter(train_iterator)).text[1, :]]

# 初始化组件
optimizer = optim.Adam(model.parameters()) # 使用Adam优化器
criterion = nn.BCEWithLogitsLoss() # 二分类损失函数（结合了 Sigmoid 激活和二元交叉熵损失）
model = model.to(device) # 将模型移至GPU（如果可用）
criterion = criterion.to(device)  # 损失函数也移至GPU

# 训练参数
N_EPOCHS = 10 # 训练轮数
best_valid_loss = float('inf') # 要求记录验证集上的最低损失，用于保存最佳模型

# 训练循环
for epoch in range(N_EPOCHS):
    start_time = time.time() # 记录epoch开始时间
    
    # 训练并评估
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    # 计算耗时
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # 保存最佳模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'wordavg-model.pt')
    
    # 打印日志
    print(f'迭代轮次: {epoch+1:02} | 迭代一轮时间: {epoch_mins}m {epoch_secs}s')
    print(f'\t训练损失: {train_loss:.3f} | 训练准确率: {train_acc*100:.2f}%')
    print(f'\t 验证损失: {valid_loss:.3f} |  验证准确率: {valid_acc*100:.2f}%')
'''



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

