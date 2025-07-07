# https://www.runoob.com/pytorch/pytorch-text-classification.html

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
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
                vectors="glove.6B.100d")
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

'''
def predict_sentiment(model, sentence):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()

'''

'''
示例预测
positive_review = "This movie was fantastic! I really enjoyed it."
negative_review = "The film was terrible and boring."

print(f"Positive review score: {predict_sentiment(model, positive_review):.4f}")
print(f"Negative review score: {predict_sentiment(model, negative_review):.4f}")
'''