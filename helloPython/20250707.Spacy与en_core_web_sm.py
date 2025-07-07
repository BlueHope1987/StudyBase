#https://www.bytezonex.com/archives/Z_0fgKYG.html
'''
Spacy与en_core_web_sm安装详细指南：超实用干货，小白也能轻松上手！

·用Spacy和en_core_web_sm开启您的自然语言处理之旅
·自然语言处理是什么？
自然语言处理（NLP）是计算机理解和处理人类语言的技术。从识别文本中的情绪到提取关键信息，NLP 广泛应用于各种领域。
·为什么选择Spacy和en_core_web_sm？
Spacy 是一个强大的 Python 库，提供了丰富的 NLP 工具，包括：
-分词
-词性标注
-命名实体识别
en_core_web_sm 是 Spacy 的预训练模型，针对广泛的自然语言处理任务进行了优化，特别是对于英语。
·安装Spacy和en_core_web_sm
Spacy 是一个 Python 库，提供各种 NLP 工具，例如分词和词性标注。en_core_web_sm 是 Spacy 的预训练模型，专门用于英语的 NLP 任务。
    ·使用 Anaconda
    conda install -c conda-forge spacy
    ·使用 pip
    pip install spacy
    安装 Spacy 后，安装 en_core_web_sm：
    python -m spacy download en_core_web_sm
    ·验证安装
    python -m spacy validate

·使用Spacy和en_core_web_sm
使用 NLP 模型 doc._.sentiment.cats 访问文本的情感得分。
'''

#加载 en_core_web_sm 模型：

import spacy

nlp = spacy.load("en_core_web_sm")

#分词
text = "I love natural language processing."
doc = nlp(text)
for token in doc:
    print(token.text, token.pos_)

#词性标注
text = "I love natural language processing."
doc = nlp(text)
for token in doc:
    print(token.text, token.pos_, token.tag_)



#示例代码：情感分析

import spacy

# 加载 en_core_web_sm 模型
nlp = spacy.load("en_core_web_sm")

# 分析文本的情绪
text = "This is a great movie!"
doc = nlp(text)

# 获取情感标记
emotion = doc._.sentiment.cats["positive"] #？？？

# 打印情绪得分
print(emotion)

