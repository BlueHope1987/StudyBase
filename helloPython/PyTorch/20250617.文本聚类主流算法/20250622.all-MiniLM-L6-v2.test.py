#https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2

#这是一个句子变换模型（sentence_transformers）：它将句子和段落映射到384维的密集向量空间，并可用于像聚类或语义搜索这样的任务。
#all-MiniLM-L6-v2 是一个基于 MiniLM-L6-H384-uncased 预训练模型微调得到的模型，主要用于句子嵌入、句子相似度计算和语义搜索等任务
#适用场景语义搜索、文本聚类、句子相似度计算、信息检索、文本分类、问答系统



from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer(r"helloPython\_Datasets\all-MiniLM-L6-v2")

#生成嵌入
embeddings = model.encode(sentences)
#print(embeddings)
print(embeddings.shape)

#计算相似性
similarities=model.similarity(embeddings,embeddings)
print(similarities)


print("通过 Hugging Face 的 transformers 库可以加载和使用该模型")


from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"helloPython\_Datasets\all-MiniLM-L6-v2")
model = AutoModel.from_pretrained(r"helloPython\_Datasets\all-MiniLM-L6-v2")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model(**inputs)




print("文本相似度比较")


import torch
import numpy as np

# 指定模型名称
model_name = r"helloPython\_Datasets\all-MiniLM-L6-v2"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)



sentences = ["Hello, world!", "Hi there!"]

# 编码文本
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 提取句嵌入（均值池化）
embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

# 计算余弦相似度
similarity = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"相似度: {similarity:.4f}")  # 示例输出: 相似度: 0.5033