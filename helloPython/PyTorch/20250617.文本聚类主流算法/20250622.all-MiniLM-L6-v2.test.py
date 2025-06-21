#https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2

#这是一个句子变换模型（sentence_transformers）：它将句子和段落映射到384维的密集向量空间，并可用于像聚类或语义搜索这样的任务。


from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer(r"helloPython\PyTorch\20250617.文本聚类主流算法\Datasets\all-MiniLM-L6-v2")

#生成嵌入
embeddings = model.encode(sentences)
#print(embeddings)
print(embeddings.shape)

#计算相似性
similarities=model.similarity(embeddings,embeddings)
print(similarities)
