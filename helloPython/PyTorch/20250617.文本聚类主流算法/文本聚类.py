#https://mp.weixin.qq.com/s/ADJB61EKzPM5y0ENbMbqoA
#文本聚类效果差？5种主流算法性能测试帮你找到最佳方案

#pip3 install pandas pyarrow fastparquet sentence_transformers tf-keras

import pandas as pd
from IPython.display import display, HTML, Image
 
df=pd.read_parquet(r"helloPython\PyTorch\20250617.文本聚类主流算法\Datasets\train-00000-of-00001.parquet")  
#https://hf-mirror.com/datasets/billingsmoore/text-clustering-example-data/tree/main

display(df.head())

#Billingsmoore提供的文本聚类示例数据集，该数据集包含925个英语句子，每个句子都标注了相应的主题类别

#数据集中不同主题的分布情况如下：

print(df.topic.value_counts())

from sentence_transformers import SentenceTransformer
 
model=SentenceTransformer(r"helloPython\PyTorch\20250617.文本聚类主流算法\Datasets\all-MiniLM-L6-v2")
'''
模型过大已忽略路径 下载config.json pytorch_model.bin tokenizer.json tokenizer_config.json vocab.txt
https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2/tree/main
''' 
sentences=df.text.to_list()  
embeddings=model.encode(sentences)  
print(embeddings.shape)