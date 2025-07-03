'''
豆包：
all-minilm-l6-v2：它是基于微软研发的 MiniLM 架构的轻量级语言模型，通过知识蒸馏技术从更大的模型中压缩而来。
它属于句子嵌入模型，主要功能是将句子和段落映射到低维向量空间，用于计算文本相似度、语义搜索等任务。
all-minilm-l6-v2 主要侧重于文本的向量化表示

要将 all-minilm-l6-v2 应用于 pipeline 问答任务，需结合其句子嵌入特性设计适配流程，核心是通过语义匹配辅助问答，而非直接生成回答。
以下是具体实现步骤及要点：

1. 明确模型定位：不直接生成答案，而是辅助语义匹配
all-minilm-l6-v2 作为轻量级句子嵌入模型，无法像 TinyLlama 等大语言模型直接生成回答，但可通过计算文本语义向量，为问答系统提供 “语义检索” 能力。例如：
在问答场景中，先通过该模型将问题转为向量，再从知识库中检索语义最匹配的文档段落，最后由其他模型（如 T5、BERT）基于检索结果生成答案。

2. 构建问答 pipeline 的核心流程

步骤 1：预处理问题与知识库
问题处理：用 tokenizer 对用户问题分词，输入 all-minilm-l6-v2 生成问题的语义向量（如 768 维嵌入）。
知识库向量化：将文档、FAQ 等拆分为段落，用同一模型生成每个段落的向量，存储为向量数据库（如 FAISS、Chroma）。

步骤 2：语义检索匹配相关文档
用问题向量在向量数据库中检索，找到余弦相似度最高的若干段落（如 Top 5），作为回答的参考信息。

步骤 3：结合检索结果生成答案
将检索到的段落和问题一起输入到生成模型（如 T5-small、LLaMA-Adapter），由生成模型基于上下文提炼答案。

3. 关键技术点与实现示例（以 Hugging Face 为例）

4. 注意事项
模型适配限制：若 pipeline 中已有大语言模型（如 GPT-3.5），all-minilm 的作用是优化检索效率，而非替代生成模型。
参数调优：检索时可调整max_length（如设为 64-128）平衡语义精度与效率，相似度阈值需根据知识库规模调整。
多语言场景：all-minilm-l6-v2 对英文支持更优，中文场景建议优先使用中文向量化模型（如uer/sbert-base-chinese-nli）。
通过以上流程，可将 all-minilm-l6-v2 作为问答 pipeline 中的 “语义检索模块”，提升知识库匹配的准确性，再结合生成模型完成最终回答。
'''


#（1）加载模型与向量化

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

# 加载模型与分词器
# model_name = "all-mpnet-base-v2"  # all-minilm-l6-v2同系列，用法一致
model_name = r"helloPython\_Datasets\all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 生成文本向量的函数
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(** inputs)
    # 取CLS token的输出作为整体向量
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding


#（2）向量数据库存储与检索

# 示例：用Chroma存储文档向量
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.create_collection(name="qa_knowledge")

# 假设已有文档段落列表docs
docs = ["文档段落1", "文档段落2", ...]
ids = [f"doc_{i}" for i in range(len(docs))]
embeddings = [get_text_embedding(doc)[0].tolist() for doc in docs]

# 插入向量数据库
collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=ids
)

# 检索示例：用户问题转向量后查询
user_question = "如何使用all-minilm做问答？"
question_emb = get_text_embedding(user_question)[0].tolist()
results = collection.query(
    query_embeddings=[question_emb],
    n_results=3  # 返回3个最相似段落
)

#（3）结合生成模型回答

from transformers import pipeline

# 加载轻量级生成模型（如T5）
qa_pipeline = pipeline("question-answering", model="t5-base", tokenizer="t5-base") #豆包给的模型很大 8.6GB

# 用检索到的段落生成答案
context = " ".join(results["documents"][0])  # 合并检索到的段落
answer = qa_pipeline(question=user_question, context=context)
print(answer["answer"])


