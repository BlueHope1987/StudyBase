#https://hf-mirror.com/google-bert/bert-base-chinese

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertModel

model_name=r"helloPython\_Datasets\bert-base-chinese"

########################################
print("Bert分词任务测试")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

input_text="你好！你[MASK][MASK]过得怎样？"
masked_tokens=tokenizer(input_text,return_tensors="pt")["input_ids"]

predictions=model(masked_tokens)
predicted_token_id=predictions.logits.argmax(-1)

#待预测的词ID转换为文本
predicted_token=tokenizer.decode(predicted_token_id[0])
print(f"输入原文：{input_text}")
print(f"完整短语：{predicted_token}")


# Git Copilot提示：BERT模型要求输入的mask标记为 [MASK]（全部大写），而不是 [masked]；用 tokenizer.mask_token_id 找到mask位置，只解码mask位置的预测结果。
# 直接用 argmax(-1) 得到的是每个位置上概率最大的token id，但你需要找到 [MASK] 位置的预测结果，而不是全部token的最大概率。
# 正确流程 找到 [MASK] 的位置 取出该位置的预测分布 取概率最大的token id并decode
# 修改建议
input_text="如果感到快乐你就[MASK][MASK][MASK]。"
inputs = tokenizer(input_text, return_tensors="pt")
mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

with torch.no_grad():
    outputs = model(**inputs)
    mask_token_logits = outputs.logits[0, mask_token_index, :]
    top_token_id = mask_token_logits.argmax(dim=-1)
    predicted_token = tokenizer.decode(top_token_id)

print(f"输入原文：{input_text}")
print(f"预测的词是：{predicted_token}")

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],  # 一次编码两个句子，若没有text_pair这个参数，就一次编码一个句子

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补pad到max_length长度
    padding='max_length',   # 少于max_length时就padding
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,  # None表示不指定数据类型，默认返回list
)

print(out)

tokenizer.decode(out)

pretrained = BertModel.from_pretrained(model_name)
print(pretrained)

############################
print("Bert问答任务测试")

from transformers import BertTokenizer,BertForQuestionAnswering

input_text="Howe的目标是什么？"
context="Howe的目标是让每个人都过得很开心。Howe有三部分组成。Howe的售价非常便宜。Howe的安装非常方便。"

tokenizer=BertTokenizer.from_pretrained(model_name)
model=BertForQuestionAnswering.from_pretrained(model_name)

inputs=tokenizer(input_text,context,return_tensors="pt")
outputs=model(**inputs)
start_scores=outputs.start_logits
end_scores=outputs.end_logits

answer_start=torch.argmax(start_scores)
answer_end=torch.argmax(end_scores)+1
answer=tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
#answer=tokenizer.decode(outputs["start_logits"][0][answer_start:answer_end])
print(answer)

#简单交互界面示例
#https://blog.csdn.net/weixin_41194129/article/details/131984237