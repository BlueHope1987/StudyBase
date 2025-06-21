#https://hf-mirror.com/google-bert/bert-base-chinese

from transformers import AutoTokenizer, AutoModelForMaskedLM,BertModel

model_name=r"helloPython\_Datasets\bert-base-chinese"

########################################
print("Bert分词任务测试")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

input_text="你好！你今天过得怎样？"
masked_tokens=tokenizer(input_text,return_tensors="pt")["input_ids"]

predictions=model(masked_tokens)
predicted_token_id=predictions.logits.argmax(-1)

#待预测的词ID转换为文本
predicted_token=tokenizer.decode(predicted_token_id[0])
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
import torch
from transformers import BertTokenizer,BertForQuestionAnswering

context=""
#input_text="Howe的目标是什么？"
#context="Howe的目标是让每个人都过得很开心。Howe有三部分组成。Howe的售价非常便宜。Howe的安装非常方便。"

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