#https://hf-mirror.com/TinyLlama/TinyLlama_v1.1_chinese
#1.1B参数量的大语言模型
'''
下载：
set GIT_LFS_SKIP_SMUDGE=1
git clone https://hf-mirror.com/TinyLlama/TinyLlama_v1.1_chinese
cd TinyLlama_v1.1_chinese
git lfs pull --include="*.bin"
'''
#pip3 install accelerate

from transformers import AutoTokenizer
import transformers 
import torch
model = r"helloPython\_Datasets\TinyLlama_v1.1_chinese"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16, #torch.float16,
    device_map="auto",
)

tokenizer.pad_token="[PAD]" #
tokenizer.padding_side="left" #
'''
    运行时错误：#probability tensor contains either `inf`, `nan` or element < 0
    据网上三种解决办法：
    1.
    tokenizer.pad_token="[PAD]"
    tokenizer.padding_side="left"
    2.使用torch.bfloat16
    3.torch版本<=2.1
'''


'''
#原范例

sequences = pipeline( 
    'The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization, we can achieve this within a span of "just" 90 days using 16 A100-40G GPUs 🚀🚀. The training has started on 2023-09-01.',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    repetition_penalty=1.5,
    eos_token_id=tokenizer.eos_token_id,
    max_length=500,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
'''

#范例2

messages=[
    {
        "role":"system",
        "content":"You are a friendly chatbot who always responds in the style of a pirate",
    },
    {
	    "role":"user",
	    "content":"How many helicopters can a human eat in one sitting?"
    },
]

prompt = pipeline.tokenizer.apply_chat_template(messages, tokenzize=False, add_generation_prompt=True)
outputs = pipeline(prompt,max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95) 
print(outputs[0]["generated_text"])

# max_new_tokens 生成最大token数量
# do_sample 是否启用采样
# temperature 控制生成文本的随机性
# top_k top_p 控制生成文本的多样性