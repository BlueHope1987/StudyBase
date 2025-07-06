#0705与豆包一起反复调试TinyLlama_v1.1_chinese不可用 张量异常 大概二十多个回合 随后换此模型 一次成功
# 但其本质上是一个文本生成模型（Causal Language Model，因果语言模型），而不是专门优化过的对话模型（Conversational Model）。
# 它的核心能力是 “根据前文预测下一个 token”，但缺乏对话模型的关键特性

'''
git clone https://hf-mirror.com/uer/gpt2-chinese-cluecorpussmall
cd helloPython\_Datasets\gpt2-chinese-cluecorpussmall

配置环境：

conda create -n tinyllama_env python=3.9 -y
conda activate tinyllama_env
# 先装PyTorch（CPU版本，不装CUDA）
pip install torch==2.0.1  # 你之前提到的版本，和模型更匹配
# 装Transformers（核心库）
pip install transformers==4.35.2  # 中等版本，兼容多数模型
# 装加速库（可选，但能提升性能）
pip install accelerate sentencepiece
# 解决 A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 as it may crash
pip install numpy==1.26.4

运行时包：
Package            Version
------------------ ---------
accelerate         1.8.1
certifi            2025.6.15
charset-normalizer 3.4.2
colorama           0.4.6
filelock           3.18.0
fsspec             2025.5.1
huggingface-hub    0.33.2
idna               3.10
Jinja2             3.1.6
MarkupSafe         3.0.2
mpmath             1.3.0
networkx           3.2.1
numpy              1.26.4
packaging          25.0
pip                25.1
psutil             7.0.0
PyYAML             6.0.2
regex              2024.11.6
requests           2.32.4
safetensors        0.5.3
setuptools         78.1.1
sympy              1.14.0
tokenizers         0.15.2
torch              2.0.1
tqdm               4.67.1
transformers       4.35.2
typing_extensions  4.14.1
urllib3            2.5.0
wheel              0.45.1

'''
'''
#测试代码 通过

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"helloPython\_Datasets\gpt2-chinese-cluecorpussmall"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 消除pad_token警告

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 输入文本（添加引导词，让模型更聚焦）
input_text = "你过得怎样？请用中文回答："
inputs = tokenizer(input_text, return_tensors="pt")

# 生成参数微调
outputs = model.generate(
    **inputs,
    max_new_tokens=300,  
    do_sample=True,
    temperature=0.2,    
    top_k=40,
    repetition_penalty=1.5,
    no_repeat_ngram_size=2
)

# 解码并处理
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
print(generated_text)
'''


#假装说话部分 答非所问且表达混乱
'''
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = r"helloPython\_Datasets\gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")

def chat(prompt, history=""):
    # 设计对话格式：用"用户："和"模型："区分轮次
    input_text = f"{history}用户：{prompt}\n模型："
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        # 修正：使用eos_token_id或自定义结束字符串
        eos_token_id=tokenizer.encode("\n用户：")[0]  # 遇到换行+用户：就停止
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(" ", "")
    # 提取模型回复部分
    response = response.split("模型：")[-1].split("\n用户：")[0]
    return response, f"{input_text}{response}\n"

# 测试多轮对话
history = ""
while True:
    user_input = input("你：")
    if user_input == "退出":
        break
    response, history = chat(user_input, history)
    print(f"模型：{response}")
'''