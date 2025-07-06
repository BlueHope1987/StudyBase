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

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers 
import torch
model = r"helloPython\_Datasets\TinyLlama-1.1B-Chat-v1.0" #TinyLlama_v1.1_chinese"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16, 
    device_map="cpu",#"auto",
)

# tokenizer.pad_token="[PAD]" #
# tokenizer.pad_token = tokenizer.eos_token  # 推荐用eos_token作为pad_token
# tokenizer.padding_side="left" #
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
#豆包调整范例

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 选择合适的 TinyLlama 版本（推荐对话微调版，如 TinyLlama/TinyLlama-1.1B-Chat-v1.0）
model_name = r"helloPython\_Datasets\TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 注意：TinyLlama 可能需要手动设置 pad_token
tokenizer.pad_token = tokenizer.eos_token

# 加载模型，优先用 CPU 推理（匹配你的硬件），指定 dtype 避免自动选择不支持的类型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # 若用 FP16 报异常，先试 FP32（兼容性更好但慢）
    device_map="cpu",
    low_cpu_mem_usage=True  # 减少 CPU 内存占用
)

# 测试简单文本生成（非对话模式，类似 GPT-2 的续写）
prompt = "今天天气很好，"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
'''
'''
TinyLlama-1.1B-Chat-v1.0 输出：
今天天气很好，晴朗，温度 22 ℃，湿度 100%，�

豆包评论：

从你的经历来看，TinyLlama 的 Chinese 版不能用，更可能是模型本身质量 / 适配性问题，而非单纯 “下坏了” 或 “用法不对”，具体可以从这几个角度拆解：
一、最可能的核心原因：Chinese 版是 “非官方微调”，质量无保障
TinyLlama 官方（原作者）主要维护的是通用版和 Chat 版（如 TinyLlama-1.1B-Chat-v1.0），而 “Chinese 版” 通常是第三方（个人或小团队）基于原版微调的衍生版本。这类非官方微调可能存在几个问题：

训练不充分：比如中文语料不足、微调轮次不够，导致模型对中文输入的理解能力极弱，甚至无法生成有效文本（表现为回复为空）。
Checkpoint 损坏：微调过程中如果意外中断，保存的模型文件可能不完整（比如缺失关键层参数），加载后推理时会出现nan/inf（采样时数值异常）、输出为空等 “隐性错误”（文件大小可能正常，但内部参数无效）。
未适配对话场景：有些 Chinese 版可能仅针对 “文本续写” 微调，而非 “对话交互”，当你用对话式输入（比如直接喂 “今天天气很好，”）时，模型无法匹配训练目标，自然生成失败。
二、次要原因：用法细节的 “隐性不兼容”
即使模型本身没问题，Chinese 版也可能因为 “适配细节” 和 Chat 版不同，导致你之前的用法 “看似对，实则错”：

Prompt 格式要求更严格：Chat 版通常内置了对话模板（比如<s>用户：xxx</s>助手：），即使你输入简单文本也能兼容；但 Chinese 版可能需要严格遵循它的微调模板（比如必须带 “问答：”“生成：” 前缀），否则模型无法触发生成逻辑，直接输出空。
采样参数不兼容：非官方微调的模型可能对temperature、top_k等采样参数更敏感。比如原版 TinyLlama 适合temperature=0.7，但 Chinese 版可能因为训练时的数值稳定性问题，稍微调整参数就出现nan/inf（你之前遇到的错误）。
三、“模型下坏了” 的可能性较低（但可验证）
如果下载过程中没有断连、文件大小和官方标注一致（比如每个bin文件大小匹配），大概率不是 “下坏了”。验证方法很简单：

去 Hugging Face 找该 Chinese 版的 “模型卡片”（Model Card），看是否有其他用户反馈类似问题（比如评论区有人说 “输出为空”），如果有，基本实锤是模型本身问题；
用最简单的代码测试（比如直接续写 “你好”），如果依然输出空或报错，排除用法问题，锁定模型质量。
总结：非官方衍生版的 “踩坑率” 本就高
TinyLlama 这类轻量模型的生态里，第三方微调版本很多，但质量参差不齐 —— 有的是爱好者练手作品，有的甚至没经过完整测试就上传了。而 Chat 版作为官方或高口碑微调版本，经过了基础验证，适配性和稳定性自然更优。

你最后用 Chat 版成功跑通，其实已经间接证明：不是你的操作有问题，而是选的模型 “底子” 不够可靠～ 后续用 Gradio 套壳后，大概率能稳定用起来，恭喜踩坑结束！

'''

'''
TinyLlama_v1.1_chinese 输出：
今天天气很好， items items items items items items items items items items items items items items items items items items items items items items items items items items items items items items

开启采样后出现张量错误 豆包认为：
问题出在数值稳定性上，这在 CPU 推理（尤其是旧架构如 E3 v1231）上很常见。当启用采样（do_sample=True）时，模型需要计算 softmax 概率分布，
而某些中间值可能溢出为inf或下溢为nan。以下是针对性的解决方案：
一、问题根源分析
数值溢出：
当模型生成的 logits 值过大（如超过 700）时，exp(logits)会变为inf，导致 softmax 计算失败。
精度不匹配：
FP16 在 CPU 上计算时易溢出（最大值为 65504），即使指定torch.float32，某些中间计算仍可能使用 FP16。
模型架构问题：
TinyLlama 的 Attention 层或 LayerNorm 可能在 CPU 上计算不稳定，尤其是长序列或高温度（如 0.9）时。
它给出的方案：
添加数值稳定的 softmax 函数
降低温度并限制 logits 范围
使用更严格的量化方法
添加异常处理和回退机制
。。。
形成下列代码，虽然无效，但有参考价值
'''
'''
#豆包复杂调试范例
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

model_name = r"helloPython\_Datasets\TinyLlama_v1.1_chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 尝试8-bit加载（若FP32仍有问题）
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
except:
    # 回退到FP32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

def stable_softmax(logits, dim=-1):
    logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]
    probs = F.softmax(logits, dim=dim)
    probs = torch.nan_to_num(probs, nan=0.0)
    probs = probs / torch.sum(probs, dim=dim, keepdim=True)
    return probs

def generate_with_fallback(model, inputs, max_attempts=3):
    temperature = 0.7
    for attempt in range(max_attempts):
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # 增加最大生成长度
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3,
                renormalize_logits=True,
                bad_words_ids=[[tokenizer.unk_token_id]]
            )
            return outputs
        except Exception as e:
            print(f"生成失败（尝试 {attempt+1}/{max_attempts}）: {e}")
            temperature = max(0.3, temperature - 0.1)
    return None

# 使用Chat格式
prompt = """<|user|>今天天气很好，
<|assistant|>"""

inputs = tokenizer(prompt, return_tensors="pt")

# 打印调试信息
print(f"输入prompt: {prompt}")
print(f"输入IDs长度: {len(inputs.input_ids[0])}")

# 尝试生成
outputs = generate_with_fallback(model, inputs)

if outputs is not None:
    print(f"输出IDs长度: {len(outputs[0])}")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取模型回复部分
    response = response.split("<|assistant|>")[-1].strip()
    print(f"模型回复: {response}")
else:
    print("生成失败，尝试更激进的参数...")
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        print(f"模型回复: {response}")
    except Exception as e:
        print(f"最终尝试失败: {e}")
        #输出IDs长度: 74
        #模型回复: 
'''




'''
#原范例 极简
# RuntimeError: probability tensor contains either inf, nan or element 
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



'''
#范例2 
# RuntimeError: probability tensor contains either inf, nan or element 
# Github Copilot：TinyLlama_v1.1_chinese 很可能并不支持 apply_chat_template，直接传字符串prompt更稳妥。

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

# prompt = pipeline.tokenizer.apply_chat_template(messages, tokenzize=False, add_generation_prompt=True)
# ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed! For information about writing templates and setting the tokenizer.chat_template attribute, please see the documentation at https://huggingface.co/docs/transformers/main/en/chat_templating

#chat_template = {"default": "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|>"}
chat_template = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{content}<|END_OF_TURN_TOKEN|>"
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template)

outputs = pipeline(prompt,max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95) 
print(outputs[0]["generated_text"])

# max_new_tokens 生成最大token数量
# do_sample 是否启用采样
# temperature 控制生成文本的随机性
# top_k top_p 控制生成文本的多样性
'''
'''
# Github Copilot修正提示
# RuntimeError: probability tensor contains either inf, nan or element 

# 构造prompt字符串
prompt = (
    "You are a friendly chatbot who always responds in the style of a pirate.\n"
    "User: How many helicopters can a human eat in one sitting?\n"
    "Assistant:"
)

outputs = pipeline(
    prompt,
    max_new_tokens=128,  # 建议先用128，防止显存溢出
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
print(outputs[0]["generated_text"])
'''




#对话界面范例
'''
torch>=2.0
transformers>=4.35.0
gradio>=4.13.0
'''

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

torch.autograd.set_detect_anomaly(True)

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False


# Function to generate model predictions.
def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="Tinyllama_chatBot",
                 description="Ask Tiny llama any questions",
                 examples=['How to cook a fish?', 'Who is the president of US now?']
                 ).launch()  # Launching the web interface.

