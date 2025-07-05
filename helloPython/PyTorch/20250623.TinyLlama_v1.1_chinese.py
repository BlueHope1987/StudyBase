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
model = r"helloPython\_Datasets\TinyLlama_v1.1_chinese"
'''
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float32,#torch.float16, 
    device_map="cpu",#"auto",
)
'''
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



#豆包调整范例

import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"是否支持GPU: {torch.cuda.is_available()}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("uer/gpt2-chinese",mirror="https://hf-mirror.com" )  # 用中文模型测试  # 用镜像加速
print(f"Tokenizer加载成功，词表大小: {len(tokenizer)}")


'''
#原范例
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
'''
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

torch.autograd.set_detect_anomaly(True)

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained(r"helloPython\_Datasets\TinyLlama_v1.1_chinese")
model = AutoModelForCausalLM.from_pretrained(r"helloPython\_Datasets\TinyLlama_v1.1_chinese")

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

'''