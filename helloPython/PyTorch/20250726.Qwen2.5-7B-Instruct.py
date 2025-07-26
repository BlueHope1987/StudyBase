# https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct
# 通义千问2.5-7B-Instruct
'''
下载：
set GIT_LFS_SKIP_SMUDGE=1
git clone https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct
cd Qwen2.5-7B-Instruct
git lfs pull --include="*.bin"

代码参考：
20250623.TinyLlama_v1.1_chinese.py
https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct
https://blog.csdn.net/qq839019311/article/details/143110729
'''


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers 
import torch
model = r"helloPython\_Datasets\Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model)

'''
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16, 
    device_map="cpu",#"auto",
)
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
    history_openai_format=[{
        "role":"system",
        "content":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    }]

    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({
            "role":"assistant",
            "content":assistant
        })
    history_openai_format.append({"role": "user", "content": message})

    stop = StopOnTokens()


    model_inputs = tokenizer([history_openai_format], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        torch_dtype=torch.qint8, # 8位量化
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        # top_k=50,
        temperature=0.45,
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
                 title="通义千问2.5-7B-Instruct chatBot测试",
                 description="Ask 通义千问2.5-7B-Instruct any questions",
                 examples=['How to cook a fish?', 'Who is the president of US now?']
                 ).launch()  # Launching the web interface.

