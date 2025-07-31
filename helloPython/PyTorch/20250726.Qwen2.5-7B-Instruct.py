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


import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
# 模型路径

model = r"helloPython\_Datasets\Qwen2.5-7B-Instruct"

'''
# 正确配置8-bit量化
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # 启用8-bit量化
    llm_int8_threshold=6.0,  # 阈值控制异常值处理
)
'''

# 豆包：如果需要4bit量化 使用该代码 内存占用约3.5G
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # 指定要以 4-bit 精度转换和加载模型。
    bnb_4bit_compute_dtype=torch.float16,  # 计算精度 计算数据类型用于更改计算期间将使用的数据类型。默认情况下，计算数据类型设置为 float32，但可以设置为 bf16 以提高速度。
    bnb_4bit_use_double_quant=True,  # 双重量化，进一步减少内存 使用嵌套量化来提高内存效率的推理和训练。
    bnb_4bit_quant_type="nf4",  # 使用NormalFloat 4-bit量化
)

# 多次尝试运行python内存反复波动过大 最小数十兆 疑似没有成功加载模型 vsc卡顿崩溃 偶尔可以开放gradio但无法推理
# 模型过大 尝试可以在vsc外简单终端尝试运行 速度更快 且深入更稳
# 但目前尝试在非管理员状态下完成载入后提示
# Cannot find empty port in range: 7860-7860. You can specify a different port by setting the GRADIO_SERVER_PORT environment variable or passing the `server_port` parameter to `launch()`.
'''
cmd命令行
conda activate pyNN
f: 
cd F:\Documents\Marvin\Source\Repos\BlueHope1987\StudyBase
python F:\Documents\Marvin\Source\Repos\BlueHope1987\StudyBase\helloPython\PyTorch\20250726.Qwen2.5-7B-Instruct.py
'''



# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,  # 应用量化配置
    device_map="cpu", 
    trust_remote_code=True,  # 允许执行模型特定代码
    low_cpu_mem_usage=True,  # 减少CPU内存使用
)

# 定义停止条件
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [tokenizer.eos_token_id]  # 使用模型的EOS token
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

# 生成函数
def predict(message, history):
    # 构建对话历史
    history_openai_format = [{
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    }]
    
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    # 编码输入 （返回input_ids和attention_mask）
    model_inputs = tokenizer.apply_chat_template(
        history_openai_format, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(model.device)



    # 创建流式生成器
    streamer = TextIteratorStreamer(
        tokenizer, 
        timeout=1800.0, # 30秒不够反应 设定30分钟的超时 
        skip_prompt=True, 
        skip_special_tokens=True
    )
    
    # 生成参数
    generate_kwargs = {
            "input_ids": model_inputs,  # 显式指定input_ids
             # torch_dtype=torch.qint8, # 8位量化 豆包：仅在生成时生效，但模型加载时仍使用完整精度（FP16/FP32），导致初始加载就占用大量内存（7B 模型 FP16 约 14GB）
            "streamer": streamer,
            "max_new_tokens": 256, #适当增大
            "temperature": 0.9, #0.7->0.9 提高随机性，减少提前终止
            "top_p": 0.95,
            "do_sample": True,
            "num_beams": 1,
            #"stopping_criteria": StoppingCriteriaList([StopOnTokens()]) # 暂时注释掉停止条件
        }
    
    # 在主线程中生成（避免内存复制）
    # 注意：这可能导致UI在生成期间暂时无响应
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    
    # 流式输出
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# 启动Gradio界面
gr.ChatInterface(
    predict,
    title="通义千问2.5-7B-Instruct ChatBot",
    description="Ask Qwen2.5-7B any questions",
    examples=["你好", "推荐一本关于AI的书", "解释一下量子计算"]
).launch(
'''
    share=False,  # 设为True可生成公共分享链接
    server_name="0.0.0.0",  # 绑定所有网络接口
    server_port=7860,  # 端口号
    max_threads=4,  # 限制最大线程数，避免资源竞争
    show_error=True  # 显示详细错误信息
'''
)