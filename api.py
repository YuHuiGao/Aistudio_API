from flask import Flask, request, jsonify, Response
import json
import requests
from PIL import Image
from io import BytesIO
import sys 
sys.path.append('/home/aistudio/external-libraries')
import argparse
import os
import uuid  # 新增：导入uuid模块
import base64
import paddle
from paddlemix.models.qwen2_5_vl import MIXQwen2_5_Tokenizer
from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from paddlemix.processors.qwen2_5_vl_processing import (
    Qwen2_5_VLImageProcessor,
    Qwen2_5_VLProcessor,
    process_vision_info,
)
from paddlemix.utils.log import logger

print("代码开始执行")
app = Flask(__name__)

model_path = "./Qwen2.5-VL-3B-Instruct"

top_p = 0.01
temperature = 0.01
max_new_tokens = 256
dtype = "bfloat16"
attn_implementation = "eager"

paddle.seed(seed=0)
compute_dtype = dtype
if "npu" in paddle.get_device():
    is_bfloat16_supported = True
else:
    is_bfloat16_supported = paddle.amp.is_bfloat16_supported()
if compute_dtype == "bfloat16" and not is_bfloat16_supported:
    logger.warning("bfloat16 is not supported on your device,change to float32")
    compute_dtype = "float32"
print("compute_dtype", compute_dtype)

paddle.set_default_dtype(compute_dtype)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, dtype=compute_dtype, attn_implementation=attn_implementation
)

image_processor = Qwen2_5_VLImageProcessor()
tokenizer = MIXQwen2_5_Tokenizer.from_pretrained(model_path)
min_pixels = 256*28*28 # 200704
max_pixels = 1280*28*28 # 1003520
processor = Qwen2_5_VLProcessor(image_processor, tokenizer, min_pixels=min_pixels, max_pixels=max_pixels)


def process_image(image_source):
    """处理图像，HTTP链接直接返回网址，base64编码保存本地并返回路径"""
    # 定义本地图片保存目录（可根据需要修改）
    SAVE_DIR = "./images"
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    try:
        if image_source.startswith(('http://', 'https://')):
            # HTTP/HTTPS链接直接返回原网址
            return image_source
        else:
            # 处理base64编码的图像
            # 去除可能的base64前缀（如data:image/png;base64,）
            if 'base64,' in image_source:
                image_source = image_source.split('base64,')[1]
            
            # 解码base64数据
            image_data = base64.b64decode(image_source)
            image = Image.open(BytesIO(image_data))
            
            # 生成唯一文件名（避免冲突）
            file_name = f"{uuid.uuid4().hex}.png"  # 使用UUID确保唯一性
            save_path = os.path.join(SAVE_DIR, file_name)
            
            # 保存图像到本地
            image.convert("RGB").save(save_path)  # 统一转为RGB格式保存
            
            # 返回本地文件路径（可根据需要返回绝对路径或相对路径）
            return os.path.abspath(save_path)
    
    except Exception as e:
        raise ValueError(f"图像处理失败: {str(e)}")

def process_request_data(request_data):
    """处理请求数据，转换为模型输入格式"""
    messages = request_data.get("messages", [])
    processed_messages = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        processed_content = []
        for item in content if isinstance(content, list) else [{"type": "text", "text": content}]:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image":
                image = process_image(item["image"])
                processed_content.append({"type": "image", "image": image})
        
        processed_messages.append({"role": role, "content": processed_content})
    print(processed_messages)
    return processed_messages

def generate_response(messages, temperature=0.6):
    """调用模型生成响应"""
    # 准备模型输入
    texts = [processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

    # Preparation for inference
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pd",
    )

    # Inference: Generation of the output
    generated_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p
    )  # already trimmed in paddle

    output_text = processor.batch_decode(
        generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

@app.route('/v1/chat/completions',methods=['POST'])
def chat_completions():
    """处理聊天完成请求"""
    print("收到处理请求")
    # 解析请求数据
    try:
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"error": {"message": "无效的请求数据", "type": "invalid_request"}}), 400
    except Exception as e:
        return jsonify({"error": {"message": f"解析请求失败: {str(e)}", "type": "parse_error"}}), 400
    # 提取参数
    temperature = request_data.get("temperature", 0.6)

    
    try:
        # 处理消息
        processed_messages = process_request_data(request_data)
        
        response = generate_response(processed_messages, temperature)
        return response
    
    except Exception as e:
        return jsonify({"error": {"message": f"处理请求失败: {str(e)}", "type": "server_error"}}), 500

@app.route('/')
def home():
    return 'Qwen2.5-VL API服务正在运行'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, threaded=False)