import requests
import base64
import os



def call_qwen_vl_api(question, image_path, api_url="api_url", api_key="your_key"):
    """
    调用Qwen2.5-VL API服务
    """
    # 检查图片文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    
    # 获取图片文件的MIME类型
    file_ext = os.path.splitext(image_path)[1].lower()
    mime_type = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp'
    }.get(file_ext, 'image/jpeg')
    
    # 将图片转换为base64编码
    with open(image_path, "rb") as image_file:
        base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
        image_data = f"data:{mime_type};base64,{base64_encoded}"
    
    # 构造请求数据
    payload = {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "temperature": 0.6,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_data
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    }
    
    # 设置请求头
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    
    try:
        # 发送POST请求
        print(f"发送请求到: {api_url}")
        response = requests.post(api_url, json=payload, headers=headers, timeout=600)
        
        # 打印状态码和响应内容，用于调试
        print(f"响应状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        response.raise_for_status()  # 如果状态码不是200，会抛出异常
        return response
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    question = "描述一下这张图片"
    image_path = "./RcdagLAki.png"  # 替换为你的图片路径
    
    result = call_qwen_vl_api(question, image_path)
    
    if result:
        print("API响应结果:")
        print(result)