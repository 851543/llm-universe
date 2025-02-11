import os
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，还需要做如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

from openai import OpenAI

# 配置 Ollama 客户端
client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama 服务器地址
    api_key='ollama'                       # 占位符 API 密钥
)

def gen_ollama_messages(prompt):
    '''
    构造 Ollama 模型请求参数 messages
    请求参数：
    prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages

def get_completion(prompt, model="qwen2.5:14b", temperature=0):
    '''
    获取 Ollama 模型调用结果
    请求参数：
    prompt: 对应的提示词
    model: 使用的模型，默认为 llama2，也可以使用其他模型如 mistral
    temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~2。
                温度系数越低，输出内容越一致。
    '''
    try:
        response = client.chat.completions.create(
            model=model,
            messages=gen_ollama_messages(prompt),
            temperature=temperature,
        )
        if len(response.choices) > 0:
            return response.choices[0].message.content
        return "生成答案错误"
    except Exception as e:
        return f"发生错误: {str(e)}"

if __name__ == '__main__':
    # 示例使用方法
    result = get_completion("你好！")
    print(result)