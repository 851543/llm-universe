# coding:utf8
# Python基础练习
# 练习时间: 2025/2/11 11:16
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())


# 如果你需要通过代理端口访问，你需要如下配置
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def openai_embedding(text: str, model: str = None):
    # 获取环境变量 OPENAI_API_KEY
    client = OpenAI(base_url='http://localhost:11434/v1',  # Ollama 服务器地址
                    api_key='ollama')

    # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model = "qwen2.5:14b"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response


response = openai_embedding(text='要生成 embedding 的输入文本，字符串形式。')
print(response)
print(f'返回的embedding类型为：{response.object}')
