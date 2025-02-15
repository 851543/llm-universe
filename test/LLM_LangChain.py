import os
import openai
from dotenv import load_dotenv, find_dotenv

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

from langchain_community.llms import Ollama

# 这里我们将参数temperature设置为0.0，从而减少生成答案的随机性。
# 如果你想要每次得到不一样的有新意的答案，可以尝试调整该参数。
llm = Ollama(
    model="qwen2.5:14b",
    temperature=0.0)

# 发送问题并获取回答
# output = llm.invoke("请你自我介绍一下自己！")

# 打印结果
# print(output)


from langchain_core.prompts import ChatPromptTemplate

# 这里我们要求模型对给定文本进行中文翻译
prompt = """请你将由三个反引号分割的文本翻译成英文！\
text: ```{text}```
"""

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
prompt.format(text=text)

from langchain.prompts.chat import ChatPromptTemplate

template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
messages = chat_prompt.format_messages(input_language="中文", output_language="英文", text=text)
output = llm.invoke(messages)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
output_parser.invoke(output)

chain = chat_prompt | llm | output_parser
text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
output = chain.invoke({"input_language": "英文", "output_language": "中文", "text": text})

print(output)
