from langchain_community.llms import Ollama

# 这里我们将参数temperature设置为0.0，从而减少生成答案的随机性。
# 如果你想要每次得到不一样的有新意的答案，可以尝试调整该参数。
llm = Ollama(
    model="qwen2.5:14b",
    temperature=0.0)

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

embedding = OllamaEmbeddings(
    model="qwen2.5:14b",
    num_gpu=1,  # 使用GPU模式运行
    show_progress=True
)
# 向量数据库持久化路径
persist_directory = 'chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True)
question_1 = "党广俊是？"

result = qa_chain(question_1)
print(result["result"])
