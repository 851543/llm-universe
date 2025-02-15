from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

embedding = OllamaEmbeddings(
    model="qwen2.5:14b",
    num_gpu=1,  # 使用GPU模式运行
    show_progress=True
)

# 定义持久化路径
persist_directory = 'chroma'

from langchain_community.document_loaders import TextLoader

"""
file_path：要加载的文件的路径。
encoding：要使用的文件编码。如果“无”，将加载文件使用默认的系统编码。
autodetect_encoding：是否尝试自动检测文件编码如果指定的编码失败。
"""
loader = TextLoader("./index", encoding='utf-8', autodetect_encoding=True)
loader = loader.load()

print(loader)

# 将新文档添加到向量数据库
vectordb = Chroma.from_documents(
    documents=loader,
    embedding=embedding,
    persist_directory=persist_directory
)

# 验证添加结果
# from langchain_community.embeddings import OllamaEmbeddings
#
# from langchain_community.vectorstores.chroma import Chroma
#
# embedding = OllamaEmbeddings(
#     model="qwen2.5:14b",
#     num_gpu=1,  # 使用GPU模式运行
#     show_progress=True
# )
# # 向量数据库持久化路径
# persist_directory = 'chroma'
#
# # 加载数据库
# vectordb = Chroma(
#     persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
#     embedding_function=embedding
# )
#
print(f"向量库中存储的数量：{vectordb._collection.count()}")
#
#
# question = "朱斌博是谁?"
# docs = vectordb.similarity_search(question)
# print(docs)



