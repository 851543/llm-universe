from langchain.docstore.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# 定义嵌入模型
embedding = OllamaEmbeddings(
    model="qwen2.5:14b",
    num_gpu=1,
    show_progress=True
)

# 定义持久化路径
persist_directory = 'chroma'

# 将文档添加到向量数据库
# vectordb = Chroma.from_documents(
#     documents=[personal_doc],
#     embedding=embedding,
#     persist_directory=persist_directory
# )

vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)


# 使用文本查询相似文档
results = vectordb.similarity_search(query="朱斌博")
for doc in results:
    print(f"匹配文档: {doc.page_content}")
