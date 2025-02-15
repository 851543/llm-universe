from langchain_community.embeddings import OllamaEmbeddings

from langchain_community.vectorstores.chroma import Chroma


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())    # read local .env file

embedding = OllamaEmbeddings(
    model="qwen2.5:14b",
    num_gpu=1,  # 使用GPU模式运行
    show_progress=True
)
# 向量数据库持久化路径
persist_directory = 'data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")


question = "什么是prompt engineering?"
docs = vectordb.similarity_search(question,k=3)
print(f"检索到的内容数：{len(docs)}")


for i, doc in enumerate(docs):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")



