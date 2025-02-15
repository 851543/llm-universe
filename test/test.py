import os

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = 'data_base/knowledge_db'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)

from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:

    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file_path))

# 下载文件并存储到text
texts = []

for loader in loaders: texts.extend(loader.load())

print(texts)