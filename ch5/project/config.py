import os
# 模型名称
MODEL_NAME = "gpt-3.5-turbo" 
current_file_path = os.path.abspath(__file__)
# 获取当前脚本所在目录的路径
current_directory = os.path.dirname(current_file_path)
# 存储嵌入向量和 Langchain 文档的目录
DB_DIR = os.path.join(current_directory, "database_store")