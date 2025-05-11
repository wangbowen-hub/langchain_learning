from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from config import *
import pickle
llm=ChatOpenAI(model=MODEL_NAME)
def load_embedding(store_name, embedding, suffix, path):
    vector_store=Chroma(
        persist_directory=f"{path}/chroma_{store_name}_{suffix}",
        embedding_function=embedding,
    )
    return vector_store

db_embedding_chunks_small=load_embedding(
    store_name="openAIEmbeddings",
    embedding=OpenAIEmbeddings(),
    suffix="chunks_small",
    path=DB_DIR,
)
db_embedding_chunks_medium = load_embedding(
    store_name="openAIEmbeddings",
    embedding=OpenAIEmbeddings(),
    suffix="chunks_medium",
    path=DB_DIR,
)

def load_pickle(prefix, suffix, path):
    """从pickle文件加载数据"""
    with open(f"{path}/{prefix}_{suffix}.pkl", "rb") as file:
        return pickle.load(file)
# 加载文档块
db_docs_chunks_small = load_pickle(
    prefix="docs_pickle", suffix="small_chunks", path=DB_DIR
)
db_docs_chunks_medium = load_pickle(
    prefix="docs_pickle", suffix="medium_chunks", path=DB_DIR
)

# 加载文件名
file_names = load_pickle(prefix="file", suffix="names", path=DB_DIR)
