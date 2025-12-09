from langchain_redis import RedisVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from config import REDIS_URL, INDEX_NAME
import os

def load_document_and_create_vectorstore():
    # 문서로드
    loader = TextLoader('documents/sample.txt', encoding='utf-8')
    docs = loader.load()
    # 청크단위로 분리
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstore = RedisVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        redis_url=REDIS_URL
    )
    return vectorstore