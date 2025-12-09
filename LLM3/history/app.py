# RAG  + Radis Chat Memory + Multi Session
from config import REDIS_URL
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.chains import
from langchain.chains.combine_documents import create_stuff_documents_chain
from vectorstore import load_document_and_create_vectorstore

from dotenv import load_dotenv
load_dotenv()
