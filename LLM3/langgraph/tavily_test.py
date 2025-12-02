import os
from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()
os.environ.get('TAVILY_API_KEY')