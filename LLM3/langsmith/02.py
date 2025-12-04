# 이론부분의 sample 코드에 대한 완전히 구현한 코드
import os
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Any,Dict,List
from dotenv import load_dotenv

# langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# langsmith 
from langsmith import Client
from langsmith.run_helpers import traceable

# 환경설정
load_dotenv()
def check_environment():
    '''환경변수 확인'''
    missing_keys = []
    if not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.getenv('LANGCHAIN_API_KEY'):
        missing_keys.append('LANGCHAIN_API_KEY')
    if missing_keys:
        print('필요한 API KEY가 없습니다')
        for key in missing_keys:
            print(f' ----------- {key}')