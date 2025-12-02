import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.environ.get('OPENAI_API_KEY'):
    raise ValueError('key check...')

class CGRAState(TypedDict):
    question : str
    documents : List[Document]
    web_search_needed : str # 웹검색 여부(yes/no)
    context : str
    answer : str
    grade_results : List[str] # 각 문서의 평가결과

# Step 1 문서
path = 'C:/Users/playdata2/Desktop/LLM/LLM3/advenced/sample_docs'
loader = DirectoryLoader(
    path = path,
    glob = '**/*.txt',
    loader_cls = TextLoader,
    loader_kwargs = {'encoding': 'utf-8'},
)
docs = loader.load()

# Step 2 텍스트 분할 (청크)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs)

# Step 3 임베딩 및 VectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name='crag_collection',
    embedding=OpenAIEmbeddings(model='text-emb')
)

# Step 4 리트리버 설정
retriever = vectorstore.as_retriever(search_kwars={'k':3})

print(f'{len(doc_splits)}개 청크로 VectorDB 구축 완료')

# 문서 관련성 평가를 위한 Grader 정의
from pydantic import BaseModel, Field
class GradeDocuments(BaseModel):
    '''문서 관련성 평가 결과를 위한 pydantic 모델'''
    binary_score:str = Field(description="문서가 질문과 관련이 있으면 'yes', 없으면 'no'")

# llm
grader_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
structured_grader = grader_llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 검색된 문서가 사용자의 질문에 답변하는데 관련이 있는지 평가하는 전문가 입니다.
     
     평가기준:
     - 문서가 질문의 키워드나 의미와 연관되어 있다면 '관련있음'으로 평가
     - 답변에 도움이 될 가능성이 조금이라도 있다면 '관련있음'
     - 완전히 무관한 내용이면 '관련없음'

     엄격하게 평가하지 말고, 약간의 연관성이라도 있으면 'yes'를 반환하세요
     '''),
     ('human', '''질문:{question}
      
      문서내용:
      {document}

      이 문서가 질문과 관련이 있습니까? 'yes' 또는 'no'로만 답하세요
      ''')
])

document_grader = grade_prompt | structured_grader

def retrieve_node(state: CGRAState) -> dict:
    '''내부 문서 검색 노드'''
    question = state['question']
    documents = retriever.invoke(question)
    return {
        'documents': documents,
        'question': question
    }


def grade_documents_node(state:CGRAState) -> dict:
    '''문서관련성 평가 노드
    검색된 문서의 관련성 여부를 llm 평가 
    관련없으면 웹 검색 플래그를 활성
    '''
    question = state['question']
    documents = state['documents']
    filtered_docs, grade_results = [],[]
    for i, doc in enumerate(documents,1):
        # 각 문서의 관련성 평가
        score = document_grader.invoke({
            'question' : question,
            'document' : doc.page_content
        })
        grade = score.binary_score.lower()
        if grade == 'yes':
            filtered_docs.append(doc)
            grade_results.append("relevant")
        else:
            grade_results.append("not_relevant")
     # 관련 문서가 없으면 웹 검색 필요
    if len(filtered_docs) == 0:
        web_search_needed = "Yes"
        print("   관련 문서 없음 → 웹 검색 필요!")
    else:
        web_search_needed = "No"
        print(f"  {len(filtered_docs)}개 관련 문서 확보!")
    
    return {
        "filtered_documents": filtered_docs,
        "web_search_needed": web_search_needed,
        "grade_results": grade_results
    }

def web_search_node(state: CGRAState) -> dict:
    """
    웹 검색 노드 (시뮬레이션)
    
    실제 환경에서는 Tavily API나 다른 검색 API를 사용합니다.
    여기서는 학습 목적으로 시뮬레이션합니다.
    """
    print("\n   [WEB SEARCH 노드] 외부 웹 검색 수행 중...")
    
    question = state["question"]
    
    # 웹 검색 시뮬레이션 (실제로는 Tavily API 등 사용)
    # 실제 구현 예시:
    # from langchain_community.tools.tavily_search import TavilySearchResults
    # web_search = TavilySearchResults(k=3)
    # web_results = web_search.invoke({"query": question})
    
    # 시뮬레이션된 웹 검색 결과
    simulated_web_results = f"""
    [웹 검색 결과 - 시뮬레이션]
    
    질문 '{question}'에 대한 웹 검색 결과:
    
    1. LLM(Large Language Model) 관련 최신 정보:
       - LLM은 자연어 처리에서 혁신적인 발전을 이루고 있습니다.
       - OpenAI, Anthropic, Google 등이 주요 제공자입니다.
       - RAG, Fine-tuning, Prompt Engineering이 주요 활용 기법입니다.
    
    2. AI 에이전트 트렌드:
       - 자율적인 AI 에이전트가 주목받고 있습니다.
       - LangGraph, AutoGPT 등이 대표적인 프레임워크입니다.
       - 멀티 에이전트 시스템이 복잡한 작업을 수행합니다.
    
    출처: 시뮬레이션된 웹 검색 (실제 환경에서는 Tavily API 사용)
    """
    
    # 웹 검색 결과를 Document 형태로 변환
    web_doc = Document(
        page_content=simulated_web_results,
        metadata={"source": "web_search", "type": "external"}
    )
    
    # 기존 필터링된 문서에 웹 검색 결과 추가
    filtered_docs = state.get("filtered_documents", [])
    filtered_docs.append(web_doc)
    
    print("   웹 검색 완료! 결과가 문서에 추가됨")
    
    return {
        "filtered_documents": filtered_docs
    }


def generate_node(state: CGRAState) -> dict:
    """
    답변 생성 노드
    필터링된 문서(내부 문서 + 웹 검색 결과)를 바탕으로 답변을 생성합니다.
    """
    print("\n   💬 [GENERATE 노드] 답변 생성 중...")
    
    question = state["question"]
    filtered_documents = state["filtered_documents"]
    
    # 컨텍스트 구성
    context = "\n\n---\n\n".join([doc.page_content for doc in filtered_documents])
    
    # 답변 생성 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 제공된 문맥을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

규칙:
1. 제공된 문맥 내의 정보를 우선적으로 사용하세요.
2. 답변은 한국어로 명확하고 구조화되게 작성하세요.
3. 웹 검색 결과가 포함된 경우, 해당 정보도 적절히 활용하세요.
4. 확실하지 않은 정보는 추측하지 마세요."""),
        ("human", """문맥(Context):
{context}

질문: {question}

답변:""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    print("   답변 생성 완료!")
    
    return {
        "context": context,
        "answer": answer
    }


# 조건부 엣지 함수 정의

def decide_to_generate(state: CGRAState) -> Literal["generate", "web_search"]:
    """
    문서 평가 결과에 따라 다음 단계를 결정합니다.
    
    - 관련 문서가 있으면 → generate (답변 생성)
    - 관련 문서가 없으면 → web_search (웹 검색)
    
    Returns:
        "generate" 또는 "web_search"
    """
    print("\n   [DECISION] 다음 단계 결정 중...")
    
    web_search_needed = state["web_search_needed"]
    
    if web_search_needed == "Yes":
        print("   결정: 웹 검색으로 이동")
        return "web_search"
    else:
        print("   결정: 답변 생성으로 이동")
        return "generate"


print("조건부 엣지 함수 정의 완료!")

print("\n CRAG StateGraph 구성 및 컴파일 중...")

# StateGraph 생성
workflow = StateGraph(CGRAState)

# 노드 추가
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("generate", generate_node)

# 엣지 추가
# START -> retrieve -> grade_documents
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")

# 조건부 엣지: grade_documents 이후 분기
# - 관련 문서 있음 → generate
# - 관련 문서 없음 → web_search
workflow.add_conditional_edges(
    "grade_documents",      # 시작 노드
    decide_to_generate,     # 조건 함수
    {
        "generate": "generate",      # "generate" 반환 시
        "web_search": "web_search"   # "web_search" 반환 시
    }
)

# web_search 이후 generate로 이동
workflow.add_edge("web_search", "generate")

# generate 이후 종료
workflow.add_edge("generate", END)

# 그래프 컴파일
app = workflow.compile()


# 테스트 시나리오
test_cases = [
    {
        "question": "LangGraph의 핵심 개념을 설명해주세요.",
        "expected": "내부 문서에서 답변 가능 → 웹 검색 불필요"
    },
    {
        "question": "CRAG 패턴의 장점은 무엇인가요?",
        "expected": "내부 문서에서 답변 가능 → 웹 검색 불필요"
    },
    {
        "question": "최신 GPT-5 모델의 특징은 무엇인가요?",
        "expected": "내부 문서에 없음 → 웹 검색 필요"
    }
]

for i, test in enumerate(test_cases, 1):
    print(f"\n{'━' * 70}")
    print(f" 테스트 {i}: {test['question']}")
    print(f"   예상 시나리오: {test['expected']}")
    print(f"{'━' * 70}")
    
    # 초기 상태
    initial_state = {
        "question": test["question"],
        "documents": [],
        "filtered_documents": [],
        "web_search_needed": "No",
        "context": "",
        "answer": "",
        "grade_results": []
    }
    
    # 그래프 실행
    print("\n CRAG 워크플로우 실행 중...")
    
    final_state = None
    for output in app.stream(initial_state):
        for node_name, node_output in output.items():
            print(f"   노드 '{node_name}' 실행 완료")
        final_state = output
    
    # 결과 출력
    if "generate" in final_state:
        answer = final_state["generate"]["answer"]
    else:
        answer = "답변을 생성할 수 없습니다."
    
    print(f"\n 최종 답변:\n{answer}")