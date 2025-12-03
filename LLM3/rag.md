# RAG (검색 증강 생성) 완벽 가이드

## 1. RAG란 무엇인가요?

RAG는 **Retrieval-Augmented Generation**의 약자로, 외부 문서나 데이터를 검색하여 LLM(대규모 언어 모델)의 답변 품질을 향상시키는 기술입니다.

### 🎯 RAG의 핵심 개념

기존 LLM의 문제점:
- ❌ 학습 데이터의 시간이 지나 오래된 정보 제공
- ❌ 환각(Hallucination): 존재하지 않는 정보를 만들어냄
- ❌ 특정 도메인의 전문 지식 부족

RAG의 해결책:
- ✅ 최신 정보 반영 가능
- ✅ 실제 문서 기반 답변으로 정확성 향상
- ✅ 출처 명시 가능 (어느 문서에서 정보를 가져왔는지 추적)
- ✅ 특정 분야의 문서만 사용하여 전문적 답변 제공

---

## 2. RAG의 작동 원리 (5단계)

```
사용자 질문
    ↓
[1단계] 질문을 벡터로 변환 (임베딩)
    ↓
[2단계] 벡터 데이터베이스에서 유사한 문서 검색 (리트리버)
    ↓
[3단계] 검색된 문서를 프롬프트에 포함 (컨텍스트 구성)
    ↓
[4단계] LLM이 컨텍스트 + 질문을 바탕으로 답변 생성
    ↓
[5단계] 최종 답변 및 출처 반환
```

---

## 3. RAG의 핵심 구성 요소

### 📄 문서 로딩 (Document Loading)
외부 소스에서 문서를 읽어들이는 단계입니다.

```python
# 지원되는 문서 형식
- 텍스트 파일 (.txt)     → TextLoader
- PDF 파일 (.pdf)        → PyPDFLoader
- 웹 페이지             → WebBaseLoader
- 디렉터리의 모든 파일  → DirectoryLoader
```

**예시:**
```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader('document.txt', encoding='utf-8')
documents = loader.load()
```

### ✂️ 청킹 (Chunking)
긴 문서를 작은 조각으로 분할하는 단계입니다.

**왜 청킹이 필요한가?**
- LLM의 컨텍스트 제한 (보통 4K~128K 토큰)
- 관련 없는 정보를 포함시키지 않기 위해
- 검색 정확도 향상

**두 가지 주요 방식:**

1. **CharacterTextSplitter** (단순)
   - 특정 문자(예: '\n')를 기준으로 분할
   - 빠르지만 문서 구조를 무시할 수 있음

2. **RecursiveCharacterTextSplitter** (권장) ⭐
   - 여러 구분자를 계층적으로 시도: ['\n\n', '\n', '.', ' ', '']
   - 문서의 의미 단위 유지
   - 더 나은 품질의 청크 생성

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 각 청크의 최대 크기
    chunk_overlap=200,    # 청크 간 겹침 (문맥 연속성 유지)
    separators=['\n\n', '\n', '.']  # 분할 기준
)
doc_splits = splitter.split_documents(documents)
```

### 🔢 임베딩 (Embedding)
텍스트를 고정 길이의 숫자 벡터로 변환하는 단계입니다.

**핵심 개념:**
- 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 배치됨
- 예: "고양이"와 "강아지"는 "자동차"보다 더 가까움

**임베딩 차원별 특징:**
| 차원 | 특징 | 추천 모델 |
|------|------|---------|
| 낮음 (384) | 빠름, 저장효율 | all-MiniLM-L6 |
| 중간 (1024) | 균형잡힌 선택 | BGE-M3, E5 |
| 높음 (1536+) | 높은 표현력 | OpenAI text-embedding-3 |

```python
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector = embeddings.embed_query('안녕하세요')  # → 1536 차원의 벡터
```

### 🗄️ 벡터 데이터베이스 (VectorDB)
임베딩된 벡터를 저장하고 검색하는 데이터베이스입니다.

**주요 VectorDB 솔루션:**

| 솔루션 | 특징 | 사용 시기 |
|--------|------|----------|
| **ChromaDB** | 로컬 개발에 적합, 파이썬 네이티브, 간편한 설치 | 프로토타이핑, 로컬 개발 |
| **Pinecone** | 완전 관리형 클라우드, 대규모 프로덕션 | 프로덕션 환경, 대규모 데이터 |
| **Weaviate** | 그래프 기반, 하이브리드 검색 지원 | 복잡한 검색 요구 |
| **FAISS** | Facebook 개발, 고성능 | 매우 큰 규모 데이터 |
| **Milvus** | 분산 환경 지원, 오픈소스 | 분산 시스템 |

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name='my_collection',
    embedding=OpenAIEmbeddings(model='text-embedding-3-small')
)
```

### 🔍 리트리버 (Retriever)
사용자 질문과 유사한 문서를 찾아내는 컴포넌트입니다.

```python
# 기본 유사도 검색
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}  # 상위 3개 문서 반환
)

# MMR 검색 (다양성 고려)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={
        'k': 3,           # 최종 반환 개수
        'fetch_k': 6,     # 먼저 검색할 후보 개수
        'lambda_mult': 0.5  # 0 = 다양성, 1 = 관련성
    }
)
```

### 💬 프롬프트 템플릿 (Prompt Template)
검색된 문서와 질문을 조합하여 LLM에 전달할 프롬프트를 구성합니다.

**효과적인 RAG 프롬프트:**
```python
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 제공된 문맥을 바탕으로 질문에 답변하는 AI입니다.

## 규칙
1. 제공된 문맥 내의 정보만 사용하세요.
2. 문맥에 없는 정보는 추측하지 말고 "제공된 문서에서 찾을 수 없습니다"라고 답하세요.
3. 답변은 한국어로 명확하고 간결하게 작성하세요.
'''),
    ('human', '''## 참조 문맥
{context}

## 질문
{question}

## 답변''')
])
```

### 🔗 RAG 체인 (LCEL 방식)
모든 구성 요소를 연결하여 파이프라인을 구성합니다.

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# 기본 RAG 체인
rag_chain = (
    {
        'context': retriever | format_docs,
        'question': RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 사용
answer = rag_chain.invoke("RAG란 무엇인가요?")
```

---

## 4. 고급 RAG 패턴

### 1️⃣ Query Transformation (질문 변환)
사용자의 질문을 검색에 더 적합한 형태로 변환하여 검색 정확도를 향상시킵니다.

```python
rewrite_prompt = ChatPromptTemplate.from_template('''
다음 질문을 검색에 더 적합한 형태로 변환해주세요.
키워드 중심으로 명확하게 바꿔주세요.

원본 질문: {question}
변환된 검색어:
''')

rewrite_chain = rewrite_prompt | llm | StrOutputParser()
transformed = rewrite_chain.invoke({'question': '라그 뭔데?'})
# 결과: "RAG 정의 설명"
```

**사용 시나리오:**
- 사용자 질문이 모호할 때
- 검색 결과가 부정확할 때

### 2️⃣ Multi-Query (다중 질의)
동일한 질문을 여러 관점에서 다시 표현하여 검색 범위를 확대합니다.

```python
multi_query_prompt = ChatPromptTemplate.from_template('''
다음 질문에 대해 3가지 다른 관점의 검색 쿼리를 생성하세요.

원본 질문: {question}
다른 관점의 쿼리들:
''')

# 예시
# 원본: "RAG가 뭔지 알려줘"
# 생성된 쿼리들:
#  - RAG 정의 및 개념
#  - 검색 증강 생성의 작동 원리
#  - RAG 기술의 장점과 응용
```

**장점:**
- 한 가지 관점으로 놓칠 수 있는 정보 수집
- 검색 결과의 다양성 보장

### 3️⃣ Self-RAG (자기 보정 RAG)
검색된 문서의 관련성을 평가하여 관련 없는 문서를 필터링합니다.

```python
check_prompt = ChatPromptTemplate.from_template('''
다음 문서가 질문에 관련이 있는지 평가하세요.
'yes' 또는 'no'로만 답변하세요.

문서: {document}
질문: {question}
관련성:
''')

# 관련 있는 문서만 필터링
relevant_docs = []
for doc in retrieved_docs:
    result = check_chain.invoke({'document': doc.page_content, 'question': question})
    if 'yes' in result.lower():
        relevant_docs.append(doc)
```

**효과:**
- 답변 품질 향상
- 불필요한 정보 제거
- 답변 생성 시간 단축

### 4️⃣ Contextual Compression (문맥 압축)
검색된 전체 문서에서 질문과 관련된 부분만 추출합니다.

```python
compress_prompt = ChatPromptTemplate.from_template('''
다음 문서에서 질문과 관련된 부분만 추출하세요.
관련 없는 부분은 제외하고, 관련 있는 내용만 출력하세요.

문서: {document}
질문: {question}

관련 내용:
''')

# 압축된 컨텍스트만 LLM에 전달
compressed_context = compress_chain.invoke({'document': doc, 'question': question})
```

**이점:**
- 프롬프트 길이 감소 (비용 절감)
- LLM이 더 집중된 정보로 답변 생성
- 답변 생성 속도 향상

### 5️⃣ Fusion Retrieval (융합 검색)
키워드 기반 검색(BM25)과 의미 기반 검색(벡터)을 결합합니다.

```python
from langchain_community.retrievers import BM25Retriever

# BM25 리트리버 (키워드 기반)
bm25_retriever = BM25Retriever.from_documents(doc_splits)
bm25_retriever.k = 3

# 벡터 리트리버 (의미 기반)
vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# 두 검색 결과 점수 합산
fusion_scores = {}

# 벡터 검색 결과
for rank, doc in enumerate(vector_retriever.invoke(question)):
    doc_key = doc.page_content[:50]
    fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + 1 / (60 + rank)

# BM25 검색 결과
for rank, doc in enumerate(bm25_retriever.invoke(question)):
    doc_key = doc.page_content[:50]
    fusion_scores[doc_key] = fusion_scores.get(doc_key, 0) + 1 / (60 + rank)

# 점수 순으로 정렬
best_docs = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
```

**활용 케이스:**
- 정확한 키워드 매칭이 필요한 경우
- 의미 기반 검색만으로 부족한 경우
- 최대한 포괄적인 검색이 필요한 경우

---

## 5. 실제 구현 예시

### 기본 RAG 시스템

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1단계: 문서 로드
loader = DirectoryLoader(
    './documents',
    glob='**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = loader.load()

# 2단계: 청킹
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
doc_splits = splitter.split_documents(documents)

# 3단계: 임베딩 및 VectorDB
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    persist_directory='./chroma_db'
)

# 4단계: 리트리버
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# 5단계: 프롬프트 템플릿
rag_prompt = ChatPromptTemplate.from_messages([
    ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요.'),
    ('human', '문맥:\n{context}\n\n질문: {question}\n\n답변:')
])

# 6단계: LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# 7단계: 체인 구성
def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 8단계: 사용
answer = rag_chain.invoke('RAG란 무엇인가요?')
print(answer)
```

### RAG 래퍼 클래스

```python
class SimpleRAGSystem:
    '''간단한 RAG 시스템 래퍼'''
    
    def __init__(self, vectorstore, llm, retriever_k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.retriever = vectorstore.as_retriever(search_kwargs={'k': retriever_k})
    
    def ask(self, question: str) -> str:
        '''질문에만 답변'''
        prompt = ChatPromptTemplate.from_messages([
            ('system', '제공된 문맥을 바탕으로 답변하세요.'),
            ('human', '문맥:\n{context}\n\n질문: {question}\n\n답변:')
        ])
        
        chain = (
            {'context': self.retriever | self._format_docs, 'question': RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(question)
    
    def ask_with_sources(self, question: str) -> dict:
        '''질문에 답변 + 출처 반환'''
        answer = self.ask(question)
        sources = self.retriever.invoke(question)
        return {
            'answer': answer,
            'sources': [doc.metadata.get('source', 'unknown') for doc in sources]
        }
    
    @staticmethod
    def _format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

# 사용
rag_system = SimpleRAGSystem(vectorstore, llm)
result = rag_system.ask_with_sources('VectorDB의 종류를 알려주세요')
print(f"답변: {result['answer']}")
print(f"출처: {result['sources']}")
```

---

## 6. RAG 성능 향상 팁

### ✅ 문서 준비 단계
- **고품질 문서 수집**: 정확하고 신뢰할 수 있는 문서 사용
- **메타데이터 추가**: source, author, date 등 정보 포함
- **문서 정제**: 불필요한 공백, 특수문자 제거

### ✅ 청킹 최적화
- **적절한 크기**: 문서 유형에 따라 300-1000 단어 권장
- **겹침 설정**: 20-30% 겹침으로 문맥 연속성 유지
- **구조 보존**: RecursiveCharacterTextSplitter 사용

### ✅ 임베딩 선택
- **한국어 최적화**: BGE-M3, KoSimCSE 추천
- **성능 vs 비용**: 개발 단계에서는 text-embedding-3-small 추천

### ✅ 검색 전략
- **다양한 검색**: MMR, Fusion Retrieval 활용
- **필터링**: 메타데이터 필터링으로 검색 범위 축소
- **재순위화**: 검색 결과를 재정렬하여 정확도 향상

### ✅ 프롬프트 엔지니어링
- **역할 정의**: "당신은 기술 문서 Q&A 시스템입니다"
- **명확한 규칙**: 문맥만 사용, 모르면 답변하지 말 것
- **출력 형식**: 구조화된 형태 요청 (목록, 테이블 등)

### ✅ 모니터링 및 개선
- **평가 지표**: 정확도, 관련성, 응답 시간 측정
- **피드백 수집**: 사용자 피드백을 통한 개선
- **비용 관리**: 토큰 사용량 모니터링

---

## 7. 자주 하는 질문 (FAQ)

**Q: RAG와 Fine-tuning의 차이는?**
- **RAG**: 외부 문서 기반, 비용 저렴, 빠른 업데이트 가능
- **Fine-tuning**: 모델 자체를 학습, 높은 비용, 업데이트 느림

**Q: 몇 개의 문서를 검색해야 하나?**
- 보통 3-5개가 최적 (너무 많으면 컨텍스트 혼동 가능)
- MMR 검색으로 다양성 보장

**Q: VectorDB를 로컬에서 테스트하려면?**
- ChromaDB 추천: `pip install chromadb`
- 프로덕션은 Pinecone, Weaviate 고려

**Q: 응답 시간이 느려요. 어떻게 개선할까?**
- 문서 수 감소
- 청크 크기 조정
- Contextual Compression 적용
- 배치 처리 고려

**Q: 한국어 임베딩 모델은?**
- **BGE-M3**: 다국어 지원, 뛰어난 성능
- **KoSimCSE**: 한국어 전용, 가볍고 빠름
- **OpenAI text-embedding-3-small**: API 기반, 한국어 가능

---

## 8. 학습 순서

1. **기초 이해**: RAG의 개념과 작동 원리 학습
2. **기본 구현**: `rag_basic.py` 따라하기
3. **고급 패턴**: Query Transformation, Multi-Query 학습
4. **최적화**: Fusion Retrieval, Self-RAG 적용
5. **실무 프로젝트**: 자신의 데이터로 RAG 시스템 구축

---

## 📚 참고 파일

| 파일명 | 설명 |
|--------|------|
| `rag_1.1.py` | 문서로딩과 청킹 기초 |
| `rag_basic.py` | 기본 RAG 시스템 구현 |
| `rag_3.1.py` | 프롬프트 템플릿과 RAG 체인 |
| `rag_3.2.py` | LCEL을 이용한 RAG 구현 |
| `reg_1.2.py` | 청킹 상세 가이드 |
| `reg_2.2.py` | 임베딩과 VectorDB |
| `reg_advanced1_2.py` | Query Transformation, Multi-Query |
| `rag_advanced3.py` | Self-RAG (자기 보정) |
| `rag_advanced4.py` | Contextual Compression (문맥 압축) |
| `rag_advanced5.py` | Fusion Retrieval (융합 검색) |
| `SimpleRAGSystem.py` | RAG 래퍼 클래스 |

---

**마지막 업데이트**: 2025년 12월 2일
