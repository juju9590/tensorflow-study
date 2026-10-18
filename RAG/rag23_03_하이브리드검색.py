#pip install rank_bm25

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

### BM25 검색 ####
#pip install rank_bm25

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = "삼성전자의 반도체 사업"

# ###### BM25 검색 ########
# # BM25(best Match 25) :  키워드 기반 검색 

from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3  # 문장 3개 뽑아준다

# bm25_docs = bm25_retriever.invoke(query)

##########################################################
### 백터 검색 
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #많이 씀
from langchain_classic.chains import create_retrieval_chain

from glob import glob

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

#03. 임베딩
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

#03. 임베딩
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)
#05. 불러오기
DB_path ='./_db/Chroma_12/'

vector_store = Chroma(
    embedding_function=embeddings, 
    # documents=texts,
    persist_directory=DB_path, 
    collection_name='chroma12',
)

# 검색기
vector_retriever = vector_store.as_retriever(  
    search_kwargs = {"k":3}
)
# vector_docs = vector_retriever.invoke(query)

###### 하이브리드 검색 ######################################
from langchain_classic.retrievers import EnsembleRetriever 

hybird_retriever = EnsembleRetriever(
    retrievers=[
        bm25_retriever,
        vector_retriever,
    ],
    weights=[0.5, 0.5]

)

hybird_docs = hybird_retriever.invoke(query)

# print("=============== 앙상블 하이브리드 검색 결과 ====================")
# 하이브리드 리트리버 : 성격이 다른 리트리버로 비교시 하이브리드
# 앙상블 리드티버 : 성향이 비슷한 리트리버 모아서 했을때 더 넓은 범위를 말함 

for doc in hybird_docs:                 
    print(doc.page_content)


















