# rag15를 가지고 프롬프트를 막 만져서 만들어보세요

# 13-1 copy

import os

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain #많이 씀
from langchain_classic.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

#03. 임베딩 준비 
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)

DB_path ='./_db/Chroma_12/'

# 불러오기
vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory=DB_path, 
    collection_name='chroma12',
)

# 검색기 Retrievers = vector_store 검색
retriever = vector_store.as_retriever(search_kwargs={"k":2}) #검색된 관련 문서 수 : 3

### 모델 연결
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model= 'gpt-5.6-terra', 
    temperature=0, 
    max_tokens=1000, 
    api_key=api_key,
    base_url=base_url,
)

prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 
컨텍스트에 없는 내용은 짤고 간결하지만 유머러스 하게 대답해주세요
처음 방문한 사람이 친근함을 느낄 수 있게 이모티콘도 사용해 주세요.

컨텍스트 : {context}
질문 : {input}
답변 : 
""")

# 체인 만들기 (둘은 "세트"다)
docu_chain = create_stuff_documents_chain(model, prompt) # ===> prompt | model
rag_chain = create_retrieval_chain(retriever, docu_chain) # ===> 검색 | docu_chain

### Gradio 챗봇 
import gradio as gr

def answer_invoke(message, history) : 
    response = rag_chain.invoke({"input" : message})
    return response['answer']

### Gradio  인터페이스 만들자
demo =gr.ChatInterface(fn=answer_invoke, title='tori_bot')

### Gradio 실행
demo.launch() # Running on local URL:
# demo.launch(share=True) # Running on public URL

