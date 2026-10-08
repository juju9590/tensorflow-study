# transformer 논문을 벡터DB로 불러와서
# 요약, 인용 등등 할 수 있는 챗봇으로.

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

path ='./_data/'
# pdf_loader = PyPDFLoader(path + "attention_is_all_your_needs.pdf")
# pdf_docs = pdf_loader.load() # PDF 불러오기
pdf_files = glob(os.path.join(path, '*.pdf'))
# print(pdf_files) #['./_data\\attention_is_all_your_needs.pdf']

# #01. PDF 불러오기
# data = []
# for pypdf_file in pdf_files:
#     loader = PyPDFLoader(pypdf_file, ) # encoding='utf-8')
#     data += loader.load()

# # print(data[0])
# # print(len(data)) #2
# # print(data[0].page_content)

# char_count = [len(doc.page_content) for doc in data]
# # print(char_count) #[2857, 4251, 1823, 2491, 3181, 3450, 3300, 3178, 2973, 3111, 3216, 3220, 812, 815, 818]

# #02. 청킹(자른다)
# Text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50, 
#     separators=["\n\n", "\n", " ", ""],
# )

# pdf_texts = Text_splitter.split_documents(data)
# # print("생성된 텍스트 청크 수 : ", len(pdf_texts)) #161 

#03. 임베팅
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key,   
    base_url=base_url,  
)

# # sample_text = 'attention is all you needs?'
# # vector = embeddings.embed_query(sample_text)
# # # print(vector)
# # # print(len(vector)) # 1536 차원

# #04.저장하기
# DB_path ='./_db/Chroma_19/'
# vector_store = Chroma.from_documents(
#     documents=pdf_texts,
#     embedding=embeddings,
#     persist_directory=DB_path, #지속가능한 디렉토리
#     collection_name='chroma19',
# )



#05. 불러오기
DB_path ='./_db/Chroma_19/'

vector_store = Chroma(
    embedding_function=embeddings, 
    persist_directory=DB_path, 
    collection_name='chroma19',
)

# DB 저장 잘 불러오는지 테스트 
# query = "What is the architecture of this model?"
# result = vector_store.similarity_search(query)
# print(f"검색 결과의 길이: {len(result)}") 
# # 검색 결과의 길이: 4

# print("DB에 저장된 문서 개수:", vector_store._collection.count())
# # DB에 저장된 문서 개수: 161
# print("컬렉션 이름:", vector_store._collection.name)
# # 컬렉션 이름: chroma19
# print("DB 저장 경로:", DB_path) #DB 저장 경로: ./_db/Chroma_19/

#06. 검색기 Retrievers = vector_store 검색
retriever = vector_store.as_retriever(search_kwargs={"k":3}) #검색된 관련 문서 수 : 3

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
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
"주어진 정보로는 답변할 수 없습니다." 라고 말씀해 주세요

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
demo =gr.ChatInterface(fn=answer_invoke, title='토리군')

### Gradio 실행
demo.launch() # Running on local URL:




