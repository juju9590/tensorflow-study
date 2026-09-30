# LCEL = Langchain Expression Language
# 파이프 라인으로 연결해 놓은 언어 |||
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 설명해주세요")

model = ChatOpenAI(
    model_name='gpt-5.6-terra', 
    temperature=0,                  
    api_key=api_key,   # 인증키
    base_url=base_url,  # 주소
)

chain = prompt | model 
# | : or 연산자 이지만 오버라이딩 = 재정의  ==> 랭체인에서는 연결로 쓰임
# 이 프로프트는 이 모델로 해결을 해 

input ={"topic" : "양자컴퓨터 학습 원리"}
response = chain.invoke(input)
# chain과 input을 연결한 것을 response라 한다


print(response.content)



