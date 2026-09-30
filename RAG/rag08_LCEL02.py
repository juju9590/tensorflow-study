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

prompt = PromptTemplate.from_template("{topic}에 대해 쉽게 {how} 500자 아내로 설명해주세요")

model = ChatOpenAI(
    # model_name='gpt-5.6-terra', 
    model_name='claude-fable-5',     
    temperature=0,                  
    api_key=api_key,   
    base_url=base_url,  
)

chain = prompt | model 

# input ={"topic" : "양자컴퓨터 학습 원리", "how" : "유치원생도 이해하기 쉽게"}
input ={"topic" : "고양이의 습성", "how" : "초보 집사에게 필요한 내용"}
# input ={"topic" : "사업 아이템 아이디어", "how" : "지금 사업을 시작하는 사장님이 접근하는 방법"}
# input ={"topic" : "AI관련 유망업종 or 직종", "how" : "tensorflow, rag, 에이전트 교육받은 사람 대상"}



response = chain.invoke(input)


print(response.content)







