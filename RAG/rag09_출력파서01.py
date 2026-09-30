# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser
# parser 뜻 : 분석하다, 다듬어주다

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
base_url='https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template("{topic}에 대해 100자 이내로 쉽게 설명해주세요")

model = ChatOpenAI(
    model_name='gpt-5.6-terra', 
    temperature=0,                  
    api_key=api_key,   
    base_url=base_url, 
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input ={"topic" : "양자컴퓨터 학습 원리"}
response = chain.invoke(input)


print(response)
# output_parser 연결하면 .content 없이도 결과를 빼준다.
#     print(response.content)
#           ^^^^^^^^^^^^^^^^
# AttributeError: 'TextAccessor' object has no attribute 'content'



