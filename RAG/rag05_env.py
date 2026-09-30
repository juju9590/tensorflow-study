
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()
# .env에 인증키를 넣었을때 가져오려면 꼭 필요
# 작업그룹의 .env를 불러옴
# 혹시 다른 그룹의 .env 불러오려면 경로를 잡아 놔야 한다
# vs code 에서는 load_dotenv() 하지 않아도 자동인식 되지만 
# 리눅스나 다른 환경에서는 꼭 위 2줄을 설정해줘야 한다


llm = ChatOpenAI(
    model_name='gpt-5.6-terra',     # 모델의 가중치 저장
    temperature=0,                  # 창작하지 않고 쓰겠다
    # openai_api_key=openai_api_key,
)

response = llm.invoke("오늘 서울 날씨에 어울리는 옷차림은?")

print(response.content)
