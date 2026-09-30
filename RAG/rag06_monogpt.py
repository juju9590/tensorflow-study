
from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MONOROUTER_API_KEY"].strip()
#.strip(): 공백이나 띄어쓰기를 무시하고 읽어준다.
base_url='https://monogpt.kr/api/monorouter/v1'
# 인증키의 API키와 주소 넣어주어야 한다.


llm = ChatOpenAI(
    model_name='gpt-5.6-terra', 
    temperature=0,                  
    api_key=api_key,   # 인증키
    base_url=base_url,  # 주소
)

response = llm.invoke("대한민국 대표 판다 이름은?")
# 대한민국에서 가장 대표적으로 알려진 판다는 **푸바오**입니다.  
# 에버랜드에서 태어난 국내 최초 자연번식 자이언트판다로, 이름은 “행복을 주는 보물”이라는 뜻입니다

print(response.content)
