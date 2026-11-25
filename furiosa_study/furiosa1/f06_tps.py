#copy f05

from langchain_openai import ChatOpenAI
import time

max_tokens = 1000

#tps = token per second
llm = ChatOpenAI(
    base_url = "http://127.0.0.1:8000/v1",    
    api_key="EMPTY",
    max_tokens=max_tokens,
    temperature=5,
    top_p=0.9, #확률 기반 토큰 선택 설정(Nucleus Sampling)
)

# temperature: 토큰 간 확률 분포를 얼마나 뾰족하거나 평평하게 만들지 조절
# top_p: 그 확률 분포에서 후보를 어디까지 허용할지 결정

start = time.perf_counter()  # PC의 시간
# start = time.time() 운영체제 기준 시간
response = llm.invoke("npu와 gpu의 차이점을 설명해줘")
end = time.perf_counter()
elapsed = round(end-start, 2)

# print(response.content)
print("추론시간(tps) : ", elapsed,'초')
print("추론토큰 : ", max_tokens)
print("평균 처리량 : ", round(max_tokens/elapsed,2),"tokens/s")


# 추론시간(tps) :  7.15 초
# 추론토큰 :  1000
# 평균 처리량 :  139.86 tokens/s

# 추론시간(tps) :  8.93 초
# 추론토큰 :  1000
# 평균 처리량 :  111.98 tokens/s
