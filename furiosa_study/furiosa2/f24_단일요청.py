import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

start = time.perf_counter()
response = llm.invoke("npu의 장점을 3가지 설명해줘")
elapsed = time.perf_counter() - start

usage = response.usage_metadata or {}

print("응답 :", response.content)
print("지연 시간 :", round(elapsed,3), '초')
print('입력 토큰 :', usage.get('input_tokens',0))
print('출력 토큰 :', usage.get('output_tokens',0))

# 응답 : <think>
# Okay, I need to explain ........
# 지연 시간 : 1.234 초
# 입력 토큰 : 20
# 출력 토큰 : 100