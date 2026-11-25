import time
from kingkeras import get_llm

llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번 설명해줘" for  i in range(8)
]

start = time.perf_counter()
for prompt in prompts :
    llm.invoke(prompt)
elapsed = time.perf_counter() - start


print('전체 요청:', len(prompts))
print('전체 소요 시간:', round(elapsed,3),'초')
print("요청 처리량 :", round(len(prompts)/elapsed, 3),"req/s")

# 전체 요청: 8
# 전체 소요 시간: 9.603 초
# 요청 처리량 : 0.833 req/s

