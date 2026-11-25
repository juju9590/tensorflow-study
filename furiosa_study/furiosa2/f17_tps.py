from kingkeras import get_llm
import time

llm = get_llm(max_tokens=1000)

prompt = 'npu의 특징을 자세히 설명해줘'

start = time.perf_counter()

response = llm.invoke(prompt)

elapsed = time.perf_counter() - start

tokens = response.usage_metadata["output_tokens"]

tps = tokens/elapsed

print("출력토큰 : ", tokens)
print("전체시간 : ", round(elapsed,2), '초')
print("TPS :", round(tps,2))

# 출력토큰 :  1000
# 전체시간 :  12.03 초
# TPS : 83.1