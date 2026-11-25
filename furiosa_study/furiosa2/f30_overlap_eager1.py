"""
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --device npu:5 \
    --host 127.0.0.1 \
    --port 8000
    --enable-overlap-scheduling
"""

from setting import get_llm
import time

llm = get_llm(max_tokens=100)

prompts = [f"NPU 추론 최적화 방법 {i}번을 설명해줘." for i in range(8)] # list comprehension + f-string 

start = time.perf_counter()
total_tokens = 0

for prompt in prompts:
    response = llm.invoke(prompt)

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    total_tokens += tokens

elapsed = time.perf_counter() - start

print("전체 요청 :", len(prompts))
print("전체 출력 토큰 :", total_tokens)
print("전체 소요시간 :", round(elapsed, 3), "초")
print("RPS :", round(len(prompts) / elapsed, 3))
print("TPS :", round(total_tokens / elapsed, 3))

# Result - Eager mode   # eager은 lazy와 대비되는 단어. lazy <-> eager
# 전체 요청 : 8
# 전체 출력 토큰 : 800
# 전체 소요시간 : 6.546 초
# RPS : 1.222
# TPS : 122.212

# Result - Overlap mode
# 전체 요청 : 8
# 전체 출력 토큰 : 800
# 전체 소요시간 : 6.56 초
# RPS : 1.22
# TPS : 121.958