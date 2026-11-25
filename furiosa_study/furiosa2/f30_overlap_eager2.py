# 28, 30-1 카피

"""
furiosa-llm serve furiosa-ai/Qwen3-4B-FP8 \
    --device npu:5 \
    --host 127.0.0.1 \
    --port 8000
"""

from concurrent.futures import ThreadPoolExecutor
from setting import get_llm
import statistics
import time

llm = get_llm(max_tokens=200)

prompts = [
    f"NPU 추론 최적화 방법 {i}번을 설명해줘" for i in range(16)
]

def ask(prompt):
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens", 0)

    return elapsed, tokens

for workers in [1,2,4,8]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(ask, prompts))
    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f"\n===== Workers: {workers} =====")

    print("전체 시간 :", round(total_time, 2), "초")
    print("평균 지연 :", round(statistics.mean(latencies), 2))
    print("요청 처리량 :", round(len(results) / total_time, 2), "req/s")
    print("Ouput TPS :", round(total_tokens / total_time, 2))

# Results - eager mode
# ===== Workers: 8 =====
# 전체 시간 : 3.98 초
# 평균 지연 : 1.99
# 요청 처리량 : 4.02 req/s
# Ouput TPS : 803.31

# Results - overlap mode
# ===== Workers: 8 =====
# 전체 시간 : 3.93 초
# 평균 지연 : 1.96
# 요청 처리량 : 4.07 req/s
# Ouput TPS : 813.91