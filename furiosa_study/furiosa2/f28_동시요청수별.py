# 동시요청수별 성능 비교


#RPS : Request per seconde

import time
import statistics
from kingkeras import get_llm
from concurrent.futures import ThreadPoolExecutor



llm = get_llm(max_tokens=100)

prompts = [
    f"NPU 추론 최적화 방법 {i}번 설명해줘" for  i in range(16)
]

def ask(prompt):

    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata or {}
    tokens = usage.get("output_tokens",0)

    return elapsed, tokens


for workes in [1,2,4,8]:

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workes) as executor:
        results = list(executor.map(ask, prompts))
    total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    total_tokens = sum(r[1] for r in results)

    print(f'\n =========== Workers : {workes} ==========')

    print('전체 시간:', round(total_time,3),'초')
    print("평균 지연 :", round(statistics.mean(latencies),2))
    print('요청 처리량 :', round(len(results)/total_time,2),"req/s")
    print('output TPS :', round(total_tokens/total_time,3),'초')

print()

#  =========== Workers : 1 ==========
# 전체 시간: 19.178 초
# 평균 지연 : 1.2
# 요청 처리량 : 0.83 req/s
# output TPS : 83.427 초

#  =========== Workers : 2 ==========
# 전체 시간: 10.821 초
# 평균 지연 : 1.35
# 요청 처리량 : 1.48 req/s
# output TPS : 147.862 초

#  =========== Workers : 4 ==========
# 전체 시간: 5.555 초
# 평균 지연 : 1.39
# 요청 처리량 : 2.88 req/s
# output TPS : 288.011 초

#  =========== Workers : 8 ==========
# 전체 시간: 3.024 초
# 평균 지연 : 1.51
# 요청 처리량 : 5.29 req/s
# output TPS : 529.121 초






