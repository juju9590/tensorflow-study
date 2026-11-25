import time
import statistics
from concurrent.futures import ThreadPoolExecutor #동시작업

from kingkeras import run_once

prompts = [
    f"npu 추론 최적화 방법 {i}번을 자세히 설명해줘"  #f" 포멧팅 스트링
    for i in range(8)
]

aaa = run_once("npu가 뭐야?", 100)

# def ask(prompt):
#     return run_once(prompt, 100)

for workers in [1,2,4,8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(
                lambda prompts : run_once(prompts,50), #함수
                prompts #리스트
            )
        )
    elapsed = time.perf_counter() - start

    total_tokens = sum(
        r['output_tokens'] for r in results
    )


    avg_latency = statistics.mean(
        r["elapsed"] for r in results
    )

    print(f'\n 동시 요청 수 :, {workers}')
    print(f'전체 시간 :, {elapsed:.2f}초')
    print(f'전체 출력 토큰 :, {total_tokens}')
    print(f'처리량 :, {total_tokens/elapsed:.2f}')
    print(f'평균 요청 지연 :, {avg_latency:.2f}초')


#      동시 요청 수 :, 1
# 전체 시간 :, 4.78초
# 전체 출력 토큰 :, 400
# 처리량 :, 83.73
# 평균 요청 지연 :, 0.60초

#  동시 요청 수 :, 2
# 전체 시간 :, 2.69초
# 전체 출력 토큰 :, 400
# 처리량 :, 148.47
# 평균 요청 지연 :, 0.67초

#  동시 요청 수 :, 4
# 전체 시간 :, 1.44초
# 전체 출력 토큰 :, 400
# 처리량 :, 277.19
# 평균 요청 지연 :, 0.72초

#  동시 요청 수 :, 8
# 전체 시간 :, 0.84초
# 전체 출력 토큰 :, 400
# 처리량 :, 475.37
# 평균 요청 지연 :, 0.83초




