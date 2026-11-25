import time
import statistics
from concurrent.futures import ThreadPoolExecutor #동시작업

from kingkeras import run_once

prompts = [
    f"npu 추론 최적화 방법 {i}번을 자세히 설명해줘"  #f" 포멧팅 스트링
    for i in range(8)
]
# print(prompt)
# print(len(prompt)) #8

# ['npu 추론 최적화 방법 0번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 1번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 2번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 3번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 4번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 5번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 6번을 자세히 설명해줘', 
# 'npu 추론 최적화 방법 7번을 자세히 설명해줘']

aaa = run_once("npu가 뭐야?", 100)
# print(aaa["text"])
# NPU는 **Neural Processing Unit**(신경 처리 장치)의 약어로, 인공지능(AI) 및 머신러닝(ML) 알고리즘을 효율적으로 실행하기 위해 설계된 전용 프로세서입니다. NPU는 주로 신경망(예: 인공 신경망)의 병렬 처리 작업을 최적화하여, 이미지 인식, 자연어 처리, 자율주행, 스마트폰 등에서 실시간 AI 처리가 가능하도록 지원합니다.


def ask(prompt):
    return run_once(prompt, 100)

for workers in [1,2,4,8]:
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            pool.map(ask, prompts)
        )
    elapsed = time.perf_counter() - start

    total_tokens = sum(
        r['output_tokens'] for r in results
    )
    # print('workers : ', workers) #
    # print(len(results)) # 8

    avg_latency = statistics.mean(
        r["elapsed"] for r in results
    )

    print(f'\n 동시 요청 수 :, {workers}')
    print(f'전체 시간 :, {elapsed:.2f}초')
    print(f'전체 출력 토큰 :, {total_tokens}')
    print(f'처리량 :, {total_tokens/elapsed:.2f}')
    print(f'평균 요청 지연 :, {avg_latency:.2f}초')


#     동시 요청 수 :, 1
# 전체 시간 :, 10.12초
# 전체 출력 토큰 :, 800
# 처리량 :, 79.09
# 평균 요청 지연 :, 1.26초

#  동시 요청 수 :, 2
# 전체 시간 :, 5.59초
# 전체 출력 토큰 :, 800
# 처리량 :, 143.21
# 평균 요청 지연 :, 1.39초

#  동시 요청 수 :, 4
# 전체 시간 :, 2.92초
# 전체 출력 토큰 :, 800
# 처리량 :, 273.55
# 평균 요청 지연 :, 1.46초

#  동시 요청 수 :, 8
# 전체 시간 :, 1.57초
# 전체 출력 토큰 :, 800
# 처리량 :, 511.04
# 평균 요청 지연 :, 1.55초




