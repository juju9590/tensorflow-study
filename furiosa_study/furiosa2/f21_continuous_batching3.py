import time
from concurrent.futures import ThreadPoolExecutor
from kingkeras import run_once

def ask(name, delay, max_tokens):
    time.sleep(delay)

    print(f'{name} 요청 시작', flush=True)

    start = time.perf_counter()
    result = run_once(
        f'npu와 llm 추론 최적화를 자세히 설명해줘. 요청 {name}',
        max_tokens
    )
    elapsed = time.perf_counter() - start

    print(f'{name} 요청 완료: {elapsed:.2f} 초', flush=True)

    return result

# 동시작업, = 동기
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(ask, "A", 0, 800),
        pool.submit(ask, "B", 1, 200),
        pool.submit(ask, "C", 2, 1000),
        pool.submit(ask, "D", 3, 200),
        pool.submit(ask, "E", 4, 300),
    ]
    results = [f.result() for f in futures]

print("====== ===== 끝 ===== ======")

# 같은 표현
# pool = ThreadPoolExecutor(max_workers=4)
# futures = [
#     pool.submit(ask, "A", 0, 800),
#     pool.submit(ask, "B", 1, 200),
#     pool.submit(ask, "C", 2, 1000),
#     pool.submit(ask, "D", 3, 200),
#     pool.submit(ask, "E", 4, 300),
# ]
# results = [f.result() for f in futures]
# pool.shutdown()



# Thread 동시작업 (동기처리)

# A 요청 시작
# B 요청 시작
# C 요청 시작
# D 요청 시작
# B 요청 완료: 2.73 초
# D 요청 완료: 2.87 초
# E 요청 시작
# A 요청 완료: 11.02 초
# E 요청 완료: 3.97 초
# C 요청 완료: 12.94 초



