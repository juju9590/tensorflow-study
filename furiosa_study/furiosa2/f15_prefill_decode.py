from kingkeras import get_llm
import time

llm = get_llm()

prompt = [
    'npu란?',
    'npu와 gpu의 차이를 5가지 설명해줘',
    'npu의 구조와 장점, 활용 분야를 자세히 설명해줘',
]

for prompt in prompt:
    start = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start

    print("질문 :", prompt)
    print("시간 :", round(elapsed, 2), '초')
    # 경과시간 : decoder
    # 채우기 직전까지 시간 : prefill

# 질문 : npu란?
# 시간 : 9.81 초
# 질문 : npu와 gpu의 차이를 5가지 설명해줘
# 시간 : 15.52 초
# 질문 : npu의 구조와 장점, 활용 분야를 자세히 설명해줘
# 시간 : 37.56 초
