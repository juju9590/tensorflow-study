from kingkeras import run_once, get_cache_metrics

prompt = [
    'npu란 무엇인가?',
    'npu의 특징을 설명해줘'*100,
    'npu의 특징을 설명해줘'*300,
]


for i, prompt in enumerate(prompt,1):
    # print(i) : (prompt,1) 1부터 시작하라는 의미 , 0이 아니라
    print(f'\n ====== 실험 {i} ====== ')

    result = run_once(prompt, 300)
    print("입력 토큰", result['input_tokens'])
    print("출력 토큰", result['output_tokens'])
    print("응답 시간", round(result['elapsed'],2))

    print("\n kv_cache 매트릭")
    get_cache_metrics()


# ====== 실험 1 ====== 
# 입력 토큰 15
# 출력 토큰 300
# 응답 시간 3.57

#  kv_cache 매트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 14573
# Prefix Cache 적중 횟수 : 10042
# Prefix Cache 적중률 : 68.91
# kv Cache 사용률 : 2.96

#  ====== 실험 2 ====== 
# 입력 토큰 908
# 출력 토큰 300
# 응답 시간 3.63

#  kv_cache 매트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 15480
# Prefix Cache 적중 횟수 : 10949
# Prefix Cache 적중률 : 70.73
# kv Cache 사용률 : 3.07

#  ====== 실험 3 ====== 
# 입력 토큰 2708
# 출력 토큰 300
# 응답 시간 3.68

#  kv_cache 매트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 18187
# Prefix Cache 적중 횟수 : 13656
# Prefix Cache 적중률 : 75.09  # 재사용율이 높으면 좋은거
# kv Cache 사용률 : 3.17 # kv Cache는 낮을수록 좋다 90% 넘어가면

