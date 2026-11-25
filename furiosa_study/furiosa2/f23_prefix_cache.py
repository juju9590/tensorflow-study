from kingkeras import run_once, get_cache_metrics

prefix = "npu 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    prompt = prefix + f'\n 질문 {i}: KV Cache란?'

    print(f'\n ===== {i}회 요청 ===== ')

    result = run_once(prompt, 100)

    print("응답 시간 :", round(result["elapsed"],3))
    print('입력 토큰 :', result['input_tokens'])

    print("\nPrefix Cache 메트릭")
    get_cache_metrics()


# ===== 1회 요청 ===== 
# 응답 시간 : 1.343
# 입력 토큰 : 1667

# Prefix Cache 메트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 2072
# Prefix Cache 적중 횟수 : 363
# Prefix Cache 적중률 : 17.52
# kv Cache 사용률 : 3.41

#  ===== 2회 요청 ===== 
# 응답 시간 : 1.164
# 입력 토큰 : 1667

# Prefix Cache 메트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 3738
# Prefix Cache 적중 횟수 : 2019
# Prefix Cache 적중률 : 54.01
# kv Cache 사용률 : 3.45

#  ===== 3회 요청 ===== 
# 응답 시간 : 1.154
# 입력 토큰 : 1667

# Prefix Cache 메트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 5404
# Prefix Cache 적중 횟수 : 3675
# Prefix Cache 적중률 : 68.01
# kv Cache 사용률 : 3.49

#  ===== 4회 요청 ===== 
# 응답 시간 : 1.162
# 입력 토큰 : 1667

# Prefix Cache 메트릭

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 7070
# Prefix Cache 적중 횟수 : 5331
# Prefix Cache 적중률 : 75.40
# kv Cache 사용률 : 3.53