from kingkeras import run_once, get_cache_metrics

prefix = "npu 추론 최적화에 대한 설명입니다." * 150

for i in range(1, 5):
    print(f'\n ===== {i}회 요청 시작 ===== ')
    prompt = prefix + f'\n 질문 {i}: KV Cache란?'

    before = get_cache_metrics() # 요청 전 캐시 상태 
    result = run_once(prompt, 100)
    after = get_cache_metrics() # 요청 후 캐시 상태 

    # 이번 요청에서 증가한 횟수
    hits = after["hits"] - before["hits"]
    queries = after["queries"] - before["queries"]

    if queries >0 :
        hit_rate = hits / queries * 100
    else :
        hit_rate = 0

    print(f'\n ===== {i}회 요청 결과 ===== ')

    print("응답 시간 :", round(result["elapsed"],3))
    print('입력 토큰 :', result['input_tokens'])

    print("이번 조회 횟수 :", int(queries)) #형변환
    print("이번 적중 횟수 :", int(hits)) #형변환
    print("이번 적중률 :", round(hit_rate,2), '%') #형변환


#     ===== 1회 요청 시작 ===== 

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 20398
# Prefix Cache 적중 횟수 : 18659
# Prefix Cache 적중률 : 91.47
# kv Cache 사용률 : 3.75

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 22064
# Prefix Cache 적중 횟수 : 20325
# Prefix Cache 적중률 : 92.12
# kv Cache 사용률 : 3.79

#  ===== 1회 요청 결과 ===== 
# 응답 시간 : 1.203
# 입력 토큰 : 1667
# 이번 조회 횟수 : 1666
# 이번 적중 횟수 : 1666
# 이번 적중률 : 100.0 %

#  ===== 2회 요청 시작 ===== 

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 22064
# Prefix Cache 적중 횟수 : 20325
# Prefix Cache 적중률 : 92.12
# kv Cache 사용률 : 3.79

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 23730
# Prefix Cache 적중 횟수 : 21991
# Prefix Cache 적중률 : 92.67
# kv Cache 사용률 : 3.82

#  ===== 2회 요청 결과 ===== 
# 응답 시간 : 1.168
# 입력 토큰 : 1667
# 이번 조회 횟수 : 1666
# 이번 적중 횟수 : 1666
# 이번 적중률 : 100.0 %

#  ===== 3회 요청 시작 ===== 

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 23730
# Prefix Cache 적중 횟수 : 21991
# Prefix Cache 적중률 : 92.67
# kv Cache 사용률 : 3.82

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 25396
# Prefix Cache 적중 횟수 : 23657
# Prefix Cache 적중률 : 93.15
# kv Cache 사용률 : 3.85

#  ===== 3회 요청 결과 ===== 
# 응답 시간 : 1.162
# 입력 토큰 : 1667
# 이번 조회 횟수 : 1666
# 이번 적중 횟수 : 1666
# 이번 적중률 : 100.0 %

#  ===== 4회 요청 시작 ===== 

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 25396
# Prefix Cache 적중 횟수 : 23657
# Prefix Cache 적중률 : 93.15
# kv Cache 사용률 : 3.85

# ====== furiosa NPU Cache 상태 =======
# Prefix Cache 조회 횟수 : 27062
# Prefix Cache 적중 횟수 : 25323
# Prefix Cache 적중률 : 93.57
# kv Cache 사용률 : 3.89

#  ===== 4회 요청 결과 ===== 
# 응답 시간 : 1.224
# 입력 토큰 : 1667
# 이번 조회 횟수 : 1666
# 이번 적중 횟수 : 1666
# 이번 적중률 : 100.0 %







