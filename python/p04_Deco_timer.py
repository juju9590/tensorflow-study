# 실무에서는 다음과 같은 기능을 여러함수에 공통으로 적용할때 유용
# ㄴ 함수 실행 시간 측정
# ㄴ 로그 기록
# ㄴ 사용자 권한 검사
# ㄴ 예외처리
# ㄴ 함수 호출 횟수 기록



import time

def timer(func):
    def wrapper():
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        print("걸린시간 :", round(end - start, 6), '초')
    return wrapper


@timer
def hello():
    print('안녕하세요')
    time.sleep(2)

hello()

# 안녕하세요
# 걸린시간 : 2.000539 초


@timer
def study():
    print('npu 공부중이다')
    time.sleep(1.5)

study()

# npu 공부중이다
# 걸린시간 : 1.500635 초
