def multiply(x):
    return x * 10

numbers = [1,2,3,4]

result = map(multiply, numbers) # map 함수 (함수, 리스트)

print(result) 
# <map object at 0x000001E2B720ACB0> 
# 이터레이터 형태로 반환 => for문, next 생각해야 함

print(list(result)) #[10, 20, 30, 40]