# 클래스와 함수의 차이
# 함수는 기능만, 괄호안의 값을 받는다.
# 클래슨 기능 + 여러가지, 상속(모든기능 + 알파)받는다

# 오늘의 키워드 : Node, Edge, state, Reducer
 
from typing import Dict, TypedDict    

sample_dict : Dict[str, str] = {    # 타입힌트...sample_dict 는 Dict[str, str] 형태로 쓰겠다 (타입힌트)
    'name' : '김말똥',
    'age' : 39,
    'job' : '회사원',
}

# print(sample_dict)
# # {'name': '김말똥', 'age': '39', 'job': '회사원'}
# # {'name': '김말똥', 'age': 39, 'job': '회사원'}

# print(sample_dict['name']) #김말똥

class Person(TypedDict) :
    name : str
    age : int
    job :  str

person = Person(name='홍길동', age=40, job='의적')
print(person) #{'name': '홍길동', 'age': 40, 'job': '의적'}


person2 : Person ={'name':'김말똥', 'age':'20', 'job' : '개발자'}  #클래스도 타입이다.
print(person2) #{'name': '김말똥', 'age': '20', 'job': '개발자'}
# : 다음에 대문자 오면 '타입 힌트'로 캐치캐치







