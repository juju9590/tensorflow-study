score = [85,45,70,55,95]

results = list(map(lambda x:'합격' if x>=60 else '불합격', score)) #map(함수, 리스트)

print(results) #['합격', '불합격', '합격', '불합격', '합격']