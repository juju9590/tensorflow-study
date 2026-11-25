# 평균 응답시간 


from kingkeras import get_llm
import time

llm = get_llm(max_tokens=500)

prompt ="npu의 특징을 말해줘"

times = []
for i in range(5):
    start = time.perf_counter()

    response = llm.invoke(prompt)

    end = time.perf_counter()

    elasped = end - start

    times.append(elasped)
    print(i+1, "회:", round(elasped,2),'초')

average = sum(times)/len(times)

print("평균 :", round(average, 2), '초')

# 1 회: 6.05 초
# 2 회: 6.07 초
# 3 회: 5.83 초
# 4 회: 5.99 초
# 5 회: 6.23 초
# 평균 : 6.03 초

