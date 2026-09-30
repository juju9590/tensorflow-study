import os
key = os.getenv("OPENAI_API_KEY")
# 환경변수에 저장 후 사용 가능, getenv : 환경변수에서 가져온다

if key is None:
    print("OPENAI_API_KEY 없습니다")
else:
    print("키 길이 :", len(key))
    print("키 확인 :", key[:8]+"..."+ key[-4:])


