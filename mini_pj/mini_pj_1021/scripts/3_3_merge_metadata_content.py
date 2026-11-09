#3_1 html 본문 추출


from pathlib import Path
from bs4 import BeautifulSoup

# 1. HTML 샘플 파일 경로
file_path = Path(
    "C:/tensorflow_study/mini_pj/data/html_samples/"
    "2025_강원특별자치도의회/sample_1.html"
)

# 2. HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 3. HTML 분석
soup = BeautifulSoup(html, "html.parser")

# 4. 불필요한 태그 제거
for tag in soup(["script", "style"]):
    tag.decompose()

# 5. HTML에서 텍스트 추출
text = soup.get_text("\n", strip=True)

# 6. 홈페이지 불필요 문구 제거
remove_text = "강원특별자치도의회 의원프로필"

if remove_text in text:
    text = text.split(remove_text)[0]

# 7. 결과 확인
# print("정리 후 글자 수:", len(text))

# print("\n===== 마지막 500자 =====")
# print(text[-500:])


# 3_2 Json 메타데이터 추출

import json

from pathlib import Path
from bs4 import BeautifulSoup

# 1. HTML 샘플 파일 경로
file_path = Path(
    "C:/tensorflow_study/mini_pj/data/html_samples/"
    "2025_강원특별자치도의회/sample_1.html"
)

# 2. HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 3. HTML 분석
soup = BeautifulSoup(html, "html.parser")

# 4. 불필요한 태그 제거
for tag in soup(["script", "style"]):
    tag.decompose()

# 5. HTML에서 텍스트 추출
text = soup.get_text("\n", strip=True)

# 6. 홈페이지 불필요 문구 제거
remove_text = "강원특별자치도의회 의원프로필"

if remove_text in text:
    text = text.split(remove_text)[0]

# 7. 결과 확인
# print("정리 후 글자 수:", len(text))

# print("\n===== 마지막 500자 =====")
# print(text[-500:])


##### 제이슨 메타 데이터 추출

# 1. 강원도 원본 JSON 경로
json_path = "C:/tensorflow_study/mini_pj/data/org2/2025_광역 회의록/2025_강원특별자치도의회.json"

# 2. JSON 파일 읽기
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 첫 번째 회의록 선택
record = data[0]

# 4. 메타데이터 추출
date = record.get("MTG_DE")
assembly = record.get("RASMBLY_NM")
meeting_title = record.get("MINTS_NM")

# 5. 출력
# print("회의 날짜:", date)
# print("의회명:", assembly)
# print("회의록명:", meeting_title)

#####################################################
# 메타데이터 + 본문 통합
# # ③-5. 최종 데이터 구조 만들기

result = {
    "date": date,
    "assembly": assembly,
    "agenda_title": None,
    "speaker_role": None,
    "speaker_name": None,
    "speaker_type": None,
    "utterance": text,
    "meeting_title": meeting_title,
    "source_id": "2025_강원특별자치도의회_0"
}

# 결과 확인
print("\n===== 최종 데이터 구조 =====")

for key, value in result.items():

    if key == "utterance":
        print(f"{key}: {len(value)}자")

    else:
        print(f"{key}: {value}")


# ===== 최종 데이터 구조 =====
# date: 20250204
# assembly: 강원특별자치도의회
# agenda_title: None
# speaker_role: None
# speaker_name: None
# speaker_type: None
# utterance: 16256자
# meeting_title: 기타 회의록 제334회(1차)
# source_id: 2025_강원특별자치도의회_0