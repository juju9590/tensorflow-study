# 원본 JSON에 안건명(agenda_title)이 있는지 확인하는 작업

from pathlib import Path
import json

# 1. 원본 JSON 경로
# 실제 파일 위치에 맞게 변경
json_path = Path(
    r"C:\mini_pj\data\org2\2024_광역 회의록\2024_강원특별자치도의회.json"
)

# 2. 파일 불러오기
if not json_path.exists():
    print("파일을 찾을 수 없습니다:", json_path)
    raise SystemExit

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. JSON 기본 구조 확인
print("===== JSON 구조 =====")
print("데이터 타입:", type(data).__name__)

if isinstance(data, dict):
    print("최상위 키:", list(data.keys()))
    print("\n첫 번째 값 미리보기:")

    first_key = next(iter(data), None)
    if first_key is not None:
        value = data[first_key]
        print("키:", first_key)
        print("값 타입:", type(value).__name__)
        print(str(value)[:1000])

elif isinstance(data, list):
    print("전체 레코드:", len(data))

    if data:
        print("첫 번째 레코드:")
        print(str(data[0])[:1000])

# 먼저 안건 목록 추출 테스트
from bs4 import BeautifulSoup

print("\n===== 회의록 안건 추출 테스트 =====")

# 첫 번째 회의록
record = data[0]

html = record.get("MINTS_HTML", "")
soup = BeautifulSoup(html, "html.parser")

# 의사일정
agenda_block = soup.select_one("ol#agenda-block")

print("\n[의사일정]")

if agenda_block:
    for li in agenda_block.select("li"):
        text = li.get_text(" ", strip=True)
        print("-", text)
else:
    print("의사일정 블록 없음")

# 부의된 안건
item_block = soup.select_one("ol#item-block")

print("\n[부의된 안건]")

if item_block:
    for li in item_block.select("li"):
        text = li.get_text(" ", strip=True)
        print("-", text)
else:
    print("부의된 안건 블록 없음")


# =====================================================
# 본문 안건 구분자 구조 확인
# =====================================================

print("\n===== 본문 내 안건 표시 후보 =====")

# 안건 제목과 비슷한 문구가 포함된 요소 탐색
keywords = [
    "회기 및 의사일정 결정의 건",
    "신년연설",
    "휴회의 건",
    "5분 자유발언"
]

for keyword in keywords:
    print(f"\n[검색어: {keyword}]")

    matches = soup.find_all(
        string=lambda s: s and keyword in s
    )

    for match in matches[:8]:
        parent = match.parent

        print(
            "태그:", parent.name,
            "| class:", parent.get("class"),
            "| id:", parent.get("id"),
            "| 텍스트:", parent.get_text(" ", strip=True)[:150]
        )

# =====================================================
# 본문 안건 구분자 전체 확인
# =====================================================

print("\n===== 본문 안건 구분자 =====")

items = soup.select("strong.item-in-contents")

print("발견된 본문 안건 수:", len(items))

for i, item in enumerate(items, 1):
    print(
        f"{i}.",
        "id:", item.get("id"),
        "| 안건:", item.get_text(" ", strip=True)
    )

# =====================================================
# 본문 발언 블록 수 확인
# =====================================================

speakers = soup.select("div.contents-block.speaker-block")

print("\n본문 발언 블록 수:", len(speakers))

for block in speakers[:3]:
    print(
        block.get_text(" ", strip=True)[:150]
    )
# =====================================================
# 8. 본문 안건별 발언 블록 연결 테스트
# =====================================================

from collections import defaultdict

agenda_speeches = defaultdict(list)

current_agenda = "안건 미배정"

# HTML 본문에서 안건과 발언 블록을 문서 순서대로 탐색
for element in soup.select(
    "strong.item-in-contents, div.contents-block.speaker-block"
):

    # 1. 안건 구분자 발견
    if element.name == "strong":
        current_agenda = element.get_text(" ", strip=True)

    # 2. 발언 블록 발견
    elif element.name == "div":
        speech = element.get_text(" ", strip=True)

        if speech:
            agenda_speeches[current_agenda].append(speech)

# =====================================================
# 9. 안건별 발언 결과 출력
# =====================================================

print("\n===== 안건별 발언 연결 결과 =====")

total = 0

for agenda, speeches in agenda_speeches.items():

    print(f"\n[안건] {agenda}")
    print(f"발언 블록: {len(speeches)}개")

    for i, speech in enumerate(speeches[:2], 1):
        print(f"  발언 {i}: {speech[:120]}")

    total += len(speeches)

print("\n전체 연결된 발언 블록:", total)
print("원본 발언 블록:", len(speakers))


# =====================================================
# 10. 실제 청크의 원본 회의록 안건 확인
# =====================================================

record_index = 24

record = data[record_index]
soup_target = BeautifulSoup(
    record.get("MINTS_HTML", ""),
    "html.parser"
)

print("\n===== 실제 청크의 원본 회의록 =====")
print("레코드 번호:", record_index)
print("날짜:", record.get("MTG_DE"))
print("의회:", record.get("RASMBLY_NM"))

# 본문 안건 확인
items = soup_target.select("strong.item-in-contents")

print("\n[본문 안건 목록]")
for item in items:
    print(
        item.get("id"),
        ":",
        item.get_text(" ", strip=True)
    )

# 안건별 발언 연결
from collections import defaultdict

agenda_speeches = defaultdict(list)
current_agenda = "안건 미배정"

for element in soup_target.select(
    "strong.item-in-contents, div.contents-block.speaker-block"
):
    if element.name == "strong":
        current_agenda = element.get_text(" ", strip=True)

    elif element.name == "div":
        speech = element.get_text(" ", strip=True)

        if speech:
            agenda_speeches[current_agenda].append(speech)

print("\n[안건별 발언 수]")
for agenda, speeches in agenda_speeches.items():
    print(f"{agenda}: {len(speeches)}개")

# =====================================================
# 11. 실제 청크와 안건별 발언 내용 대조
# =====================================================

import pandas as pd
from pathlib import Path

# 임베딩 메타데이터 CSV 검색
base_dir = Path(
    r"C:\mini_pj\data\processed\embeddings_by_region\2024\강원특별자치도의회"
)

csv_files = sorted(base_dir.rglob("*.csv"))

target_chunk_id = "2024_강원특별자치도의회_24_70"

target_text = None

# 실제 청크 찾기
for file in csv_files:

    df = pd.read_csv(
        file,
        dtype=str,
        keep_default_na=False
    )

    if not {"chunk_id", "text"}.issubset(df.columns):
        continue

    matched = df[df["chunk_id"] == target_chunk_id]

    if not matched.empty:
        target_text = matched.iloc[0]["text"]
        print("청크 발견:", file.name)
        break

if target_text is None:
    print("대상 청크를 찾지 못했습니다.")
    raise SystemExit

# 청크의 앞부분을 기준으로 원본 발언과 대조
# 공백 및 줄바꿈 차이는 무시
def normalize(text):
    return " ".join(str(text).split())

normalized_chunk = normalize(target_text)

print("\n===== 청크·안건 연결 테스트 =====")

matches = []

for agenda, speeches in agenda_speeches.items():

    for speech in speeches:

        normalized_speech = normalize(speech)

        # 청크와 발언에 공통으로 포함된 문구 확인
        snippet = normalized_chunk[:60]

        if snippet and snippet in normalized_speech:
            matches.append(agenda)

print("청크 ID:", target_chunk_id)

if matches:
    print("안건 후보:")
    for agenda in dict.fromkeys(matches):
        print("-", agenda)
else:
    print("일치하는 안건을 찾지 못했습니다.")
    print("청크 시작 부분:", normalized_chunk[:150])
