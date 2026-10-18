import json
from pathlib import Path
from collections import Counter

from bs4 import BeautifulSoup
from kiwipiepy import Kiwi

# 1. 경로 설정
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "DB_sample"

json_files = sorted(data_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"JSON 파일이 없습니다: {data_dir}")

for i, path in enumerate(json_files, start=1):
    print(f"{i}. {path.name}")

number = int(input("\n분석할 파일 번호: "))

if not 1 <= number <= len(json_files):
    raise ValueError("목록에 있는 번호를 입력하세요.")

input_path = json_files[number - 1]

with input_path.open("r", encoding="utf-8-sig") as f:
    records = json.load(f)

# 2. 형태소 분석기와 제외할 단어
kiwi = Kiwi()

stopwords = {
    "위원", "위원장", "의원", "의장", "위원회",
    "회의", "회의록", "의사일정", "안건",
    "말씀", "감사", "질의", "답변",
    "사항", "부분", "경우", "관련",
    "생각", "내용", "이번", "오늘",
}

word_counts = Counter()

# 3. 회의록별 본문 분석
for i, record in enumerate(records, start=1):
    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # 일반명사(NNG), 고유명사(NNP)만 추출
    words = [
        token.form
        for token in kiwi.tokenize(text)
        if token.tag in {"NNG", "NNP"}
        and len(token.form) >= 2
        and token.form not in stopwords
    ]

    word_counts.update(words)

    if i % 10 == 0 or i == len(records):
        print(f"분석 중: {i}/{len(records)}건")

# 4. 상위 20개 출력
print(f"\n===== {input_path.name}: 상위 20개 명사 =====")

for rank, (word, count) in enumerate(
    word_counts.most_common(20), start=1
):
    print(f"{rank:2d}. {word:<12} {count:,}회")