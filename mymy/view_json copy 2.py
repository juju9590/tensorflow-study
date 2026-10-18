import json
import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup


# 1. 경로 설정
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "DB_sample"
output_dir = base_dir / "readable_text"


# 2. JSON 파일 목록
json_files = sorted(data_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"JSON 파일이 없습니다: {data_dir}")

print("===== JSON 파일 목록 =====")

for number, path in enumerate(json_files, start=1):
    print(f"{number}. {path.name}")


# 3. 분석할 파일 선택
try:
    number = int(input("\n읽을 파일 번호를 입력하세요: "))
except ValueError:
    raise ValueError("파일 번호는 숫자로 입력하세요.")

if not 1 <= number <= len(json_files):
    raise ValueError("목록에 있는 번호를 입력하세요.")

input_path = json_files[number - 1]

with input_path.open("r", encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"\n선택한 파일: {input_path.name}")
print(f"회의록 수: {len(records)}건")


# 4. 키워드와 동의어 설정
keyword = input("\n검색할 키워드를 입력하세요: ").strip()

if not keyword:
    raise ValueError("키워드를 입력하세요.")

synonyms = {
    "저출생": ["저출생", "저출산"],
    "저출산": ["저출생", "저출산"],
}

# 등록된 동의어가 없으면 입력한 키워드만 검색
search_words = synonyms.get(keyword, [keyword])

print(f"검색할 표현: {', '.join(search_words)}")


# 5. 결과 폴더 설정
safe_keyword = re.sub(r'[<>:"/\\|?*]', "_", keyword)
safe_keyword = safe_keyword.rstrip(" .") or "키워드"

save_dir = output_dir / input_path.stem / f"검색_{safe_keyword}"
save_dir.mkdir(parents=True, exist_ok=True)

matched_count = 0


# 6. 전체 회의록에서 키워드 검색
for i, record in enumerate(records, start=1):
    soup = BeautifulSoup(
        record.get("MINTS_HTML", "") or "",
        "html.parser"
    )

    # 본문과 관계없는 코드 제거
    for tag in soup(["script", "style"]):
        tag.decompose()

    # HTML을 텍스트로 변환
    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 동의어 중 실제로 포함된 표현 확인
    found_words = [
        word for word in search_words
        if word in text
    ]

    # 검색어가 없으면 다음 회의록으로 이동
    if not found_words:
        continue

    # 날짜 변환
    date_raw = str(record.get("MTG_DE", ""))

    try:
        date = datetime.strptime(
            date_raw, "%Y%m%d"
        ).strftime("%Y-%m-%d")
    except ValueError:
        date = date_raw

    # 회의록 정보와 전체 본문
    content = (
        f"의회: {record.get('RASMBLY_NM', '')}\n"
        f"위원회: {record.get('MTGNM', '')}\n"
        f"회의록: {record.get('MINTS_NM', '')}\n"
        f"날짜: {date}\n"
        f"검색된 표현: {', '.join(found_words)}\n\n"
        f"[회의록 본문]\n\n{text}\n"
    )

    # 원본 JSON 순번을 파일명으로 사용
    # output_path = save_dir / f"회의록_{i:04d}.txt"

    # 변경
    file_date = re.sub(r'[<>:"/\\|?*]', "_", date) or "날짜없음"
    output_path = save_dir / f"{safe_keyword}_{file_date}_{i:04d}.txt"

    output_path.write_text(content, encoding="utf-8")

    matched_count += 1
    print(f"저장 완료: {output_path.name}")


# 7. 결과 안내
print("\n===== 검색 완료 =====")
print(f"검색 키워드: {keyword}")
print(f"검색한 표현: {', '.join(search_words)}")
print(f"전체 {len(records)}건 중 {matched_count}건 저장")
print(f"결과 폴더: {save_dir}")