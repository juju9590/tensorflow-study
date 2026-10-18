import json
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup

# 이 코드 파일이 있는 mymy 폴더 기준
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "DB_sample"
output_dir = base_dir / "readable_text"

# 1. JSON 파일 목록 확인
json_files = sorted(data_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"JSON 파일이 없습니다: {data_dir}")

print("===== JSON 파일 목록 =====")
for number, path in enumerate(json_files, start=1):
    print(f"{number}. {path.name}")

# 2. 읽을 파일 선택
number = int(input("\n읽을 파일 번호를 입력하세요: "))

if not 1 <= number <= len(json_files):
    raise ValueError("목록에 있는 번호를 입력하세요.")

input_path = json_files[number - 1]

with input_path.open("r", encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"\n선택한 파일: {input_path.name}")
print(f"회의록 수: {len(records)}건")

# 3. 파일별 결과 폴더 생성
save_dir = output_dir / input_path.stem
save_dir.mkdir(parents=True, exist_ok=True)

# 4. 처음 3건을 읽기 편한 글로 변환
# for i, record in enumerate(records[:3], start=1):
for i, record in enumerate(records, start=1):     #전체
    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    date_raw = str(record.get("MTG_DE", ""))
    try:
        date = datetime.strptime(
            date_raw, "%Y%m%d"
        ).strftime("%Y-%m-%d")
    except ValueError:
        date = date_raw

    content = (
        f"의회: {record.get('RASMBLY_NM', '')}\n"
        f"위원회: {record.get('MTGNM', '')}\n"
        f"회의록: {record.get('MINTS_NM', '')}\n"
        f"날짜: {date}\n\n"
        f"[회의록 본문]\n\n{text}\n"
    )

    output_path = save_dir / f"회의록_{i:04d}.txt"
    output_path.write_text(content, encoding="utf-8")

    print(f"저장 완료: {output_path}")

print(f"\n결과 폴더: {save_dir}")