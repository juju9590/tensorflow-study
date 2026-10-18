import csv
import json
import re
from pathlib import Path
from datetime import datetime

from bs4 import BeautifulSoup


# 1. 경로 설정
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "DB_sample"
output_dir = base_dir / "search_csv"
output_dir.mkdir(parents=True, exist_ok=True)


# 2. HTML을 텍스트로 변환
def clean_html(html):
    soup = BeautifulSoup(html or "", "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    return lines


# 3. 발언 시작 여부 확인
def is_speaker_line(line):
    # 서울 등: ○위원장 김영옥, ○ 김규남
    if line.startswith("○"):
        return True

    # 인천 등: 위원장 유경희, 여성가족국장 시현정
    pattern = (
        r"^(?:"
        r"[가-힣·ㆍ]+위원장(?:직무대행|대리)?|"
        r"위원장(?:직무대행|대리)?|"
        r"부위원장|의장|부의장|"
        r"[가-힣·ㆍ]+(?:국장|과장|실장|본부장|"
        r"원장|사장|이사장|센터장|단장|"
        r"전문위원|담당관|교육감|시장|후보자)|"
        r"시장|교육감"
        r")\s+[가-힣]{2,5}$"
    )

    if re.fullmatch(pattern, line):
        return True

    # 김영옥 위원 / 홍길동 의원
    if re.fullmatch(r"[가-힣]{2,5}\s+(?:위원|의원)", line):
        return True

    return False


# 4. 회의록을 발언별로 분리
def split_speeches(lines):
    blocks = []
    current_lines = []

    for line in lines:
        if is_speaker_line(line):
            if current_lines:
                blocks.append("\n".join(current_lines))

            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


# 5. 날짜 변환
def format_date(value):
    value = str(value or "")

    try:
        return datetime.strptime(
            value, "%Y%m%d"
        ).strftime("%Y-%m-%d")
    except ValueError:
        return value


# 6. JSON 파일 선택
json_files = sorted(data_dir.glob("*.json"))

if not json_files:
    raise FileNotFoundError(f"JSON 파일이 없습니다: {data_dir}")

print("===== JSON 파일 목록 =====")

for i, path in enumerate(json_files, start=1):
    print(f"{i}. {path.name}")

try:
    number = int(input("\n읽을 파일 번호: "))
except ValueError:
    raise ValueError("파일 번호는 숫자로 입력하세요.")

if not 1 <= number <= len(json_files):
    raise ValueError("목록에 있는 번호를 입력하세요.")

input_path = json_files[number - 1]

with input_path.open("r", encoding="utf-8-sig") as f:
    records = json.load(f)

print(f"\n선택한 파일: {input_path.name}")
print(f"전체 회의록: {len(records)}건")


# 7. 키워드 입력
# 빈칸으로 입력하면 모든 회의록 저장
keyword = input(
    "\n검색 키워드 입력 — 전체 회의록은 Enter: "
).strip()

synonyms = {
    "저출생": ["저출생", "저출산"],
    "저출산": ["저출생", "저출산"],
}

search_words = synonyms.get(keyword, [keyword]) if keyword else []

if search_words:
    print(f"검색할 표현: {', '.join(search_words)}")
else:
    print("모든 회의록을 저장합니다.")


# 8. 날짜순 정렬
def date_sort_key(record):
    value = format_date(record.get("MTG_DE", ""))

    try:
        return (0, datetime.strptime(value, "%Y-%m-%d"))
    except ValueError:
        return (1, datetime.max)


records.sort(key=date_sort_key)


# 9. 출력 파일 설정
safe_keyword = re.sub(
    r'[<>:"/\\|?*]', "_", keyword
).rstrip(" .") or "전체"

output_path = (
    output_dir
    / f"{input_path.stem}_{safe_keyword}_발언별.csv"
)

columns = [
    "날짜",
    "의회",
    "위원회",
    "회의록",
    "검색된 표현",
    "회의록 본문",
]

meeting_count = 0
row_count = 0


# 10. 회의록별 전체 발언을 CSV로 저장
with output_path.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for index, record in enumerate(records, start=1):
        lines = clean_html(record.get("MINTS_HTML", ""))
        full_text = "\n".join(lines)

        # 키워드가 포함된 회의록 선택
        # 선택된 회의록의 모든 발언을 저장
        if search_words and not any(
            word in full_text for word in search_words
        ):
            continue

        blocks = split_speeches(lines)

        if not blocks:
            continue

        meeting_count += 1

        # 같은 회의의 출처 정보를 모든 행에 반복
        date = format_date(record.get("MTG_DE", ""))
        council = record.get("RASMBLY_NM", "")
        committee = record.get("MTGNM", "")
        meeting_name = record.get("MINTS_NM", "")

        for block in blocks:
            found_words = [
                word for word in search_words
                if word in block
            ]

            writer.writerow({
                "날짜": date,
                "의회": council,
                "위원회": committee,
                "회의록": meeting_name,
                "검색된 표현": ", ".join(found_words),
                "회의록 본문": block,
            })

            row_count += 1

        print(
            f"처리 완료: {index}/{len(records)} "
            f"— {meeting_name}"
        )


# 11. 결과 안내
print("\n===== 저장 완료 =====")
print(f"선택된 회의록: {meeting_count}건")
print(f"저장된 행: {row_count}행")
print(f"저장 위치: {output_path}")


