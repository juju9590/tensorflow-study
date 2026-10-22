from pathlib import Path
import pandas as pd

# 13_5 검증 결과
path = Path(
    r"C:\mini_pj\data\processed\validation_samples"
    r"\agenda_matching\2024\강원특별자치도의회"
    r"\improved\agenda_match_improved.csv"
)

df = pd.read_csv(path, dtype=str, keep_default_na=False)

# 1. 미배정 청크 확인
unassigned = df[df["match_status"] == "UNASSIGNED"]

print("UNASSIGNED 청크:", len(unassigned))

# 2. 원본 회의록별 미배정 건수
summary = (
    unassigned.groupby(["record_index", "date"])
    .size()
    .reset_index(name="unassigned_count")
    .sort_values("unassigned_count", ascending=False)
)

print("\n===== 미배정 상위 20개 회의록 =====")
print(summary.head(20).to_string(index=False))

# 3. 전체 회의록별 청크 수와 비교
total = (
    df.groupby(["record_index", "date"])
    .size()
    .reset_index(name="total_count")
)

summary = summary.merge(
    total,
    on=["record_index", "date"],
    how="left"
)

summary["unassigned_ratio"] = (
    summary["unassigned_count"]
    / summary["total_count"] * 100
).round(2)

print("\n===== 미배정 비율 상위 20개 =====")
print(
    summary.sort_values(
        ["unassigned_ratio", "unassigned_count"],
        ascending=False
    ).head(20).to_string(index=False)
)

import json
from pathlib import Path
from bs4 import BeautifulSoup

# 원본 JSON
json_path = Path(
    r"C:\mini_pj\data\org2\2024_광역 회의록"
    r"\2024_강원특별자치도의회.json"
)

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

# 미배정률 100% 회의록
idx = 215
record = records[idx]

soup = BeautifulSoup(
    record.get("MINTS_HTML", ""),
    "html.parser"
)

print("===== 원본 회의록 확인 =====")
print("레코드:", idx)
print("날짜:", record.get("MTG_DE"))
print("의회:", record.get("RASMBLY_NM"))

# 회의록 제목
for selector in [
    "#minutes-title-line",
    "#agenda-block",
    "#item-block"
]:
    elements = soup.select(selector)

    print(f"\n[{selector}] {len(elements)}개")

    for element in elements[:3]:
        print(element.get_text(" ", strip=True)[:500])

# 기존 안건 구분자
items = soup.select("strong.item-in-contents")
print("\n기존 안건 구분자:", len(items))

# 발언 블록
speakers = soup.select("div.contents-block.speaker-block")
print("발언 블록:", len(speakers))

# 안건 관련 CSS 후보 확인
print("\n===== 안건 관련 태그 후보 =====")

keywords = [
    "안건", "감사", "업무보고", "감사계획"
]

for tag in soup.find_all(["div", "p", "strong", "span", "li"]):

    text = tag.get_text(" ", strip=True)

    # 긴 본문 컨테이너는 제외
    if not (5 <= len(text) <= 150):
        continue

    if any(keyword in text for keyword in keywords):

        print(
            "tag:", tag.name,
            "| class:", tag.get("class"),
            "| id:", tag.get("id"),
            "| text:", text[:100]
        )

        # 출력이 지나치게 길어지는 것 방지
        # 아래에서 최대 20개만 확인하도록 개선 가능
       

