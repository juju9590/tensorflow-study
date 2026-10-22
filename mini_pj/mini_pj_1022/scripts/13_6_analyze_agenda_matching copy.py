from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd
import json

# 1. 경로
json_path = Path(
    r"C:\mini_pj\data\org2\2024_광역 회의록"
    r"\2024_강원특별자치도의회.json"
)

output_dir = Path(
    r"C:\mini_pj\data\processed\validation_samples"
    r"\agenda_matching\2024\강원특별자치도의회"
)
output_dir.mkdir(parents=True, exist_ok=True)

# 2. 원본 로드
with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

results = []

# 3. 회의록 전체 유형 분석
for idx, record in enumerate(records):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    session = soup.select_one("p.session-line")
    title = soup.select_one("#minutes-title-line")
    inspection = soup.select_one("#inspection-line")

    agenda_count = len(
        soup.select("strong.item-in-contents")
    )

    session_text = (
        session.get_text(" ", strip=True)
        if session else ""
    )

    if "행정사무감사" in session_text:
        meeting_type = "행정사무감사"
    elif agenda_count > 0:
        meeting_type = "일반 안건 회의"
    else:
        meeting_type = "기타 / 추가 검증"

    results.append({
        "record_index": idx,
        "date": record.get("MTG_DE", ""),
        "meeting_type": meeting_type,
        "meeting_title": (
            title.get_text(" ", strip=True)
            if title else ""
        ),
        "session_title": session_text,
        "agenda_count": agenda_count,
        "audit_target": (
            inspection.get_text(" ", strip=True)
            if inspection else ""
        )
    })

# 4. 결과 저장
df = pd.DataFrame(results)

df.to_csv(
    output_dir / "meeting_type_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# 5. 유형별 통계
print("\n===== 회의록 유형별 현황 =====")
print(df["meeting_type"].value_counts())

print("\n===== 행정사무감사 현황 =====")
audit_df = df[df["meeting_type"] == "행정사무감사"]

print("행정사무감사 회의록:", len(audit_df))
print("감사 대상 기관 확인:", audit_df["audit_target"].ne("").sum())

print("\n===== 기타 검증 대상 =====")
other = df[df["meeting_type"] == "기타 / 추가 검증"]

print(
    other[
        ["record_index", "date", "meeting_title", "session_title"]
    ].head(20).to_string(index=False)
)

print("\n분석 완료!")