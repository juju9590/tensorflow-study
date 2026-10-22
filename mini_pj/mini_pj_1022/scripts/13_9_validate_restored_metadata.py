
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import pandas as pd
import json
import re

# =====================================================
# 1. 경로 설정
# =====================================================

base_dir = Path(r"C:\mini_pj\data")

year = "2024"
region = "강원특별자치도의회"

json_path = (
    base_dir / "org2"
    / f"{year}_광역 회의록"
    / f"{year}_{region}.json"
)

validation_dir = (
    base_dir / "processed" / "validation_samples"
    / "agenda_matching" / year / region
)

input_path = (
    validation_dir / "restored"
    / "metadata_restore_candidates.csv"
)

output_dir = validation_dir / "verified"
output_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2. 원본 불러오기
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

df = pd.read_csv(
    input_path,
    dtype=str,
    keep_default_na=False
)

print("원본 회의록:", len(records))
print("검증 대상 청크:", len(df))

# =====================================================
# 3. 텍스트 정규화
# =====================================================

def normalize(text):
    text = str(text or "").replace("\\n", " ")
    return re.sub(r"\s+", "", text)

# =====================================================
# 4. 원본에서 안건별 발언 추출
# =====================================================

def extract_agenda_groups(record):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    groups = defaultdict(list)
    current_agenda = "안건 미배정"

    for element in soup.select(
        "strong.item-in-contents, "
        "div.contents-block.speaker-block"
    ):
        if element.name == "strong":
            current_agenda = element.get_text(" ", strip=True)

        elif element.name == "div":
            speech = normalize(
                element.get_text(" ", strip=True)
            )
            if speech:
                groups[current_agenda].append(speech)

    return {
        agenda: "\u241E".join(speeches)
        for agenda, speeches in groups.items()
    }

agenda_cache = {
    i: extract_agenda_groups(record)
    for i, record in enumerate(records)
}

# =====================================================
# 5. 청크 내부 검사 위치
# =====================================================

def get_anchors(text, size=45, count=7):

    text = normalize(text)

    if not text:
        return []

    if len(text) <= size:
        return [text]

    max_start = len(text) - size

    positions = [
        round(max_start * i / (count - 1))
        for i in range(count)
    ]

    return list(dict.fromkeys(
        text[p:p + size] for p in positions
    ))

# =====================================================
# 6. 안건 연결 정확성 검증
# =====================================================

def verify_agenda(text, candidate, groups):

    anchors = get_anchors(text)

    if not anchors or candidate not in groups:
        return "REVIEW", 0, 0, 0

    matched = 0
    other_agenda_hits = 0
    unmatched = 0

    for anchor in anchors:

        found = [
            agenda
            for agenda, group_text in groups.items()
            if anchor in group_text
        ]

        if candidate in found:
            matched += 1

        if any(a != candidate for a in found):
            other_agenda_hits += 1

        if not found:
            unmatched += 1

    ratio = matched / len(anchors)

    if other_agenda_hits > 0:
        status = "REVIEW_MULTI"

    elif matched == len(anchors):
        status = "STRONG_MATCH"

    elif ratio >= 0.7:
        status = "PARTIAL_MATCH"

    else:
        status = "REVIEW_LOW"

    return (
        status,
        round(ratio, 4),
        other_agenda_hits,
        unmatched
    )

# =====================================================
# 7. 전체 후보 검증
# =====================================================

results = []
stats = Counter()

for _, row in df.iterrows():

    restore_status = row["restore_status"]

    if restore_status != "AGENDA_CANDIDATE":
        result = row.to_dict()
        result["verification_status"] = "NOT_APPLICABLE"
        result["verified_ratio"] = ""
        result["other_agenda_hits"] = ""
        result["unmatched_anchors"] = ""
        results.append(result)
        continue

    try:
        idx = int(row["record_index"])
    except (ValueError, TypeError):
        idx = -1

    if not (0 <= idx < len(records)):
        status, ratio, other, unmatched = (
            "INVALID_INDEX", 0, 0, 0
        )
    else:
        record = records[idx]

        source_ok = (
            str(row["date"]).strip()
            == str(record.get("MTG_DE", "")).strip()
            and
            str(row["assembly"]).strip()
            == str(record.get("RASMBLY_NM", "")).strip()
        )

        if not source_ok:
            status, ratio, other, unmatched = (
                "SOURCE_MISMATCH", 0, 0, 0
            )
        else:
            status, ratio, other, unmatched = verify_agenda(
                row["text"],
                row["agenda_title"],
                agenda_cache[idx]
            )

    stats[status] += 1

    result = row.to_dict()
    result["verification_status"] = status
    result["verified_ratio"] = ratio
    result["other_agenda_hits"] = other
    result["unmatched_anchors"] = unmatched
    results.append(result)

# =====================================================
# 8. 결과 저장
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_dir / "metadata_verification_all.csv",
    index=False,
    encoding="utf-8-sig"
)

# 추가 검토 대상
review_df = result_df[
    result_df["verification_status"].isin([
        "REVIEW_MULTI",
        "REVIEW_LOW",
        "PARTIAL_MATCH",
        "REVIEW",
        "INVALID_INDEX",
        "SOURCE_MISMATCH"
    ])
]

review_df.to_csv(
    output_dir / "metadata_verification_review.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame([
    {
        "verification_status": status,
        "count": count,
        "percentage": round(
            count / sum(stats.values()) * 100, 2
        )
    }
    for status, count in sorted(stats.items())
])

summary_df.to_csv(
    output_dir / "metadata_verification_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. 회의록별 검토 표본 추출
# =====================================================

# 후보 100개: 회의록별 1개씩 우선 추출
candidate_df = result_df[
    result_df["restore_status"] == "AGENDA_CANDIDATE"
]

sample = (
    candidate_df.groupby("record_index", group_keys=False)
    .sample(n=1, random_state=42)
    .head(100)
)

sample.to_csv(
    output_dir / "metadata_manual_review_sample.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 10. 결과 출력
# =====================================================

print("\n===== 안건명 후보 검증 결과 =====")
print(summary_df.to_string(index=False))

print("\n수동 검토 표본:", len(sample))
print("\n결과 저장:", output_dir)
print("검증 완료!")
