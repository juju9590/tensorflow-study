# 근거 원문 추적 검증

from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
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

embedding_dir = (
    base_dir / "processed"
    / "embeddings_by_region" / year / region
)

output_dir = (
    base_dir / "processed"
    / "validation_samples"
    / "evidence_validation" / year / region
)

output_dir.mkdir(parents=True, exist_ok=True)

csv_files = sorted(embedding_dir.glob("*metadata.csv"))

if not csv_files:
    raise FileNotFoundError("메타데이터 CSV 없음")

# =====================================================
# 2. 원본 JSON 로드
# =====================================================

with open(json_path, "r", encoding="utf-8") as f:
    records = json.load(f)

print("원본 회의록:", len(records))
print("배치 CSV:", len(csv_files))

# =====================================================
# 3. 텍스트 정규화 및 위치 매핑
# =====================================================

def normalize(text):
    text = str(text or "").replace("\\n", "\n")
    return re.sub(r"\s+", "", text)


def build_index(text):
    """
    공백을 제거한 문자열과
    원래 문자열의 문자 위치를 함께 생성
    """
    normalized = []
    positions = []

    for i, char in enumerate(text):
        if not char.isspace():
            normalized.append(char)
            positions.append(i)

    return "".join(normalized), positions

# =====================================================
# 4. 원본 회의록 발언 텍스트 구성
# =====================================================

def extract_source(record):

    soup = BeautifulSoup(
        record.get("MINTS_HTML", ""),
        "html.parser"
    )

    blocks = soup.select(
        "div.contents-block.speaker-block"
    )

    if not blocks:
        return "", "", []

    texts = [
        block.get_text(" ", strip=True)
        for block in blocks
    ]

    # 발언 순서 유지
    source_text = "\n".join(texts)

    normalized, positions = build_index(source_text)

    return source_text, normalized, positions


# 회의록 280개를 한 번만 전처리
source_cache = {
    i: extract_source(record)
    for i, record in enumerate(records)
}

# =====================================================
# 5. 청크 검사 지점 생성
# =====================================================

def get_anchors(text, size=80, count=5):

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
# 6. 근거 원문 검증
# =====================================================

def validate_evidence(chunk_text, source_data):

    source_text, normalized_source, positions = source_data

    anchors = get_anchors(chunk_text)

    if not anchors or not normalized_source:
        return "UNMATCHED", "", 0, 0, -1, -1

    matched = []

    for anchor in anchors:

        pos = normalized_source.find(anchor)

        if pos >= 0:
            matched.append((anchor, pos))

    total = len(anchors)
    matched_count = len(matched)

    if matched_count == 0:
        status = "UNMATCHED"

    elif matched_count == total:
        status = "ALL_ANCHORS_MATCH"

    else:
        status = "PARTIAL_MATCH"

    evidence = ""
    start_offset = -1
    end_offset = -1

    if matched:
        # 최초 일치 구간을 근거 후보로 추출
        anchor, pos = matched[0]

        start_offset = positions[pos]
        end_offset = positions[pos + len(anchor) - 1] + 1

        # 원본에 존재하는 실제 텍스트 구간
        evidence = source_text[start_offset:end_offset]

        # 180자 제한
        if len(evidence) > 180:
            evidence = evidence[:180]
            end_offset = start_offset + len(evidence)

        # 추출 문구 원문 일치 재확인
        assert (
            source_text[start_offset:end_offset]
            == evidence
        )

    return (
        status,
        evidence,
        matched_count,
        total,
        start_offset,
        end_offset
    )

# =====================================================
# 7. 전체 청크 검증
# =====================================================

results = []
stats = Counter()

for file_no, file in enumerate(csv_files, 1):

    file_count = 0

    for df in pd.read_csv(
        file,
        dtype=str,
        keep_default_na=False,
        chunksize=2000
    ):

        required = {
            "record_index", "date", "assembly",
            "chunk_id", "text"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"필수 컬럼 없음: {file.name}"
            )

        for _, row in df.iterrows():

            file_count += 1

            try:
                idx = int(row["record_index"])
            except (ValueError, TypeError):
                idx = -1

            status = "INVALID_INDEX"
            evidence = ""
            matched_count = 0
            total_anchors = 0
            start = -1
            end = -1

            if 0 <= idx < len(records):

                record = records[idx]

                source_ok = (
                    row["date"].strip()
                    == str(record.get("MTG_DE", "")).strip()
                    and
                    row["assembly"].strip()
                    == str(record.get("RASMBLY_NM", "")).strip()
                )

                if source_ok:
                    (
                        status,
                        evidence,
                        matched_count,
                        total_anchors,
                        start,
                        end
                    ) = validate_evidence(
                        row["text"],
                        source_cache[idx]
                    )
                else:
                    status = "SOURCE_MISMATCH"

            stats[status] += 1

            results.append({
                "batch_file": file.name,
                "date": row["date"],
                "assembly": row["assembly"],
                "meeting_title": row.get(
                    "meeting_title", ""
                ),
                "source_id": row.get("source_id", ""),
                "record_index": row["record_index"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row.get(
                    "chunk_index", ""
                ),
                "validation_status": status,
                "matched_anchors": matched_count,
                "total_anchors": total_anchors,
                "match_ratio": (
                    round(matched_count / total_anchors, 4)
                    if total_anchors else 0
                ),
                "evidence_text": evidence,
                "evidence_length": len(evidence),
                "start_offset": start,
                "end_offset": end,
                "text": row["text"]
            })

    print(
        f"[{file_no}/{len(csv_files)}] "
        f"{file.name}: {file_count:,}개"
    )

# =====================================================
# 8. 결과 저장
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    output_dir / "evidence_validation_all.csv",
    index=False,
    encoding="utf-8-sig"
)

review_df = result_df[
    result_df["validation_status"]
    != "ALL_ANCHORS_MATCH"
]

review_df.to_csv(
    output_dir / "evidence_validation_review.csv",
    index=False,
    encoding="utf-8-sig"
)

summary_df = pd.DataFrame([
    {
        "validation_status": status,
        "count": count,
        "percentage": round(
            count / len(result_df) * 100, 2
        ) if len(result_df) else 0
    }
    for status, count in sorted(stats.items())
])

summary_df.to_csv(
    output_dir / "evidence_validation_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. 결과 출력
# =====================================================

print("\n===== 원문 근거 검증 결과 =====")
print("전체 청크:", len(result_df))
print(summary_df.to_string(index=False))

print("\n근거 문구 추출:",
      (result_df["evidence_text"] != "").sum())

print("180자 초과 근거:",
      (result_df["evidence_length"] > 180).sum())

print("\n결과 저장 위치:", output_dir)
print("검증 완료!")
