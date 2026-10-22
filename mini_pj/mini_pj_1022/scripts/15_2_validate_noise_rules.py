# 순서	검증 내용	목적
# ①	유형별 탐지 문구 빈도	가장 많이 걸린 문구 확인
# ②	MIXED 유형 상세 분석	실제 혼합 청크인지 확인
# ③	UI 문구 24개	제거 가능한지 확인
# ④	출석자 명단 333개	발언과 혼합됐는지 구분
# ⑤	지역별 오탐 검증	잘못 분류된 정상 발언 보호
# ⑥	정제 규칙 확정	부분 재임베딩 대상 결정


from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import json
import ast
import re

# =====================================================
# 1. 경로 설정
# =====================================================

BASE_DIR = Path(r"C:\mini_pj\data\processed")

INPUT_DIR = BASE_DIR / "noise_validation"
OUTPUT_DIR = INPUT_DIR / "rule_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.rglob("*_noise_check.csv"))

print("검증 파일:", len(files))

if not files:
    raise FileNotFoundError("의심 청크 CSV가 없습니다.")

# =====================================================
# 2. 패턴 정보 복원
# =====================================================

def parse_patterns(value):
    if pd.isna(value) or not str(value).strip():
        return {}

    try:
        data = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        try:
            data = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return {}

    return data if isinstance(data, dict) else {}


# =====================================================
# 3. 탐지 문구 주변 문맥 추출
# =====================================================

def extract_context(text, pattern, window=100):

    if not pattern:
        return ""

    pos = text.find(pattern)

    if pos < 0:
        return ""

    start = max(0, pos - window)
    end = min(len(text), pos + len(pattern) + window)

    return text[start:end].replace("\n", " ")


# =====================================================
# 4. 의심 청크 재분석
# =====================================================

results = []
pattern_count = Counter()
pattern_occurrences = Counter()
region_patterns = Counter()
category_count = Counter()
subtype_count = Counter()

for file_no, file in enumerate(files, 1):

    relative = file.relative_to(INPUT_DIR)
    year = relative.parts[0]
    region = relative.parts[1]

    file_count = 0

    for df in pd.read_csv(
        file,
        dtype=str,
        keep_default_na=False,
        chunksize=2000
    ):

        required = {
            "chunk_id", "text", "noise_type",
            "detected_patterns"
        }

        if not required.issubset(df.columns):
            raise ValueError(
                f"필수 컬럼 누락: {file}"
            )

        for row in df.to_dict("records"):

            text = row["text"]
            patterns = parse_patterns(
                row["detected_patterns"]
            )

            categories = sorted(patterns.keys())

            # 한 청크에서 실제 탐지된 서로 다른
            # 문구의 개수
            unique_patterns = {
                (category, str(pattern))
                for category, items in patterns.items()
                for pattern in items
            }

            pattern_types = len(unique_patterns)
            category_types = len(categories)

            # MIXED 유형 상세 분류
            if row["noise_type"] == "MIXED":

                if category_types > 1:
                    subtype = "MULTIPLE_CATEGORIES"

                elif pattern_types > 1:
                    subtype = "MULTIPLE_PATTERNS"

                else:
                    subtype = "SINGLE_PATTERN_LONG_TEXT"

            else:
                subtype = row["noise_type"]

            # 대표 탐지 문구
            first_category = ""
            first_pattern = ""

            if unique_patterns:
                first_category, first_pattern = sorted(
                    unique_patterns
                )[0]

            context = extract_context(
                text, first_pattern, window=100
            )

            # 문구별 빈도
            # 청크 빈도와 총 출현 횟수 구분
            for category, pattern in unique_patterns:

                key = (category, pattern)

                pattern_count[key] += 1

                pattern_occurrences[key] += text.count(
                    pattern
                )

                region_patterns[
                    (year, region, category, pattern)
                ] += 1

            category_count[row["noise_type"]] += 1
            subtype_count[subtype] += 1

            # 자동 삭제하지 않고 검토 목적만 분류
            if "UI_NOISE" in categories:
                review_group = "UI_REVIEW"

            elif "HTML_RESIDUE" in categories:
                review_group = "HTML_REVIEW"

            elif "ATTENDANCE" in categories:
                review_group = "ATTENDANCE_REVIEW"

            elif "HEADER_AGENDA" in categories:
                review_group = "HEADER_REVIEW"

            else:
                review_group = "OTHER_REVIEW"

            results.append({
                "year": year,
                "region": region,
                "batch_file": row.get(
                    "batch_file", ""
                ),
                "row_index": row.get(
                    "row_index", ""
                ),
                "chunk_id": row["chunk_id"],
                "source_id": row.get(
                    "source_id", ""
                ),
                "record_index": row.get(
                    "record_index", ""
                ),
                "date": row.get("date", ""),
                "noise_type": row["noise_type"],
                "noise_subtype": subtype,
                "category_count": category_types,
                "pattern_count": pattern_types,
                "categories": "|".join(categories),
                "review_group": review_group,
                "first_category": first_category,
                "first_pattern": first_pattern,
                "context_100": context,
                "text_length": len(text),
                "text": text
            })

            file_count += 1

    print(
        f"[{file_no}/{len(files)}] "
        f"{year}/{region} "
        f"| 의심 {file_count:,}개"
    )

# =====================================================
# 5. 전체 상세 결과
# =====================================================

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT_DIR / "noise_detail_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n전체 의심 청크:", len(result_df))

# =====================================================
# 6. 탐지 문구 빈도 TOP 100
# =====================================================

pattern_rows = []

for (category, pattern), count in pattern_count.items():

    pattern_rows.append({
        "category": category,
        "pattern": pattern,
        "chunk_count": count,
        "occurrence_count": pattern_occurrences[
            (category, pattern)
        ]
    })

pattern_df = pd.DataFrame(pattern_rows)

if not pattern_df.empty:
    pattern_df = pattern_df.sort_values(
        "chunk_count", ascending=False
    )

pattern_df.to_csv(
    OUTPUT_DIR / "noise_pattern_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 탐지 문구 TOP 30 =====")
print(
    pattern_df.head(30).to_string(index=False)
    if not pattern_df.empty else "탐지 문구 없음"
)

# =====================================================
# 7. MIXED 유형 상세 분석
# =====================================================

mixed_df = result_df[
    result_df["noise_type"] == "MIXED"
]

mixed_summary = (
    mixed_df["noise_subtype"]
    .value_counts()
    .rename_axis("subtype")
    .reset_index(name="count")
)

mixed_summary.to_csv(
    OUTPUT_DIR / "noise_mixed_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== MIXED 상세 분석 =====")
print(mixed_summary.to_string(index=False))

# =====================================================
# 8. 지역별 탐지 문구 빈도
# =====================================================

region_rows = [
    {
        "year": year,
        "region": region,
        "category": category,
        "pattern": pattern,
        "chunk_count": count
    }
    for (
        year, region, category, pattern
    ), count in region_patterns.items()
]

region_df = pd.DataFrame(region_rows)

if not region_df.empty:
    region_df = region_df.sort_values(
        "chunk_count", ascending=False
    )

region_df.to_csv(
    OUTPUT_DIR / "noise_patterns_by_region.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 9. 검토용 파일 생성
# =====================================================

# 전체 의심 청크를 검토할 수 있도록
# 원문 전체 대신 대표 문맥 중심 파일 생성
review_cols = [
    "year", "region", "batch_file",
    "row_index", "chunk_id",
    "noise_type", "noise_subtype",
    "categories", "review_group",
    "first_pattern", "context_100",
    "text_length"
]

review_df = result_df[review_cols].copy()

review_df.to_csv(
    OUTPUT_DIR / "noise_manual_review.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 10. 최종 통계
# =====================================================

print("\n===== 잡음 분류별 건수 =====")

for key, value in category_count.most_common():
    print(f"{key}: {value:,}")

print("\n===== 검토 그룹별 건수 =====")

print(
    result_df["review_group"]
    .value_counts()
    .to_string()
)

print("\n===== 검증 완료 =====")
print("분석 대상:", f"{len(result_df):,}")
print("결과 저장:", OUTPUT_DIR)
print("원본 CSV 및 NPY 변경 없음")


