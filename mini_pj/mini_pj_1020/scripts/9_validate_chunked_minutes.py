# 청킹 데이터 검증 (임베딩 전에 다음 5가지만 확인 필요)
# 1. chunk_id 중복이 없는가?
# 2. text가 비어 있는 청크는 없는가?
# 3. 청크 길이가 설정한 1,000자를 초과하지 않는가?
# 4. date, assembly, source_id가 유지됐는가?
# 5. 각 지역의 원본 회의록이 빠짐없이 청킹됐는가?
# ㄴ 이 검증이 통과하면 본격적으로 임베딩과 정책 이슈 군집화로 진행가능


import csv
import sys
from pathlib import Path

# ==================================================
# 1. 경로 설정
# ==================================================

ORIGINAL_DIR = Path("C:/tensorflow_study/mini_pj/data/processed/by_region")
CHUNK_DIR = Path("C:/tensorflow_study/mini_pj/data/processed/chunked_by_region")

original_files = sorted(ORIGINAL_DIR.glob("*.csv"))
chunk_files = sorted(CHUNK_DIR.glob("*_chunked.csv"))

print("원본 지역 파일:", len(original_files))
print("청킹 지역 파일:", len(chunk_files))

# 긴 텍스트 필드 읽기 허용
limit = sys.maxsize

while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10


# ==================================================
# 2. 원본 회의록 ID 확인
# ==================================================

original_ids = set()

for file_path in original_files:

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            original_ids.add(row["source_id"])

print("원본 회의록 수:", len(original_ids))


# ==================================================
# 3. 청킹 데이터 검증
# ==================================================

chunk_ids = set()
chunk_source_ids = set()

total_chunks = 0
empty_text = 0
over_1000 = 0
missing_metadata = 0
duplicate_ids = 0

min_length = float("inf")
max_length = 0
total_length = 0

region_results = []

for file_path in chunk_files:

    region_chunks = 0
    region_errors = 0

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            total_chunks += 1
            region_chunks += 1

            text = row.get("text", "")
            chunk_id = row.get("chunk_id", "")
            source_id = row.get("source_id", "")

            # ① 빈 본문
            if not text.strip():
                empty_text += 1
                region_errors += 1

            # ② 청크 길이
            text_length = len(text)

            if text_length > 1000:
                over_1000 += 1
                region_errors += 1

            min_length = min(min_length, text_length)
            max_length = max(max_length, text_length)
            total_length += text_length

            # ③ 필수 메타데이터
            required = [
                "date",
                "assembly",
                "source_id",
                "chunk_id",
                "chunk_index"
            ]

            if any(not row.get(col, "").strip() for col in required):
                missing_metadata += 1
                region_errors += 1

            # ④ 청크 ID 중복
            if chunk_id in chunk_ids:
                duplicate_ids += 1
                region_errors += 1

            chunk_ids.add(chunk_id)

            # ⑤ 원본 회의록 연결
            chunk_source_ids.add(source_id)

    region_results.append(
        (file_path.name, region_chunks, region_errors)
    )

    print(
        f"{file_path.stem}: "
        f"{region_chunks:,}개 / "
        f"오류 {region_errors}건"
    )


# ==================================================
# 4. 원본 회의록과 비교
# ==================================================

missing_sources = original_ids - chunk_source_ids
unknown_sources = chunk_source_ids - original_ids

average_length = (
    total_length / total_chunks if total_chunks else 0
)

if total_chunks == 0:
    min_length = 0


# ==================================================
# 5. 최종 결과
# ==================================================

print("\n===== 전체 청킹 검증 결과 =====")

print("원본 회의록:", len(original_ids))
print("전체 청크:", f"{total_chunks:,}")
print("청크 ID 중복:", duplicate_ids)
print("빈 청크:", empty_text)
print("1000자 초과:", over_1000)
print("필수 메타데이터 결측:", missing_metadata)

print("\n===== 청크 길이 =====")
print("최소:", min_length)
print("평균:", round(average_length, 2))
print("최대:", max_length)

print("\n===== 원본 연결 검증 =====")
print("청킹되지 않은 원본:", len(missing_sources))
print("원본에 없는 source_id:", len(unknown_sources))

print("\n===== 최종 판단 =====")

passed = (
    len(original_files) == len(chunk_files) == 17
    and len(original_ids) == 6007
    and total_chunks == 402683
    and duplicate_ids == 0
    and empty_text == 0
    and over_1000 == 0
    and missing_metadata == 0
    and len(missing_sources) == 0
    and len(unknown_sources) == 0
)

if passed:
    print("기본 구조 검증 통과")
else:
    print("검토 필요")

if missing_sources:
    print("\n누락된 원본 ID 예시:")
    print(list(missing_sources)[:10])

if unknown_sources:
    print("\n존재하지 않는 원본 ID 예시:")
    print(list(unknown_sources)[:10])


# 원본 지역 파일: 17
# 청킹 지역 파일: 17
# 원본 회의록 수: 6007
# 2025_강원특별자치도의회_chunked: 35,649개 / 오류 0건
# 2025_경기도의회_chunked: 23,798개 / 오류 0건
# 2025_경상남도의회_chunked: 17,792개 / 오류 0건
# 2025_경상북도의회_chunked: 23,761개 / 오류 0건
# 2025_광주광역시의회_chunked: 8,357개 / 오류 0건
# 2025_대구광역시의회_chunked: 16,412개 / 오류 0건
# 2025_대전광역시의회_chunked: 8,568개 / 오류 0건
# 2025_부산광역시의회_chunked: 17,603개 / 오류 0건
# 2025_서울특별시의회_chunked: 61,760개 / 오류 0건
# 2025_세종특별자치시의회_chunked: 12,631개 / 오류 0건
# 2025_울산광역시의회_chunked: 15,823개 / 오류 0건
# 2025_인천광역시의회_chunked: 26,034개 / 오류 0건
# 2025_전라남도의회_chunked: 43,577개 / 오류 0건
# 2025_전북특별자치도의회_chunked: 14,569개 / 오류 0건
# 2025_제주특별자치도의회_chunked: 26,375개 / 오류 0건
# 2025_충청남도의회_chunked: 38,087개 / 오류 0건
# 2025_충청북도의회_chunked: 11,887개 / 오류 0건

# ===== 전체 청킹 검증 결과 =====
# 원본 회의록: 6007
# 전체 청크: 402,683
# 청크 ID 중복: 0
# 빈 청크: 0
# 1000자 초과: 0
# 필수 메타데이터 결측: 0

# ===== 청크 길이 =====
# 최소: 8
# 평균: 933.99
# 최대: 1000

# ===== 원본 연결 검증 =====
# 청킹되지 않은 원본: 0
# 원본에 없는 source_id: 0

# ===== 최종 판단 =====
# 기본 구조 검증 통과