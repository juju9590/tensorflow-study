# 청크 내용 점검 코드
# 1. 100자 미만, 100~299자, 300자 이상 청크 수
# 2. 전체에서 가장 짧은 청크 20개
# 3. 17개 지역에서 무작위 청크 3개씩



import csv
import sys
import random
import heapq
from pathlib import Path

# ==================================================
# 1. 경로 설정
# ==================================================

INPUT_DIR = Path("C:/tensorflow_study/mini_pj/data/processed/chunked_by_region")
OUTPUT_PATH = Path("C:/tensorflow_study/mini_pj/data/processed/chunk_review_samples.csv")

csv_files = sorted(INPUT_DIR.glob("*_chunked.csv"))

random.seed(777)

# 긴 텍스트 필드 읽기 허용
limit = sys.maxsize

while True:
    try:
        csv.field_size_limit(limit)
        break
    except OverflowError:
        limit //= 10


# ==================================================
# 2. 청크 점검
# ==================================================

shortest = []
random_samples = []

total = 0
under_100 = 0
under_300 = 0
over_300 = 0

for file_path in csv_files:

    region_count = 0
    region_samples = []

    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            total += 1
            region_count += 1

            text = row["text"]
            length = len(text)

            row["text_length"] = length

            # ① 길이별 통계
            if length < 100:
                under_100 += 1

            elif length < 300:
                under_300 += 1

            else:
                over_300 += 1

            # ② 전체에서 가장 짧은 20개 보관
            item = (-length, total, row.copy())

            if len(shortest) < 20:
                heapq.heappush(shortest, item)

            elif length < -shortest[0][0]:
                heapq.heapreplace(shortest, item)

            # ③ 지역별 무작위 3개 추출
            if len(region_samples) < 3:
                region_samples.append(row.copy())

            else:
                idx = random.randrange(region_count)

                if idx < 3:
                    region_samples[idx] = row.copy()

    random_samples.extend(region_samples)

    print(
        f"{file_path.stem}: "
        f"{region_count:,}개 확인 완료"
    )


# ==================================================
# 3. 검토 표본 정리
# ==================================================

review_data = []

# 짧은 청크 20개
for _, _, row in sorted(shortest, key=lambda x: -x[0]):
    row["review_type"] = "짧은 청크"
    review_data.append(row)

# 지역별 무작위 청크
for row in random_samples:
    row["review_type"] = "지역별 표본"
    review_data.append(row)


# ==================================================
# 4. CSV 저장
# ==================================================

columns = [
    "review_type",
    "assembly",
    "date",
    "source_id",
    "chunk_id",
    "chunk_index",
    "text_length",
    "text"
]

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=columns,
        extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(review_data)


# ==================================================
# 5. 결과 출력
# ==================================================

print("\n===== 청크 길이 분포 =====")
print("전체 청크:", f"{total:,}")
print("100자 미만:", f"{under_100:,}")
print("100~299자:", f"{under_300:,}")
print("300자 이상:", f"{over_300:,}")

print("\n===== 가장 짧은 청크 10개 =====")

for _, _, row in sorted(shortest, key=lambda x: -x[0])[:10]:

    print("\n지역:", row["assembly"])
    print("길이:", row["text_length"])
    print("내용:", repr(row["text"][:250]))

print("\n===== 표본 저장 =====")
print("선정 표본:", len(review_data))
print("저장 위치:", OUTPUT_PATH)
