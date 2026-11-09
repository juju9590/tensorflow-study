# 6번에서 추출된 2025_all_minutes_baseline.csv를 지역별로 csv파일로 분리

###########################################
#  추출구조
# #########################################

# 2025_all_minutes_baseline.csv (기존 통합 원본)

# by_region/   # … 총 17개 지역
# ㄴ 2025_강원특별자치도의회.csv
# ㄴ 2025_경기도의회.csv
# ㄴ 2025_경상남도의회.csv

# chunked_by_region/    # … 총 17개 청킹 결과
# 2025_강원특별자치도의회_chunked.csv
# 2025_경기도의회_chunked.csv

###################################################

import pandas as pd
from pathlib import Path

# 1. 경로
input_path = Path(
    "C:/tensorflow_study/mini_pj/data/processed/2025_all_minutes_baseline.csv"
)

output_dir = Path(
    "C:/tensorflow_study/mini_pj/data/processed/by_region"
)

output_dir.mkdir(parents=True, exist_ok=True)

# 2. 통합 CSV 불러오기
df = pd.read_csv(input_path, dtype={"date": "string"})

print("전체 회의록:", len(df))

# 3. 지역별 저장
for assembly, group in df.groupby("assembly"):

    file_path = output_dir / f"2025_{assembly}.csv"

    group.to_csv(
        file_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"{assembly}: {len(group)}건 저장 완료")

print("\n전체 지역:", df["assembly"].nunique())
print("저장된 회의록:", len(df))

