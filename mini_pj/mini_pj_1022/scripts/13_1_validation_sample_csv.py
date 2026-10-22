from pathlib import Path
import pandas as pd

# 1. 경로 설정
input_dir = Path(r"C:\mini_pj\data\processed\embeddings_by_region")
output_dir = Path(r"C:\mini_pj\data\processed\validation_samples")

output_dir.mkdir(parents=True, exist_ok=True)

# 2. 하위 폴더까지 CSV 검색
files = sorted(input_dir.rglob("*.csv"))
#지역별 폴더가 있기 때문에, 기존 코드의 glob("*.csv")를 rglob("*.csv")로 변경

print(f"검색된 CSV 파일: {len(files)}개")

# 3. 파일별 랜덤 표본 추출
for file in files:
    df = pd.read_csv(file)

    sample = df.sample(
        n=min(100, len(df)),
        random_state=42
    ).copy()

    # 임베딩 벡터 컬럼 제외
    embedding_cols = [
        col for col in sample.columns
        if col.lower() in ["embedding", "embeddings", "vector"]
    ]

    sample = sample.drop(columns=embedding_cols)

    # 4. 연도 및 지역 폴더명으로 파일명 생성
    relative_path = file.relative_to(input_dir)
    file_name = "_".join(relative_path.parts)

    save_path = output_dir / f"sample_{file_name}"

    sample.to_csv(
        save_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[완료] {file_name}: {len(sample)}개")

print("\n전체 표본 추출 완료!")