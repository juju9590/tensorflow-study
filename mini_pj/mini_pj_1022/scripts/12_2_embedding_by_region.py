# 항목	설정
# GPU	집 데스크톱 RTX 3060 12GB
# 모델	dragonkue/snowflake-arctic-embed-l-v2.0-ko
# 입력	지역별 청킹 CSV 34개
# 전체 대상	684,311개 청크
# 모델 내부 batch_size	16부터 시작
# 파일 저장 단위	5,000개 청크
# 벡터	1,024차원 float32
# 중간 재시작	완료된 배치 자동 건너뛰기
# 메타데이터	기존 14개 컬럼 보존



from pathlib import Path
import os
import json
import time
import hashlib

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. 기본 설정
# ============================================================

# scripts 폴더의 상위 폴더 = 프로젝트 폴더
PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_DIR / "data" / "processed" / "chunked_by_region"
)

# 저장 공간이 부족하면 이 경로만 외장하드로 변경
OUTPUT_DIR = (
    PROJECT_DIR / "data" / "processed" / "embeddings_by_region"
)

MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"

BATCH_SIZE = 16          # GPU에서 한 번에 처리할 청크 수
SAVE_SIZE = 5000         # 저장 파일당 청크 수
MAX_SEQ_LENGTH = 1024    # 모델 입력 최대 토큰 길이
EXPECTED_DIM = 1024

YEAR_FOLDERS = {
    2024: "2024_광역회의록",
    2025: "2025_광역회의록"
}

EXPECTED_COUNTS = {
    2024: 281628,
    2025: 402683
}

REQUIRED_COLS = [
    "date", "assembly", "source_id",
    "chunk_id", "chunk_index", "text"
]


# ============================================================
# 2. GPU 확인
# ============================================================

print("===== GPU 확인 =====")
print("PyTorch:", torch.__version__)
print("CUDA 사용 가능:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU를 사용할 수 없습니다. GPU 환경을 먼저 확인하세요."
    )

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

device = "cuda"


# ============================================================
# 3. 입력 파일 확인
# ============================================================

input_files = []

for year, folder_name in YEAR_FOLDERS.items():

    folder = INPUT_DIR / folder_name

    if not folder.exists():
        raise FileNotFoundError(f"폴더 없음: {folder}")

    files = sorted(folder.glob("*_chunked.csv"))

    if len(files) != 17:
        raise ValueError(
            f"{year}년 파일이 17개가 아닙니다: {len(files)}개"
        )

    for file_path in files:
        input_files.append((year, file_path))

print("\n입력 파일 수:", len(input_files))
print("전체 예상 청크:", f"{sum(EXPECTED_COUNTS.values()):,}")
print("저장 경로:", OUTPUT_DIR)


# ============================================================
# 4. 임베딩 모델 로드
# ============================================================

print("\n===== 임베딩 모델 로드 =====")

model = SentenceTransformer(
    MODEL_NAME,
    device=device
)

# 12GB GPU에서 메모리 사용량 제어
model.max_seq_length = MAX_SEQ_LENGTH

print("모델:", MODEL_NAME)
print("입력 최대 토큰:", model.max_seq_length)
print("임베딩 차원:", model.get_sentence_embedding_dimension())

if model.get_sentence_embedding_dimension() != EXPECTED_DIM:
    raise ValueError("임베딩 차원이 예상과 다릅니다.")


# ============================================================
# 5. 배치 내용 해시 생성
# ============================================================

def make_batch_hash(batch):
    """
    chunk_id와 text가 이전 실행과 같은지 확인하기 위한 해시.
    원본 텍스트가 변경되면 기존 임베딩을 다시 생성함.
    """
    h = hashlib.sha256()

    for chunk_id, text in zip(
        batch["chunk_id"],
        batch["text"]
    ):
        for value in (str(chunk_id), str(text)):
            raw = value.encode("utf-8")
            h.update(len(raw).to_bytes(8, "little"))
            h.update(raw)

    return h.hexdigest()


# ============================================================
# 6. 기존 배치 완료 여부 확인
# ============================================================

def is_batch_complete(
    vector_path,
    metadata_path,
    info_path,
    batch,
    batch_hash
):

    if not all([
        vector_path.exists(),
        metadata_path.exists(),
        info_path.exists()
    ]):
        return False

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        if info.get("model") != MODEL_NAME:
            return False

        if info.get("max_seq_length") != MAX_SEQ_LENGTH:
            return False

        if info.get("batch_hash") != batch_hash:
            return False

        if info.get("rows") != len(batch):
            return False

        # NPY의 전체 데이터를 RAM에 로딩하지 않음
        vectors = np.load(
            vector_path,
            mmap_mode="r",
            allow_pickle=False
        )

        if vectors.shape != (len(batch), EXPECTED_DIM):
            return False

        if vectors.dtype != np.float32:
            return False

        # 저장된 메타데이터의 ID 순서 검사
        saved_ids = pd.read_csv(
            metadata_path,
            usecols=["chunk_id"],
            dtype={"chunk_id": "string"},
            encoding="utf-8-sig"
        )["chunk_id"].tolist()

        if saved_ids != batch["chunk_id"].tolist():
            return False

        return True

    except Exception:
        return False


# ============================================================
# 7. 안전한 배치 저장
# ============================================================

def save_batch(
    vectors,
    batch,
    vector_path,
    metadata_path,
    info_path,
    batch_hash
):

    vector_tmp = Path(str(vector_path) + ".tmp")
    metadata_tmp = Path(str(metadata_path) + ".tmp")
    info_tmp = Path(str(info_path) + ".tmp")

    # ① 벡터 임시 저장
    with open(vector_tmp, "wb") as f:
        np.save(f, vectors)

    # ② 메타데이터 임시 저장
    batch.to_csv(
        metadata_tmp,
        index=False,
        encoding="utf-8-sig"
    )

    # ③ 저장 파일 확정
    os.replace(vector_tmp, vector_path)
    os.replace(metadata_tmp, metadata_path)

    # ④ 완료 정보 저장
    info = {
        "model": MODEL_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "rows": len(batch),
        "dimension": EXPECTED_DIM,
        "dtype": "float32",
        "batch_hash": batch_hash
    }

    with open(info_tmp, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    # 완료 표시는 가장 마지막에 생성
    os.replace(info_tmp, info_path)


# ============================================================
# 8. 지역별 임베딩
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.time()

total_completed = 0
total_new = 0
total_skipped = 0

year_counts = {2024: 0, 2025: 0}

for file_no, (year, file_path) in enumerate(input_files, start=1):

    region_name = file_path.stem.removeprefix(
        f"{year}_"
    ).removesuffix("_chunked")

    region_output = OUTPUT_DIR / str(year) / region_name
    region_output.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"[{file_no}/{len(input_files)}] {year}년 {region_name}")
    print(f"{'=' * 60}")

    region_count = 0

    reader = pd.read_csv(
        file_path,
        chunksize=SAVE_SIZE,
        encoding="utf-8-sig",
        dtype={
            "date": "string",
            "assembly": "string",
            "source_id": "string",
            "chunk_id": "string",
            "text": "string"
        }
    )

    for batch_no, batch in enumerate(reader, start=1):

        batch_start = time.time()

        if not set(REQUIRED_COLS).issubset(batch.columns):
            raise ValueError(f"필수 컬럼 누락: {file_path.name}")

        if batch[REQUIRED_COLS].isna().any().any():
            raise ValueError(f"필수값 결측: {file_path.name}")

        texts = batch["text"].astype(str).tolist()

        batch_hash = make_batch_hash(batch)

        prefix = f"batch_{batch_no:04d}"

        vector_path = region_output / f"{prefix}.npy"
        metadata_path = region_output / f"{prefix}_metadata.csv"
        info_path = region_output / f"{prefix}_info.json"

        # ----------------------------------------------------
        # 이미 완료된 배치는 건너뛰기
        # ----------------------------------------------------
        if is_batch_complete(
            vector_path,
            metadata_path,
            info_path,
            batch,
            batch_hash
        ):

            total_skipped += len(batch)

            print(
                f"  배치 {batch_no:04d}: "
                f"{len(batch):,}개 기존 완료 → 건너뜀"
            )

        else:

            # ------------------------------------------------
            # GPU 임베딩
            # ------------------------------------------------
            vectors = model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            vectors = np.asarray(
                vectors,
                dtype=np.float32
            )

            if vectors.shape != (len(batch), EXPECTED_DIM):
                raise ValueError(
                    f"벡터 크기 오류: {vectors.shape}"
                )

            if not np.isfinite(vectors).all():
                raise ValueError("NaN 또는 무한대 벡터 발생")

            # ------------------------------------------------
            # 결과 저장
            # ------------------------------------------------
            save_batch(
                vectors,
                batch,
                vector_path,
                metadata_path,
                info_path,
                batch_hash
            )

            total_new += len(batch)

            elapsed = time.time() - batch_start
            speed = len(batch) / elapsed

            print(
                f"  배치 {batch_no:04d}: "
                f"{len(batch):,}개 저장 완료 / "
                f"{elapsed:.1f}초 / "
                f"{speed:.2f}개/초"
            )

            del vectors

        region_count += len(batch)
        year_counts[year] += len(batch)
        total_completed += len(batch)

        print(
            f"  전체 진행: "
            f"{total_completed:,} / 684,311 "
            f"({total_completed / 684311 * 100:.2f}%)"
        )

    print(
        f"\n[{region_name}] "
        f"처리 및 검증 완료: {region_count:,}개"
    )

    # GPU 캐시 정리
    torch.cuda.empty_cache()


# ============================================================
# 9. 최종 결과
# ============================================================

elapsed_total = time.time() - start_time

print("\n" + "=" * 60)
print("===== 전체 임베딩 완료 =====")
print("=" * 60)

print("2024년:", f"{year_counts[2024]:,}")
print("2025년:", f"{year_counts[2025]:,}")
print("전체 처리:", f"{total_completed:,}")
print("이번 실행 새로 임베딩:", f"{total_new:,}")
print("기존 완료 배치:", f"{total_skipped:,}")

print(
    "총 실행시간:",
    round(elapsed_total / 3600, 2),
    "시간"
)

print("저장 경로:", OUTPUT_DIR)

if year_counts == EXPECTED_COUNTS:
    print("\n최종 판단: 전체 임베딩 완료")
else:
    print("\n최종 판단: 연도별 처리 건수 확인 필요")



# 파일	역할	이번 검증
# JSON	원본 텍스트 및 메타데이터 저장	필요시 원문 비교
# CSV	청크 텍스트와 메타데이터 확인	랜덤 표본 추출 대상
# NPY	임베딩 벡터 저장	이번 단계에서는 사용 안 함
