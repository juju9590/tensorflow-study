"""17개 광역의회 청크 중 1,000개를 균형 표본 추출해 GPU 임베딩 실험.

실행: conda activate mini && python scripts/11_embedding_pilot_1000.py
기본 경로: 이 파일을 mini_pj/scripts/ 안에 둔 경우를 기준으로 함.
"""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# 1. 설정: 경로만 실제 파일 위치에 맞게 수정
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "data" / "processed" / "chunked_by_region"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "embedding_pilot_1000"

MODEL_NAME = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
SAMPLE_SIZE = 1000
MIN_TEXT_LENGTH = 100  # 이전 검증에서 100자 미만의 짧은 잡음 청크 19건 확인
SEED = 42
BATCH_SIZE = 8  # 4060 Laptop 8GB 등에서는 4로 낮춰 테스트
CLUSTER_COUNT = 8  # 탐색용 임시 군집 수, 최적 주제 수가 아님

QUERIES = [
    "지역 소멸과 청년 인구 유출을 막기 위한 지원 정책",
    "지역 내 대중교통 개선 및 교통약자 이동 지원 정책",
    "소상공인과 지역 중소기업 지원 예산",
    "기후변화와 재난 예방을 위한 환경 안전 정책",
    "저출생 대응 및 돌봄과 복지 정책",
]


def sample_region_csv(path: Path, quota: int, rng: random.Random):
    """큰 지역 CSV도 전체를 RAM에 올리지 않고 균일 무작위 표본 추출."""
    reservoir = []
    eligible = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"chunk_id", "source_id", "date", "assembly", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: 누락된 컬럼 {sorted(missing)}")

        for row in reader:
            content = row["text"] or ""
            if len(content.strip()) < MIN_TEXT_LENGTH:
                continue
            eligible += 1
            if len(reservoir) < quota:
                reservoir.append(row)
            else:
                pos = rng.randrange(eligible)
                if pos < quota:
                    reservoir[pos] = row

    if eligible < quota:
        raise ValueError(f"{path.name}: 표본 후보 {eligible}개, 요청 {quota}개")
    return reservoir, eligible


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU가 감지되지 않았습니다. torch CUDA 설치와 VS Code 인터프리터를 확인하세요.")

    files = sorted(INPUT_DIR.glob("*_chunked.csv"))
    if not files:
        raise FileNotFoundError(f"지역별 청킹 CSV를 찾을 수 없습니다: {INPUT_DIR}")
    if len(files) > SAMPLE_SIZE:
        raise ValueError("지역 파일 수가 전체 표본 수보다 많습니다.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    base, remainder = divmod(SAMPLE_SIZE, len(files))
    samples = []
    region_report = []

    print(f"지역 파일: {len(files)}개 / 표본 목표: {SAMPLE_SIZE:,}개", flush=True)
    for idx, path in enumerate(files):
        quota = base + int(idx < remainder)
        selected, eligible = sample_region_csv(path, quota, rng)
        samples.extend(selected)
        region_report.append({"file": path.name, "eligible": eligible, "sampled": len(selected)})
        print(f"[{idx+1}/{len(files)}] {path.name}: {eligible:,}개 중 {len(selected)}개 선택", flush=True)

    if len(samples) != SAMPLE_SIZE:
        raise AssertionError(f"표본 수 불일치: {len(samples)}")
    ids = [row["chunk_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("선정 표본에 중복 chunk_id가 있습니다.")

    metadata = pd.DataFrame(samples)
    metadata.insert(0, "sample_index", range(len(metadata)))
    metadata.to_csv(OUTPUT_DIR / "embedding_sample_1000.csv", index=False, encoding="utf-8-sig")

    device = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device}", flush=True)
    print("모델 로드 시작 (첫 실행이면 다운로드 시간 포함)", flush=True)
    start_load = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start_load

    texts = metadata["text"].tolist()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_embed = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    torch.cuda.synchronize()
    embed_seconds = time.perf_counter() - start_embed
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    reserved_memory_mb = torch.cuda.max_memory_reserved() / (1024**2)

    if embeddings.shape[0] != SAMPLE_SIZE:
        raise ValueError(f"임베딩 행 수 오류: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("NaN 또는 무한대가 임베딩에 포함됐습니다.")
    np.save(OUTPUT_DIR / "embedding_vectors_1000.npy", embeddings)

    print("\n유사 정책 문서 검색 테스트", flush=True)
    retrieval_rows = []
    for query in QUERIES:
        # 모델 카드 권장: 질의문에는 query 프롬프트, 문서에는 프롬프트 없음.
        query_vec = model.encode(
            query, prompt_name="query", normalize_embeddings=True, convert_to_numpy=True
        )
        similarities = embeddings @ query_vec
        top_idx = np.argsort(-similarities)[:5]
        print(f"\n질문: {query}", flush=True)
        for rank, i in enumerate(top_idx, 1):
            row = metadata.iloc[int(i)]
            score = float(similarities[i])
            print(f" {rank}. {row['assembly']} | 유사도 {score:.4f} | {row['text'][:90]!r}", flush=True)
            retrieval_rows.append({
                "query": query,
                "rank": rank,
                "cosine_similarity": score,
                "chunk_id": row["chunk_id"],
                "source_id": row["source_id"],
                "date": row["date"],
                "assembly": row["assembly"],
                "text": row["text"],
            })
    pd.DataFrame(retrieval_rows).to_csv(
        OUTPUT_DIR / "retrieval_top5.csv", index=False, encoding="utf-8-sig"
    )

    # 탐색적인 군집화. 실무 정책 이슈 군집의 품질 판단은 별도 수작업 검토 필요.
    print("\n탐색용 KMeans 군집화 시작", flush=True)
    km = KMeans(n_clusters=CLUSTER_COUNT, random_state=SEED, n_init=10)
    labels = km.fit_predict(embeddings)
    metadata["cluster"] = labels
    metadata.to_csv(OUTPUT_DIR / "cluster_samples_1000.csv", index=False, encoding="utf-8-sig")
    silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))
    counts = metadata["cluster"].value_counts().sort_index().to_dict()
    for k, count in counts.items():
        print(f"군집 {k}: {count}개", flush=True)

    report = {
        "model": MODEL_NAME,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "sample_size": len(samples),
        "region_files": len(files),
        "random_seed": SEED,
        "min_text_length": MIN_TEXT_LENGTH,
        "batch_size": BATCH_SIZE,
        "model_load_seconds": round(load_seconds, 3),
        "embedding_seconds": round(embed_seconds, 3),
        "embeddings_per_second": round(len(samples) / embed_seconds, 2),
        "embedding_dimension": int(embeddings.shape[1]),
        "gpu_peak_allocated_mb": round(peak_memory_mb, 1),
        "gpu_peak_reserved_mb": round(reserved_memory_mb, 1),
        "vector_file_mb": round(embeddings.nbytes / (1024**2), 2),
        "cluster_count": CLUSTER_COUNT,
        "cluster_silhouette_cosine": round(silhouette, 4),
        "region_sampling": region_report,
    }
    with (OUTPUT_DIR / "embedding_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n===== 임베딩 실험 완료 =====", flush=True)
    print(f"모델 로딩: {load_seconds:.1f}초", flush=True)
    print(f"1,000개 임베딩: {embed_seconds:.1f}초", flush=True)
    print(f"임베딩 차원: {embeddings.shape[1]}", flush=True)
    print(f"GPU 피크 할당 메모리: {peak_memory_mb:.1f}MB", flush=True)
    print(f"탐색 군집 실루엣 점수(코사인): {silhouette:.4f}", flush=True)
    print(f"저장 경로: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
