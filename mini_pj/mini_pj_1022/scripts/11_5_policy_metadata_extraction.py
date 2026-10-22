# 항목	설정
# 파일명	11_5_policy_metadata_extraction.py
# 입력	embedding_region_51.csv
# 테스트 수량	우선 10개 청크
# 추출 모델	Qwen3 8B, Qwen2.5 7B, Gemma3 4B, Gemma4 E2B
# 실행 방식	Ollama 로컬 API
# GPU	RTX 4060 Laptop GPU 8GB
# 출력	정책 메타데이터 JSON 및 CSV
# 원문 연결	source_id, chunk_id 유지
# ㄴ 이번에는 임베딩하지 않아. 먼저 각 모델의 정책 정보 추출 능력을 비교하고, 그다음 가장 적합한 모델로 임베딩용 텍스트를 생성할 거야.
# ㄴ 그리고 하나의 청크에 정책 이슈가 여러 개라면 여러 개의 결과를 추출하도록 만들자.

# python -m pip install ollama pandas


# ============================================================
# 1. 라이브러리
# ============================================================
from pathlib import Path
import pandas as pd
import json
import time
import subprocess
from ollama import chat


# ============================================================
# 2. 파일 경로 및 모델 설정
# ============================================================
base_dir = Path(
    r"C:\mini_pj\data\processed\embedding_pilot_1000"
)

csv_path = base_dir / "embedding_region_51.csv"

models = [
    "qwen3:8b",
    "qwen2.5:7b",
    "gemma3:4b",
    "gemma4:e2b"
]


# ============================================================
# 3. CSV 불러오기 및 10개 추출
# ============================================================
df = pd.read_csv(csv_path, encoding="utf-8-sig")

required_cols = [
    "date", "assembly", "source_id",
    "chunk_id", "chunk_index", "text"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"필수 컬럼 누락: {col}")

# 지역별 첫 번째 청크를 후보로 선택
candidates = (
    df.dropna(subset=["text"])
    .groupby("assembly", group_keys=False)
    .head(1)
)

# 서로 다른 10개 지역에서 추출
df_test = candidates.sample(
    n=10,
    random_state=42
).reset_index(drop=True)

print("테스트 청크:", len(df_test))
print("지역 수:", df_test["assembly"].nunique())


# ============================================================
# 4. JSON 출력 구조 정의
# ============================================================
issue_schema = {
    "type": "object",
    "properties": {
        "policy_category": {"type": "string"},
        "core_issue": {"type": "string"},
        "policy_stance": {"type": "string"},
        "policy_action": {"type": "string"},
        "keywords": {
            "type": "array",
            "items": {"type": "string"}
        },
        "summary": {"type": "string"},
        "evidence_text": {"type": "string"}
    },
    "required": [
        "policy_category", "core_issue",
        "policy_stance", "policy_action",
        "keywords", "summary", "evidence_text"
    ]
}

output_schema = {
    "type": "object",
    "properties": {
        "is_substantive": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": issue_schema
        }
    },
    "required": ["is_substantive", "issues"]
}


# ============================================================
# 5. 정책 메타데이터 추출 프롬프트
# ============================================================
system_prompt = """
당신은 한국 지방의회 회의록의 정책 분석 전문가입니다.

목표:
회의록에서 실질적인 정책 이슈를 추출합니다.

규칙:
1. 원문에 명확히 나타난 정보만 사용합니다.
2. 원문에 없는 정책이나 사실을 생성하지 않습니다.
3. 회의 진행, 단순 인사, 의원 명단은 정책 이슈로 분류하지 않습니다.
4. 다만 조례 가결·부결 등 실제 정책 처리 결과는 보존합니다.
5. 정책 이슈가 없으면 is_substantive=false, issues=[]로 출력합니다.
6. 하나의 청크에 여러 정책 이슈가 있으면 각각 추출합니다.
7. core_issue는 구체적인 정책 문제 또는 사업명으로 작성합니다.
8. policy_stance는 찬성, 반대, 우려, 조건부,
   중립 등 원문에서 확인되는 입장으로 작성합니다.
9. policy_action은 제안, 검토, 심사, 의결, 시행 등
   원문에서 확인되는 행위로 작성합니다.
10. 판단이 불가능한 항목은 빈 문자열로 작성합니다.
11. evidence_text는 원문에 실제 존재하는
    연속된 문장 또는 구절을 그대로 복사합니다.
12. 날짜, 지역, 발언자, source_id를 새로 생성하지 않습니다.
13. 반드시 지정된 JSON 구조로 응답합니다.
"""


# ============================================================
# 6. 모델별 정책 추출
# ============================================================
results = []

for model_name in models:

    print(f"\n========== {model_name} 시작 ==========")

    model_start = time.time()

    for i, row in df_test.iterrows():

        text = str(row["text"])

        start = time.time()

        try:
            response = chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"회의록 원문:\n{text}"
                    }
                ],
                format=output_schema,
                options={
                    "temperature": 0,
                    "num_ctx": 4096,
                    "num_predict": 1200
                },
                think=False if model_name == "qwen3:8b" else None,
                keep_alive="5m"
            )

            data = json.loads(response.message.content)

            is_substantive = data["is_substantive"]
            issues = data["issues"]

            elapsed = round(time.time() - start, 2)

            # 정책 이슈가 없는 경우도 기록
            if not issues:
                issues = [{}]

            for issue_idx, issue in enumerate(issues, start=1):

                evidence = issue.get("evidence_text", "")

                evidence_valid = (
                    bool(evidence)
                    and evidence in text
                )

                results.append({
                    "model": model_name,
                    "date": row["date"],
                    "assembly": row["assembly"],
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    "chunk_index": row["chunk_index"],
                    "issue_index": issue_idx,
                    "is_substantive": is_substantive,
                    "policy_category": issue.get(
                        "policy_category", ""
                    ),
                    "core_issue": issue.get("core_issue", ""),
                    "policy_stance": issue.get(
                        "policy_stance", ""
                    ),
                    "policy_action": issue.get(
                        "policy_action", ""
                    ),
                    "keywords": json.dumps(
                        issue.get("keywords", []),
                        ensure_ascii=False
                    ),
                    "summary": issue.get("summary", ""),
                    "evidence_text": evidence,
                    "evidence_valid": evidence_valid,
                    "elapsed_sec": elapsed,
                    "error": ""
                })

            print(
                f"[{i+1}/10] {row['assembly']} | "
                f"이슈 {len(data['issues'])}개 | "
                f"{elapsed}초"
            )

        except Exception as e:

            print(f"[{i+1}/10] 오류: {e}")

            results.append({
                "model": model_name,
                "date": row["date"],
                "assembly": row["assembly"],
                "source_id": row["source_id"],
                "chunk_id": row["chunk_id"],
                "chunk_index": row["chunk_index"],
                "issue_index": 0,
                "is_substantive": None,
                "policy_category": "",
                "core_issue": "",
                "policy_stance": "",
                "policy_action": "",
                "keywords": "[]",
                "summary": "",
                "evidence_text": "",
                "evidence_valid": False,
                "elapsed_sec": round(time.time() - start, 2),
                "error": str(e)
            })

    print(
        f"{model_name} 총 처리 시간:",
        round(time.time() - model_start, 2),
        "초"
    )

    # 모델 메모리 해제
    subprocess.run(
        ["ollama", "stop", model_name],
        capture_output=True,
        text=True
    )

    # 모델별 완료 시 중간 저장
    pd.DataFrame(results).to_csv(
        base_dir / "metadata_model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 7. 최종 결과 저장
# ============================================================
df_result = pd.DataFrame(results)

output_path = base_dir / "metadata_model_comparison.csv"

df_result.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 테스트 완료 ==========")
print("전체 결과 행:", len(df_result))
print("저장 경로:", output_path)


# ============================================================
# 8. 모델별 결과 요약
# ============================================================
print("\n========== 모델별 요약 ==========")

for model_name in models:

    sub = df_result[df_result["model"] == model_name]

    success = sub[sub["error"] == ""]

    valid = success["evidence_valid"].sum()

    print(f"\n모델: {model_name}")
    print("결과 행:", len(sub))
    print("근거 일치:", int(valid))
    print("오류 행:", (sub["error"] != "").sum())
    print("추출 시간 합계:", round(
        sub.groupby("chunk_id")["elapsed_sec"].first().sum(), 2
    ), "초")

# ========== 모델별 요약 ==========

# 모델: qwen3:8b
# 결과 행: 20
# 근거 일치: 9
# 오류 행: 1
# 추출 시간 합계: 501.29 초

# 모델: qwen2.5:7b
# 결과 행: 12
# 근거 일치: 3
# 오류 행: 0
# 추출 시간 합계: 79.8 초

# 모델: gemma3:4b
# 결과 행: 21
# 근거 일치: 12
# 오류 행: 1
# 추출 시간 합계: 100.38 초

# 모델: gemma4:e2b
# 결과 행: 11
# 근거 일치: 1
# 오류 행: 9
# 추출 시간 합계: 77.38 초
