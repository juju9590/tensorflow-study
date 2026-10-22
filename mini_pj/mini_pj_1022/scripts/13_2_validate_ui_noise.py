# 검증 항목	              검증 방법	                            처리
# UI 문구	              메뉴, 로그인, 페이지 이동 등	          의심 항목 표시
# HTML 잔여물	           HTML 태그, &nbsp; 등	                의심 항목 표시
# 발언자 표시	            ○, 발언자 이름·직책 패턴	           현황 집계
# 메타데이터 결측	        genda_title, speaker_name 등	     결측 건수 집계
# 짧은 청크	텍스트          50자 미만	                          의심 항목 표시
# 중복 청크	동일한          text 반복	                          현황 집계

# UI와 HTML도 자동 삭제하지 않고 정규표현식으로 탐지된 문구가 실제 제거 대상인지 먼저 확인

from pathlib import Path
import pandas as pd
import re

# =====================================================
# 1. 경로 설정
# =====================================================

input_dir = Path(
    r"C:\mini_pj\data\processed\validation_samples"
)

output_dir = input_dir / "validation_results"
output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(input_dir.glob("sample_*.csv"))

print(f"검증 대상 CSV 파일: {len(files)}개")

# =====================================================
# 2. 검사 패턴 설정
# =====================================================

# 웹페이지 UI 문구 (단독 문구 중심)
ui_pattern = re.compile(
    r"(?m)^\s*("
    r"로그인|회원가입|아이디 찾기|비밀번호 찾기|"
    r"이전 페이지|다음 페이지|"
    r"목록으로|페이지 이동|글자 크기 조절|"
    r"전체메뉴|통합검색"
    r")\s*$"
)

# HTML 태그
html_pattern = re.compile(
    r"</?[a-zA-Z][^>]*>"
)

# HTML 엔티티
entity_pattern = re.compile(
    r"&(?:nbsp|amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);"
)

# 발언자 구분 기호
speaker_pattern = re.compile(r"○")

# =====================================================
# 3. 전체 표본 검증
# =====================================================

results = []

for file in files:

    df = pd.read_csv(file, dtype=str, keep_default_na=False)

    if "text" not in df.columns:
        print(f"[제외] text 컬럼 없음: {file.name}")
        continue

    for _, row in df.iterrows():

        text = row["text"]

        ui_matches = ui_pattern.findall(text)
        html_matches = html_pattern.findall(text)
        entity_matches = entity_pattern.findall(text)

        result = {
            "sample_file": file.name,
            "date": row.get("date", ""),
            "assembly": row.get("assembly", ""),
            "source_id": row.get("source_id", ""),
            "chunk_id": row.get("chunk_id", ""),
            "text": text,
            "text_length": len(text),
            "ui_detected": bool(ui_matches),
            "ui_matches": str(ui_matches),
            "html_detected": bool(html_matches),
            "html_matches": str(html_matches[:5]),
            "entity_detected": bool(entity_matches),
            "entity_matches": str(entity_matches[:5]),
            "speaker_marker_count": len(
                speaker_pattern.findall(text)
            ),
            "agenda_missing": not row.get(
                "agenda_title", ""
            ).strip(),
            "speaker_name_missing": not row.get(
                "speaker_name", ""
            ).strip(),
            "speaker_role_missing": not row.get(
                "speaker_role", ""
            ).strip(),
            "short_chunk": len(text.strip()) < 50,
        }

        results.append(result)

# =====================================================
# 4. 결과 데이터프레임
# =====================================================

result_df = pd.DataFrame(results)

if result_df.empty:
    print("검증할 데이터가 없습니다.")
    raise SystemExit

# 동일한 텍스트 기준 중복 검사
result_df["duplicate_text"] = result_df.duplicated(
    subset=["text"],
    keep=False
)

# 하나라도 정제 의심 항목이 있는 경우
result_df["noise_flag"] = (
    result_df["ui_detected"]
    | result_df["html_detected"]
    | result_df["entity_detected"]
    | result_df["short_chunk"]
)

# =====================================================
# 5. 결과 저장
# =====================================================

# 전체 검증 결과
result_df.to_csv(
    output_dir / "validation_all.csv",
    index=False,
    encoding="utf-8-sig"
)

# 정제 의심 항목만 저장
result_df[result_df["noise_flag"]].to_csv(
    output_dir / "validation_suspected.csv",
    index=False,
    encoding="utf-8-sig"
)

# =====================================================
# 6. 검증 요약
# =====================================================

summary = {
    "전체 청크 수": len(result_df),
    "UI 문구 탐지": int(result_df["ui_detected"].sum()),
    "HTML 태그 탐지": int(result_df["html_detected"].sum()),
    "HTML 엔티티 탐지": int(result_df["entity_detected"].sum()),
    "50자 미만 청크": int(result_df["short_chunk"].sum()),
    "중복 텍스트 청크": int(result_df["duplicate_text"].sum()),
    "안건명 결측": int(result_df["agenda_missing"].sum()),
    "발언자명 결측": int(
        result_df["speaker_name_missing"].sum()
    ),
    "발언자 직책 결측": int(
        result_df["speaker_role_missing"].sum()
    ),
    "정제 의심 청크": int(result_df["noise_flag"].sum())
}

summary_df = pd.DataFrame(
    summary.items(),
    columns=["검증 항목", "건수"]
)

summary_df.to_csv(
    output_dir / "validation_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== 1차 검증 결과 =====")
print(summary_df.to_string(index=False))

print("\n검증 완료!")
print("결과 저장 위치:", output_dir)


# 메타데이터 결측 확인
cols = [
    "meeting_title",
    "agenda_title",
    "speaker_name",
    "speaker_role",
    "source_id",
    "chunk_id"
]

for col in cols:
    missing = (
        result_df[col].isna()
        | result_df[col].astype(str).str.strip().eq("")
    ) if col in result_df.columns else None

    if missing is None:
        print(f"{col}: 검증 결과에 컬럼 없음")
    else:
        print(f"{col}: 결측 {missing.sum()}건")


# =====================================================
# 7. 원본 표본 CSV의 메타데이터 확인
# =====================================================

sample_file = files[0]

df_sample = pd.read_csv(sample_file)

print("\n검사 파일:", sample_file.name)

cols = [
    "meeting_title",
    "agenda_title",
    "speaker_name",
    "speaker_role",
    "speaker_type"
]

print("\n===== 메타데이터 결측 현황 =====")

for col in cols:
    if col in df_sample.columns:
        missing = (
            df_sample[col].isna()
            | df_sample[col].astype(str).str.strip().eq("")
        )

        print(
            f"{col}: "
            f"정상 {len(df_sample) - missing.sum()}건 / "
            f"결측 {missing.sum()}건"
        )

print("\n===== 실제 데이터 5행 =====")
print(df_sample[cols].head(5).to_string(index=False))