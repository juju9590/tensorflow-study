import pandas as pd
import requests
import json
import re
import os
from tqdm import tqdm

# --- 설정 (로컬 환경에 맞게 수정) ---
OLLAMA_URL = "http://localhost:11434/v1"
INPUT_FILE = "./data/to_text2/cleaned_2024_광역.txt" # 이전 단계에서 병합된 txt 파일
OUTPUT_FILE = "./data/master_council_data.csv"
# MODEL_NAME = "qwen2.5-coder:14b" 
MODEL_NAME = "huihui_ai/exaone3.5-abliterated" # LG에서 개발한 모델을 기반

def get_metadata_from_llm(text: str) -> dict:
    """Ollama를 통해 발언 내용에서 메타데이터를 JSON 형태로 추출합니다."""
    
    prompt = f"""당신은 지방의회 정책 분석 전문가입니다.
아래 주어진 [발언 텍스트]를 읽고 정책 분야 분류, 핵심 이슈, 키워드, 요약을 추출하여 반드시 아래 제시된 JSON 포맷으로만 응답하세요. 설명이나 마크다운을 추가하지 마세요.

[JSON 출력 포맷]
{{
    "category": "보건/복지, 도시계획/건설, 경제/산업, 행정, 교육, 기타 중 택 1",
    "issue": "해당 발언의 가장 핵심적인 쟁점이나 이슈 (한 문장)",
    "keywords": "핵심 키워드 3~5개 (콤마로 구분된 문자열)",
    "summary": "발언의 요지와 맥락을 1~2줄로 명확하게 요약"
}}

[발언 텍스트]
{text}
"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Ollama에서 순수 JSON만 강제 출력하도록 설정
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result_text = response.json().get("response", "").strip()
        
        # 간혹 LLM이 JSON 앞뒤에 쓰레기값을 넣는 경우를 대비한 정규식 추출
        match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            return json.loads(result_text)
            
    except Exception as e:
        # LLM 처리 실패 또는 JSON 파싱 오류 시 기본값 반환
        return {
            "category": "분석실패",
            "issue": "추출 실패",
            "keywords": "",
            "summary": "텍스트 길이 초과 또는 처리 오류"
        }

if __name__ == "__main__":
    print(f"데이터 로드 중: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    
    # 이어하기(Checkpoint) 로직: 중간에 멈춰도 기존 작업분 이후부터 다시 시작
    start_idx = 0
    results = []
    
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        start_idx = len(existing_df)
        results = existing_df.to_dict('records')
        print(f"기존 저장된 데이터 {start_idx}건을 발견했습니다. 이어서 작업을 시작합니다.")
    else:
        print("새로운 Master CSV 생성을 시작합니다.")

    print(f"LLM({MODEL_NAME}) 분석 시작...")
    
    # tqdm을 사용해 진행률 표시
    for idx, row in tqdm(df.iloc[start_idx:].iterrows(), total=len(df) - start_idx):
        utterance = str(row.get('utterance', ''))
        
        # LLM 메타데이터 추출
        llm_result = get_metadata_from_llm(utterance)
        
        # 원본 데이터에 LLM 결과 병합
        row_data = row.to_dict()
        row_data.update(llm_result)
        results.append(row_data)
    
        # 20건 처리할 때마다 안전하게 중간 저장 (메모리 부족 방지 및 정전 대비)
        if (idx + 1) % 20 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # 전체 완료 후 최종 저장
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n모든 작업 완료! 최종 결과가 {OUTPUT_FILE}에 저장되었습니다.")