import os
import glob
import json
import re
from openai import OpenAI
from bs4 import BeautifulSoup
import _constant as constant

def analyze_samples_with_local_llm(
    input_dir=constant.sample_dir,
    output_dir=constant.rules_output_dir,
    model_name=constant.make_model_name,
):
    # 로컬 LLM API 서버 주소 설정
    client = constant.client

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 하위 폴더의 모든 txt 파일을 읽어오도록 수정 (** 및 recursive=True)
    sample_files = glob.glob(os.path.join(input_dir, "**", "*.txt"), recursive=True)
    
    
    
    if not sample_files:
        print(f"오류: '{input_dir}' 폴더(및 하위 폴더)에 분석할 샘플 파일이 없습니다.")
        return

    print(f"총 {len(sample_files)}개 파일에 대해 로컬 LLM 자동 분석을 시작합니다...\n")

    # 시스템 프롬프트
    system_prompt = (
        "You are an expert HTML/CSS parser. Your ONLY job is to extract CSS selectors for agenda, speaker, and utterance. "
        "CRITICAL RULE: Return ONLY a valid JSON object. Do not explain."
    )

    for sample_file in sample_files:
        base_name = os.path.basename(sample_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        # 확장자를 .json으로 변경
        output_rule_file = os.path.join(output_dir, f"rule_{file_name_without_ext}.json")

        print(f"[{base_name}] LLM 분석 요청 중...")

        try:
            # 샘플 파일 읽기
            with open(sample_file, 'r', encoding='utf-8') as f:
                sample_html = f.read()

            # [수정 1] BeautifulSoup을 이용해 HTML을 구조적으로 파싱
            soup = BeautifulSoup(sample_html, 'html.parser')

            # [수정 2] 방해되는 시맨틱 태그를 구조적으로 완전 삭제 (헤더, 사이드바, 네비게이션 등)
            for tag in soup(['header', 'nav', 'aside', 'footer', 'script', 'style', 'form', 'noscript']):
                tag.decompose()

            # [수정 3] 사이드바, 팝업, 메뉴 관련 클래스가 포함된 태그 덩어리 삭제
            # 정규식을 써서 해당 단어가 포함된 class를 통째로 날립니다.
            for tag in soup.find_all(class_=re.compile(r'sidebar|menu|dropdown|nav|logo|popup', re.IGNORECASE)):
                tag.decompose()

            # [수정 4] 남아있는 깔끔한 본문을 문자열로 변환 후 마스킹 처리
            clean_html = str(soup)
            masked_html = re.sub(r'>[^<]+<', '>[TEXT]<', clean_html)
            
            # [수정 5] 앞부분의 껍데기가 완벽히 날아갔으므로, 실제 본문이 포함되도록 넉넉하게 자름
            masked_html = masked_html[:20000]

            # 마스킹된 HTML에 맞춘 프롬프트
            user_prompt = f"""
            Analyze the provided [MASKED HTML] and find the best CSS selectors for the agenda title, speaker name, and utterance content.

            [RULES]
            1. ONLY output standard CSS selectors.
            2. NEVER output the string "[TEXT]".
            3. If class names contain random IDs (like "sMan123", "an0"), use wildcard attribute selectors like [class^="sMan"] or [id^="an"].

            [EXAMPLE OF KOREAN COUNCIL MINUTES]
            [MASKED HTML]
            <div id="an0"><p class="em1"><span class="bold">[TEXT]</span></p></div>
            <div class="sManPV11215"><p><span class="bold">[TEXT]</span>[TEXT]</p><p class="em1">[TEXT]</p></div>

            [OUTPUT]
            {{
                "agenda_selector": "div[id^='an'] p.em1 span.bold",
                "speaker_selector": "div[class^='sMan'] span.bold",
                "utterance_selector": "div[class^='sMan']"
            }}

            --- ACTUAL TASK ---
            [MASKED HTML]
            {masked_html}
            """

            # 로컬 LLM에 API 요청
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"} # JSON 형식 강제
            )

            # 결과 처리 및 마크다운 찌꺼기 제거
            llm_result = response.choices[0].message.content.strip()
            llm_result = re.sub(r'^```json\s*|\s*```$', '', llm_result, flags=re.MULTILINE).strip()
            
            rule_json = json.loads(llm_result)
            
            # 원하시는 최종 JSON 구조로 가공
            final_rule = {
                file_name_without_ext: {
                    "agenda": rule_json.get("agenda_selector", ""),
                    "speaker": rule_json.get("speaker_selector", ""),
                    "utterance": rule_json.get("utterance_selector", "")
                }
            }

            # JSON 파일로 저장 (덮어쓰기 버그 수정됨)
            with open(output_rule_file, 'w', encoding='utf-8') as f:
                json.dump(final_rule, f, ensure_ascii=False, indent=4)
                
            print(f"  -> 분석 완료! '{os.path.basename(output_rule_file)}' 파일로 저장됨.\n")

        except Exception as e:
            print(f"  -> [오류] {base_name} 분석 중 문제 발생: {e}\n")

    print(f"모든 파일의 자동 분석이 완료되었습니다. 결과물은 '{output_dir}' 폴더를 확인하세요.")

if __name__ == "__main__":
    analyze_samples_with_local_llm()

# **가장 크게 달라진 점 요약:**
# 1. 모델이 무조건 `qwen2.5-coder:7b`로 실행되도록 설정했습니다. (Ollama에 해당 모델이 없다면 `ollama run qwen2.5-coder:7b` 명령어로 먼저 다운로드 해주세요.)
# 2. `re.sub`를 이용한 텍스트 마스킹 기법이 추가되었습니다.
# 3. 데이터가 `{"2024_강원특별자치도의회": {"agenda": "...", "speaker": "...", "utterance": "..."}}` 형태로 깔끔하게 `.json` 파일로 저장됩니다.

