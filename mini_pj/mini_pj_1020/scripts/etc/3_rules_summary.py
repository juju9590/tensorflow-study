import os
import glob
import json
import re
from openai import OpenAI
import _constant as constant

def generate_strict_parsing_rules(
    input_dir=constant.rules_output_dir,
    output_file=constant.all_rule_file,
    model_name=constant.make_model_name,
):
    client = constant.client

    # 기존 폴더에서 .json 파일만 읽어오도록 수정
    rule_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not rule_files:
        print(f"[{input_dir}] 폴더에서 읽을 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(rule_files)}개의 JSON 파일에서 엄격한 파싱 규칙 추출을 시작합니다...\n")

    korean_pattern = re.compile(r'[가-힣]')

    # JSON 데이터 분석에 맞게 시스템 프롬프트 수정
    system_prompt = (
        "You are a strict data parsing system. "
        "Analyze the provided JSON rule data and extract HTML CSS selectors (like 'div.class', 'span#id', 'p'). "
        "NEVER output Korean words, slashes(/), or explanations in the JSON values. "
        "If you cannot find the selector, output 'div' for agenda, 'span' for speaker, and 'p' for utterance."
    )

    consolidated_rules = {}

    for file_path in rule_files:
        filename = os.path.basename(file_path)
        council_name = os.path.splitext(filename)[0].replace("rule_sample_", "").replace("rule_", "")

        try:
            # JSON 파일로 읽어 문자열로 변환
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_json = json.load(f)
                content = json.dumps(raw_json, ensure_ascii=False)

            # 프롬프트에 JSON 데이터를 분석하라는 명시적 지시 추가
            user_prompt = f"""
다음 JSON 데이터를 분석하여 3가지 요소의 CSS 선택자를 JSON으로 추출해.
[제약사항] value 값에는 절대 한글이나 '/'가 들어가면 안 됨. 오직 영문과 CSS 기호(. # >)만 사용.

Good Example:
{{ "agenda": "div.agnd_0", "speaker": "span.spkrbld", "utterance": "p" }}

Bad Example (DO NOT DO THIS):
{{ "agenda": "안건 제목 태그", "speaker": "발언자/클래스", "utterance": "발언 내용" }}

[분석할 JSON 데이터]
{content[:2000]}
"""
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={ "type": "json_object" }
            )

            llm_result = response.choices[0].message.content.strip()
            
            if "```json" in llm_result:
                llm_result = llm_result.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_result:
                llm_result = llm_result.split("```")[1].strip()

            rule_json = json.loads(llm_result)
            
            is_valid = True
            for key in ["agenda", "speaker", "utterance"]:
                val = rule_json.get(key, "")
                if not val or korean_pattern.search(val) or '/' in val:
                    is_valid = False
                    break
            
            if is_valid:
                consolidated_rules[council_name] = rule_json
                print(f"[성공] '{council_name}' -> {rule_json}")
            else:
                print(f"[경고] '{council_name}' 추출 값에 한글/오류 포함됨. 기본값으로 덮어씁니다.")
                consolidated_rules[council_name] = {
                    "agenda": "div",  
                    "speaker": "span.spkrbld", 
                    "utterance": "p"
                }

        except json.JSONDecodeError:
            print(f"[실패] '{council_name}' - JSON 파싱 실패. 기본값 주입.")
            consolidated_rules[council_name] = {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"}
        except Exception as e:
            print(f"[오류] '{council_name}': {e}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_rules, f, ensure_ascii=False, indent=4)
        
    print(f"\n최종 규칙 정리가 완료되었습니다. '{output_file}'을 확인하세요.")

if __name__ == "__main__":
    generate_strict_parsing_rules()