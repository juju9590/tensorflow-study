import json
import os
import glob
import random
import re
from bs4 import BeautifulSoup
import _constant as constant

def extract_samples_from_folder(
        input_dir="./input_data", 
        output_dir="./sample_outputs", 
        num_records=2):
    """지정된 폴더의 모든 JSON 파일에서 LLM 분석용 HTML 샘플을 각각 랜덤 추출하여 저장"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    if not json_files:
        print(f"오류: '{input_dir}' 폴더에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 파일에서 중간 영역 샘플 랜덤 추출을 시작합니다...\n")

    for input_file in json_files:
        base_name = os.path.basename(input_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_sample_file = os.path.join(output_dir, f"sample_{file_name_without_ext}.txt")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            records = data if isinstance(data, list) else [data]
            
            # -----------------------------------------------------------
            # [변경점] 처음 5개와 마지막 5개를 제외하고 중간 영역에서 랜덤 추출
            # -----------------------------------------------------------
            if len(records) > 10:
                # 앞 5개, 뒤 5개를 자른 중간 레코드들
                middle_records = records[5:-5]
            else:
                # 만약 전체 레코드가 10개 이하라면 자르지 않고 전체를 대상으로 함
                middle_records = records
                
            # 지정된 수(num_records)만큼 겹치지 않게 랜덤 추출
            sample_size = min(num_records, len(middle_records))
            selected_records = random.sample(middle_records, sample_size)
            # -----------------------------------------------------------
            
            sample_texts = []
            for record in selected_records:
                html_content = record.get("MINTS_HTML", "")
                if html_content:
                    # [변경] LLM에게 의미 없는 윗부분 껍데기 태그를 먼저 제거
                    soup = BeautifulSoup(html_content, "html.parser")
                    for tag in soup(["script", "style", "head", "noscript", "form"]):
                        tag.decompose()
                    
                    # 정제된 HTML을 문자열로 변환 후 20000자까지 넉넉하게 자르기
                    clean_html = str(soup)
                    # 의미 없는 껍데기 태그 제거
                    soup = BeautifulSoup(html_content, "html.parser")
                    for tag in soup(["script", "style", "head", "noscript", "form"]):
                        tag.decompose()
                    
                    # [핵심] 긴 발언 내용을 모두 15글자로 잘라서 "읽을거리"를 없앰
                    for string_node in soup.find_all(string=True):
                        text = string_node.strip()
                        if len(text) > 15:
                            # 텍스트를 강제로 자르고 생략 표시
                            string_node.replace_with(text[:15] + "...[내용 생략됨]") #text[:15]
                    
                    clean_html = str(soup)
                    sample_texts.append(clean_html[:10000])
            
            if sample_texts:
                with open(output_sample_file, 'w', encoding='utf-8') as f:
                    for idx, text in enumerate(sample_texts):
                        f.write(f"--- [Random Sample Record {idx+1} from {base_name}] ---\n")
                        f.write(text + "\n\n")
                print(f"[성공] '{base_name}' -> '{os.path.basename(output_sample_file)}' 생성 완료")
            else:
                print(f"[알림] '{base_name}': 파싱할 HTML(MINTS_HTML) 데이터를 찾을 수 없음")
                
        except Exception as e:
            print(f"[오류] '{base_name}' 처리 중 문제 발생: {e}")

    print(f"\n모든 파일의 샘플 추출이 완료되었습니다. 결과물은 '{output_dir}' 폴더를 확인하세요.")

# 실행 파이프라인
if __name__ == "__main__":
    extract_samples_from_folder(
        input_dir=constant.org_dir,
        output_dir=constant.sample_dir,
        num_records=constant.records_num
    )