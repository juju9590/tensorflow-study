import os
import glob
import json
import re
from bs4 import BeautifulSoup
import pandas as pd

class DynamicCouncilPreprocessor:
    def __init__(self, min_char_len: int = 30):
        self.min_char_len = min_char_len
        self.formality_patterns = [
            # r"존경하는\s+[^\s!]+(?:\s+여러분)?",
            # r"안녕하십니까\?",
            # r"감사합니다\.",
            # r"수고\s*(?:많으셨습니다|하셨습니다)\.",
            # r"이의가?\s+없으십니까\?",
            # r"이의가\s+없으므로\s+[^\.]+\.",
            # r"성원이\s+되었으므로\s+[^\.]+\.",
            # r"산회를\s+선포합니다\.",
            # r"개의를\s+선포합니다\.",
            r"\([^\)]*(?:개식|폐식|개의|산회|인사|기립|묵념|박수|마이크)[^\)]*\)",
        ]
        self.combined_regex = re.compile("|".join(self.formality_patterns), re.MULTILINE)

    def clean_text(self, text: str) -> str:
        cleaned = self.combined_regex.sub(" ", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def parse_speaker(self, raw_speaker: str) -> tuple:
        text = re.sub(r"^[○□■\-\s]+", "", raw_speaker).strip()
        tokens = text.split()
        
        if len(tokens) >= 2:
            role, name = tokens[0], "".join(tokens[1:])
        elif len(tokens) == 1:
            role, name = "미상", tokens[0]
        else:
            return "미상", "미상", "기타"

        speaker_type = "의회" if any(r in role for r in ["의원", "의장", "부의장", "위원장", "부위원장", "위원"]) else "집행부"
        return role, name, speaker_type

    def process_record(self, record: dict, rule: dict, filename_info: str) -> list:
        results = []
        html_content = record.get("MINTS_HTML", "")
        if not html_content:
            return results

        soup = BeautifulSoup(html_content, "html.parser")

        for tag in soup(["script", "style", "form", "table", "ul.skipUl", "#topFunction", "#mySidenav2"]):
            tag.decompose()

        agenda_selector = rule.get("agenda", "div")
        speaker_selector = rule.get("speaker", "span.spkrbld")
        utterance_selector = rule.get("utterance", "p")

        try:
            speaker_tags = soup.select(speaker_selector)
            agenda_tags = soup.select(agenda_selector)
        except Exception:
            return results

        # [핵심 디버깅 및 자동 복구 로직]
        if len(speaker_tags) == 0:
            print(f"  [경고] '{filename_info}'에서 발언자({speaker_selector})를 찾지 못해 기본값(span.spkrbld)으로 재시도합니다.")
            
            # LLM이 이상한 태그를 줬을 때, 가이드라인의 표준 태그로 강제 변경
            speaker_selector = "span.spkrbld"
            try:
                speaker_tags = soup.select(speaker_selector)
            except Exception:
                pass
            
            # 기본 태그로도 못 찾으면 진짜 구조가 특이한 파일이므로 스킵
            if len(speaker_tags) == 0:
                print(f"  [실패] 기본값으로도 '{filename_info}'의 발언자를 찾을 수 없습니다. 건너뜁니다.")
                return results

        for spk_tag in speaker_tags:
            raw_speaker = spk_tag.get_text(strip=True)
            role, name, speaker_type = self.parse_speaker(raw_speaker)

            agenda_title = "일반안건"
            for prev in spk_tag.find_all_previous():
                if prev in agenda_tags:
                    agenda_title = prev.get_text(strip=True)[:50]
                    break

            # --- 본문 추출 로직 강화 ---
            utterance_cleaned = ""
            
            # 1. LLM이 utterance 규칙(예: div.speech-content)을 명시한 경우, 발언자 태그 '바로 다음'에 오는 해당 태그를 찾음
            if utterance_selector and utterance_selector != "p":
                next_content = spk_tag.find_next(utterance_selector)
                if next_content:
                    utterance_cleaned = self.clean_text(next_content.get_text(strip=True))

            # 2. 위 방법으로 못 찾았거나 기본 'p' 규칙인 경우, 기존처럼 부모 영역 긁어오기
            if not utterance_cleaned or len(utterance_cleaned) < 10:
                parent_block = spk_tag.find_parent("div") or spk_tag.parent
                spk_tag.decompose() 
                utterance_raw = parent_block.get_text(strip=True) if parent_block else ""
                utterance_cleaned = self.clean_text(utterance_raw)

            if len(utterance_cleaned) >= self.min_char_len:
                results.append({
                    "date": record.get("MTG_DE", ""),
                    "assembly": record.get("RASMBLY_NM", ""),
                    "agenda_title": agenda_title,
                    "speaker_role": role,
                    "speaker_name": name,
                    "speaker_type": speaker_type,
                    "utterance": utterance_cleaned
                })
        return results

# 실행 파이프라인
if __name__ == "__main__":
    input_dir = "./data/org/2024_광역 회의록"
    output_dir = "./data/to_text2/2024_광역 회의록2"
    rules_file = "./data/llm_rules_outputs/consolidated_parsing_rules2.json"

    os.makedirs(output_dir, exist_ok=True)

    with open(rules_file, "r", encoding="utf-8") as f:
        parsing_rules = json.load(f)

    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    preprocessor = DynamicCouncilPreprocessor(min_char_len=30)

    for input_file in json_files:
        base_name = os.path.basename(input_file)
        file_key = os.path.splitext(base_name)[0]
        
        matched_rule = parsing_rules.get(file_key, {"agenda": "div", "speaker": "span.spkrbld", "utterance": "p"})

        print(f"[{base_name}] 전처리 시작...")
        
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                
            all_utterances = []
            for record in raw_data:
                # 파일명 정보를 에러 로그용으로 넘김
                all_utterances.extend(preprocessor.process_record(record, matched_rule, base_name))

            df_clean = pd.DataFrame(all_utterances)
            
            if not df_clean.empty:
                output_path = os.path.join(output_dir, f"cleaned_{file_key}.txt")
                df_clean.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
                print(f"  -> 완료: {len(df_clean)}건 추출 ({output_path})")
            else:
                print(f"  -> 추출된 유효 데이터 없음")
        except Exception as e:
             print(f"  -> 파일 처리 중 치명적 오류 발생: {e}")