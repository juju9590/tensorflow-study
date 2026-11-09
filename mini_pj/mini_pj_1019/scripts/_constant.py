import os
import glob
import json
import re
import random
from bs4 import BeautifulSoup
from openai import OpenAI


target = "2025"

org_dir="./data/org2/" + target + "_광역 회의록"          # 원본 18개의 JSON 파일이 있는 폴더 경로
sample_dir="./data/sample/" + target + "_광역 회의록"     # 샘플 TXT 파일들이 저장될 폴더 경로
records_num=10                                 # 각 파일당 추출할 회의록 레코드 개수            

rules_output_dir="./data/llm_rules_outputs/광역 회의록"
all_rule_file="./data/llm_rules_outputs/consolidated_parsing_rules2.json"


text_dir = "./data/to_text2/" + target + "_광역 회의록"



make_model_name="qwen2.5-coder:7b" 

# model_name="hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:Q4_K_M"
# model_name="richardyoung/qwen3-4b-instruct-2507-abliterated:latest" 
# model_name="qwen2.5-coder:7b" 
# model_name="huihui_ai/exaone3.5-abliterated" 
# model_name="gemma4:12b" 
# model_name="qwen2.5-coder:14b"

client = OpenAI(
    base_url="http://localhost:11434/v1", 
    api_key="not-needed" 
)