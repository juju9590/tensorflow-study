# pip install beautifulsoup4


from pathlib import Path
from bs4 import BeautifulSoup

# 1. HTML 샘플 파일 경로
file_path = Path(
    "C:/tensorflow_study/mini_pj/data/html_samples/"
    "2025_강원특별자치도의회/sample_1.html"
)

# 2. HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html = f.read()

# 3. HTML 분석
soup = BeautifulSoup(html, "html.parser")

# 4. 불필요한 태그 제거
for tag in soup(["script", "style"]):
    tag.decompose()

# 5. HTML에서 텍스트 추출
text = soup.get_text("\n", strip=True)

# 6. 홈페이지 불필요 문구 제거
remove_text = "강원특별자치도의회 의원프로필"

if remove_text in text:
    text = text.split(remove_text)[0]

# 7. 결과 확인
print("정리 후 글자 수:", len(text))

print("\n===== 마지막 500자 =====")
print(text[-500:])
