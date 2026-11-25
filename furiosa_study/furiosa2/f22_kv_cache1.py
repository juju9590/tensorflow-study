from kingkeras import run_once

short_prompt = 'npu란 무엇인가?'
long_prompt =(
    "npu는 인공지능 연산을 가속하는 프로세서다" * 100
)+ "위 내용을 요약해줘."

tests=[
    ("짧은 입력:", short_prompt, 80),
    ('긴 입력:', long_prompt, 80),
    ('긴 입력과 긴 출력:',long_prompt,400),
]

for name, prompt, max_tokens in tests:
    result = run_once(prompt, max_tokens)

    print(f'\n실험:{name}')
    print('입력 토큰:',result['input_tokens'])
    print('출력 토큰:',result['output_tokens'])
    print('소요 토큰:',round(result['elapsed'],2))

    input("서버 kv cache 사용률을 확인하고 싶다면 엔터를 쳐라")

