import json
import re
import pandas as pd
from konlpy.tag import Okt
from collections import Counter

# 형태소 분석기 초기화
okt = Okt()

# JSON 파일 로드 (문장 리스트)
with open("./sample_context.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# 한글 2글자 이상만 허용
two_or_more_korean = re.compile(r"^[가-힣]{2,}$")

tokens = []
for s in sentences:
    for word in okt.morphs(s, norm=True, stem=True):
        if two_or_more_korean.match(word):  # 2글자 이상 한글만
            tokens.append(word)

# 등장 빈도 계산
counter = Counter(tokens)

# ✅ JSON 파일로 저장
data_to_save = {
    "total_tokens": tokens,              # 전체 토큰 리스트
    "token_count": len(tokens),          # 전체 토큰 개수
    "unique_token_count": len(counter),  # 고유 단어 수
    "frequency": counter.most_common()   # (단어, 빈도) 리스트
}

with open("token_data.json", "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=2)

# ✅ CSV 파일로 저장 (단어, 빈도)
df = pd.DataFrame(counter.most_common(), columns=["단어", "빈도"])
df.to_csv("word_frequency.csv", index=False, encoding="utf-8-sig")

print("✅ JSON & CSV 파일 저장 완료!")
print(f"총 토큰 수: {len(tokens)}개 / 고유 단어 수: {len(counter)}개")
