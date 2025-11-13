# percent.py
import json
import pandas as pd
from itertools import combinations
from konlpy.tag import Okt

# ===== 설정 =====
CONTEXT_PATH   = "sample_context.json"   # 문장 리스트(JSON)
CSV_PATH       = "word_frequency_-1,0,1.csv"    # 단어,빈도 CSV
TOP_N_WORDS     = 30                      # 상위 단어 수
MIN_COMB_SIZE   = 2                     # 최소 조합 크기
MAX_COMB_SIZE   = 4                       # 최대 조합 크기
TOP_K_RESULTS   = 10                      # 출력 상위 개수

okt = Okt()

# ===== 데이터 로드 =====
df = pd.read_csv(CSV_PATH)
top_words = df["단어"].head(TOP_N_WORDS).astype(str).tolist()

with open(CONTEXT_PATH, "r", encoding="utf-8") as f:
    sentences = json.load(f)
if not isinstance(sentences, list):
    raise ValueError("sample_context.json은 문자열 리스트여야 합니다.")
num_sents = len(sentences)
if num_sents == 0:
    raise ValueError("문장이 없습니다.")

# ===== 문장 형태소화 및 역색인 구축 =====
# postings[word] = 해당 단어가 등장한 문장 인덱스 집합
postings = {w: set() for w in top_words}
for i, s in enumerate(sentences):
    morphs = set(okt.morphs(s, norm=True, stem=True))
    for w in top_words:
        if w in morphs:
            postings[w].add(i)

# ===== 커버리지 계산 함수 =====
def coverage_count(tokens):
    it = iter(tokens)
    try:
        acc = postings[next(it)].copy()
    except StopIteration:
        return 0
    for t in it:
        acc &= postings[t]
        if not acc:
            return 0
    return len(acc)

# ===== 후보 생성: MIN~MAX 조합 =====
candidates = []
for k in range(MIN_COMB_SIZE, MAX_COMB_SIZE + 1):
    candidates.extend(combinations(top_words, k))

# ===== 결과 계산 =====
results = []
for cand in candidates:
    cnt = coverage_count(cand)
    pct = (cnt / num_sents) * 100
    results.append({
        "tokens": cand,
        "count": cnt,
        "pct": pct,
        "k": len(cand)
    })

# ===== 정렬: 퍼센트↓ → count↓ → 단어수↑ → 사전순 =====
results.sort(key=lambda x: (-x["pct"], -x["count"], x["k"], tuple(x["tokens"])))

# ===== 출력 =====
for r in results[:TOP_K_RESULTS]:
    tokens_str = ", ".join(r["tokens"])
    print(f"{tokens_str} = {round(r['pct'])}%")