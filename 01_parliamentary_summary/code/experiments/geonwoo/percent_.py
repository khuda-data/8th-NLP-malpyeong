import json
import pandas as pd
from itertools import combinations
from konlpy.tag import Okt

# ===== 설정 =====
CONTEXT_PATH   = "sample_context.json"   # 문장 리스트(JSON)
CSV_PATH       = "word_frequency_-1,0,1.csv"    # 단어,빈도 CSV
TOP_N_WORDS     = 30                      # 상위 단어 수
MIN_COMB_SIZE   = 2                       # 최소 조합 크기
MAX_COMB_SIZE   = 4                       # 최대 조합 크기
TOP_K_RESULTS   = 10                      # 출력 상위 개수

# 제외할 콤보 (예시)
SELECTED_COMBOS = [["의사", "일정"], ["위원"]]

# 선택 콤보에 포함된 단어들을 후보 단어 풀에서 제외할지 여부
EXCLUDE_SELECTED_TOKENS_FROM_VOCAB = True

# ===== 형태소 분석기 =====
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

# ===== 문장 형태소화 및 역색인(postings) 구축 =====
postings = {w: set() for w in top_words}
for i, s in enumerate(sentences):
    morphs = set(okt.morphs(s, norm=True, stem=True))
    for w in top_words:
        if w in morphs:
            postings[w].add(i)

# ===== 유틸: 한 콤보가 모두 포함된 문장 인덱스 집합 =====
def included_indices_for_combo(tokens):
    tokens = [t for t in tokens if t in postings]  # 상위 단어 집합 밖 단어는 무시
    if not tokens:
        return set()
    it = iter(tokens)
    acc = postings[next(it)].copy()
    for t in it:
        acc &= postings[t]
        if not acc:
            return set()
    return acc

# ===== 제외 인덱스: 모든 콤보의 "포함 문장"을 합집합으로 제외 =====
# SELECTED_COMBOS가 ["a","b"] 같은 평탄 리스트로 들어오는 경우도 허용
def normalize_selected_combos(x):
    if not x:
        return []
    # 평탄 리스트(str의 리스트)면 1개 콤보로 해석
    if all(isinstance(t, str) for t in x):
        return [list(x)]
    # 리스트의 리스트면 그대로
    if all(isinstance(t, (list, tuple)) for t in x):
        return [list(c) for c in x]
    raise ValueError("SELECTED_COMBOS 형식 오류: 리스트 또는 리스트의 리스트여야 합니다.")

SELECTED_COMBOS = normalize_selected_combos(SELECTED_COMBOS)

excluded_idx = set()
for combo in SELECTED_COMBOS:
    excluded_idx |= included_indices_for_combo(combo)

residual_idx = set(range(num_sents)) - excluded_idx
num_residual = len(residual_idx)
if num_residual == 0:
    print("잔여 문장이 없습니다. 종료합니다.")
    raise SystemExit

# ===== 잔여 영역에서의 postings 재구성 =====
postings_res = {w: (postings[w] & residual_idx) for w in top_words}

# ===== 커버리지 계산(잔여 분모) =====
def coverage_count_res(tokens):
    tokens = [t for t in tokens if t in postings_res]
    if not tokens:
        return 0
    it = iter(tokens)
    acc = postings_res[next(it)].copy()
    for t in it:
        acc &= postings_res[t]
        if not acc:
            return 0
    return len(acc)

# ===== 후보 단어 풀 구성 =====
if EXCLUDE_SELECTED_TOKENS_FROM_VOCAB:
    selected_token_set = set(t for combo in SELECTED_COMBOS for t in combo)
    candidate_vocab = [w for w in top_words if w not in selected_token_set]
else:
    candidate_vocab = top_words

# ===== 후보 생성 및 평가 =====
candidates = []
for k in range(MIN_COMB_SIZE, MAX_COMB_SIZE + 1):
    candidates.extend(combinations(candidate_vocab, k))

results = []
for cand in candidates:
    cnt = coverage_count_res(cand)
    pct = (cnt / num_residual) * 100
    results.append({
        "tokens": cand,
        "count": cnt,
        "pct": pct,
        "k": len(cand)
    })

# ===== 정렬 및 출력 =====
results.sort(key=lambda x: (-x["pct"], -x["count"], x["k"], tuple(x["tokens"])))

print(f"총 문장 수: {num_sents}, 제외된 문장: {len(excluded_idx)}, 잔여 문장: {num_residual}")
if SELECTED_COMBOS:
    print("제외 기준 콤보:", ", ".join(["+".join(c) for c in SELECTED_COMBOS]))
print(f"상위 {TOP_K_RESULTS}개 조합:")
for r in results[:TOP_K_RESULTS]:
    print(f'{", ".join(r["tokens"])} = {round(r["pct"])}%  ({r["count"]}/{num_residual})')