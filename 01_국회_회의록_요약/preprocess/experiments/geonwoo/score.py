from collections import Counter
import json

def rouge1_f1(pred: str, ref: str):
    """
    pred: 모델이 생성한 요약 문장
    ref : 정답(참조) 요약 문장
    return: (precision, recall, f1)
    """
    # 공백 기준 토크나이즈 (필요하면 여기만 바꿔서 형태소/wordpiece 등 사용)
    pred_tokens = pred.strip().split()
    ref_tokens = ref.strip().split()

    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0, 0.0, 0.0

    pred_cnt = Counter(pred_tokens)
    ref_cnt = Counter(ref_tokens)

    # 겹치는 unigram 수 (multiset 교집합)
    overlap = sum((pred_cnt & ref_cnt).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

# 예시
if __name__ == "__main__":
    with open("./experiments/geonwoo/summaries/result.json", "r", encoding="utf-8") as f:
        outputs = json.load(f)
    for i, output in enumerate(outputs):
        ref = output["gold_output"]
        pred = output["final_summary"]
        p, r, f1 = rouge1_f1(pred, ref)
        print(f"[Sample {i+1}] ROUGE-1 precision:", p)
        print(f"[Sample {i+1}] ROUGE-1 recall:", r)
        print(f"[Sample {i+1}] ROUGE-1 F1:", f1)
        print()