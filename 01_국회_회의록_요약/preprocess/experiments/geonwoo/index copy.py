import json
import re


def extract_n(text: str):
    """
    '의사일정 제N항'에서 N을 정수로 추출.
    여러 개가 있을 경우 리스트로 반환.
    """
    matches = re.findall(r"의사\s*일정\s*제\s*(\d+)\s*항", text)
    return [int(n) for n in matches] if matches else []

# dev 파일 경로 설정
input_path = "국회회의록안건별요약_dev.json"
output_path = "국회회의록안건별요약_dev_sample.json"

# JSON 파일 읽기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 첫 번째 항목 추출
sample = data[2]
# print(len(sample["input"]["conversation"]))
# print(sample["input"]["issue"])

inp = sample["input"]
issue = inp["issue"]

begin = issue["begin"]
end = issue["end"]
keyword_sentence_id = issue['sentence_id']
main_issue_id = keyword_sentence_id.split(".")[0]
keyword_sentence_int = int(keyword_sentence_id.split(".")[-1])

begin_id = keyword_sentence_id
end_id = keyword_sentence_id
ids = []


for item in data:
    k_id = item["input"]["issue"]["sentence_id"]
    k_ids = []
    convs = []
    
    for i in range(0, 2):
        k_ids.append(".".join(k_id.split(".")[:-1]) + "." + str(int(k_id.split(".")[-1]) - i))

    for i in range(1, 2):
        k_ids.append(".".join(k_id.split(".")[:-1]) + "." + str(int(k_id.split(".")[-1]) + i))

    # print(k_ids)
    for k_id_ in k_ids:
        for conv in item["input"]["conversation"]:
            if conv["id"] == k_id_:
                convs.append(extract_n(conv["utterance"]))

    ids.append(convs)

print(ids)

# print("\n".join(list(map(lambda x: x["utterance"],ids[0:50]))))

# with open("./sample_context.json","w",encoding="utf-8") as f:
#     json.dump(list(map(lambda x: x["utterance"],ids)), f, ensure_ascii=False, indent=2)
    

# for item in data:
#     for conv in item["input"]["conversation"]:
#         if "선포" in conv["utterance"]:

#             id_ = conv["id"]
#             id_int = id_.split(".")[-1]
#             issue_id = id_.split(".")[0]

#             if main_issue_id == issue_id:
#                 ids.append(id_)
#                 if int(id_int) - keyword_sentence_int > 0:
#                     if end_id == keyword_sentence_id:
#                         end_id = id_
#                 else:
#                     begin_id = id_

# # 결과 출력
# # print(ids)
# print("original begin:",begin,"end:",end)

# print("keyword_sentence_id:", keyword_sentence_id)
# print("begin_id:", begin_id)
# print("end_id:", end_id)

# print(ids)