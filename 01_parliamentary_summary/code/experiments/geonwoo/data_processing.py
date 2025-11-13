import json

INDEX = 83

input_path = "국회회의록안건별요약_dev.json"
output_path = f"국회회의록안건별요약_dev_sample_{INDEX}.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

sample = data[INDEX - 1]

inp = sample["input"]
issue = inp["issue"]

begin = issue["begin"]
end = issue["end"]
keyword_sentence_id = issue['sentence_id']
main_issue_id = keyword_sentence_id.split(".")[0]
keyword_sentence_int = int(keyword_sentence_id.split(".")[-1])

begin_id = keyword_sentence_id
end_id = inp["conversation"][-1]["id"]

is_begine_found = False
is_end_found = False

ids = []

for conv in sample["input"]["conversation"]:
    if "선포" in conv["utterance"]:
        id_ = conv["id"]
        id_int = id_.split(".")[-1]
        issue_id = id_.split(".")[0]

        if main_issue_id == issue_id:
            ids.append(id_)

            if not is_begine_found and int(id_int) < keyword_sentence_int:
                begin_id = id_
                is_begine_found = True
            if not is_end_found and int(id_int) > keyword_sentence_int:
                end_id = id_
                is_end_found = True

print("original begin:",begin,"end:",end)

print("keyword_sentence_id:", keyword_sentence_id)
print("begin_id:", begin_id)
print("end_id:", end_id)

print(ids)