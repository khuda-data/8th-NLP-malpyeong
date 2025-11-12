import json

INDEX = 15

#############################################
input_path = "국회회의록안건별요약_dev.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)
sample = data[INDEX]
#############################################

output = []

for sample in data:
    conversation = sample["input"]["conversation"]
    data_ = {
        "id": sample["id"],
        "conversation": []
    }

    for conv in conversation:
        # if "의사일정 제" in conv["utterance"] and "상정하겠습니다" in conv["utterance"]:
        #     data_["conversation"].append({
        #         "id": conv["id"],
        #         "utterance": conv["utterance"]
        #     })

        if "선포" in conv["utterance"]:
            data_["conversation"].append({
                "id": conv["id"],
                "utterance": conv["utterance"]
            })

        # if "마치" in conv["utterance"]:
        #     data_["conversation"].append({
        #         "id": conv["id"],
        #         "utterance": conv["utterance"]
        #     })

    output.append(data_)

with open(f"./sample_extracted_{INDEX}.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)