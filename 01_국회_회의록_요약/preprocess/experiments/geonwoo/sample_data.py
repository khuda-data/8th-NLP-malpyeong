import json

# dev 파일 경로 설정
input_path = "국회회의록안건별요약_dev.json"
output_path = "국회회의록안건별요약_dev_sample.json"

INDEX = 36

# JSON 파일 읽기
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 첫 번째 항목 추출
sample = data[INDEX - 1]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)