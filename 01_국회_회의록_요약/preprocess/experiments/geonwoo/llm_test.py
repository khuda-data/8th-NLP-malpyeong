"""
Ollama(qwen2.5 8B) 기반 청크 요약 실행 스크립트
ollama pull hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:Q4_K_M

사용 시나리오
- data_processing.py가 생성한 processed JSON을 입력으로 받아, 각 청크를 순차 요약합니다.
- 각 청크 프롬프트의 {previous_summary} 자리에는 직전 청크의 요약을 넣어 연쇄 요약을 수행합니다.
 - 누적 요약 방식: 첫 두 청크(1,2)로 1차 요약 생성 후, 이후 청크를 순차적으로 합쳐가며 갱신합니다.

사전 준비
1) Ollama 설치 및 서버 실행 (기본 호스트: http://localhost:11434)
	 - 모델 다운로드:  ollama pull qwen2.5:8b
2) Python 의존성 설치
	 - pip install requests

입력 파일
- 기본값: ./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json
- data_processing.py에서 생성됩니다. (CONFIG_PROMPT에 {previous_summary}, {dialogue}가 포함)

실행 방법 (예시)
- 기본 실행
	python -u experiments/geonwoo/llm_test.py

- 옵션 지정 실행
	python -u experiments/geonwoo/llm_test.py \
    --processed-file "./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json" \
    --model "hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:Q4_K_M" \
    --temperature 0.2 \
    --num-predict 1024 \
    --limit 1 \
    --out "./experiments/geonwoo/summaries/blossom8b_test.json"

- 특정 N번째(1-based) 샘플만 실행
	python -u experiments/geonwoo/llm_test.py \
		--processed-file "./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json" \
		--model "qwen2.5:8b" \
		--index 5


출력
- 요약 결과 JSON (기본): ./experiments/geonwoo/summaries/summaries_국회회의록안건별요약_dev_qwen2.5-8b.json
- 포함 항목: id, chunk_summaries(누적 요약 목록), final_summary(마지막 누적 요약), gold_output

기타
- Ollama가 다른 호스트/포트라면 OllamaClient(host="http://127.0.0.1:11434")로 조정하세요.
- 모델 태그는 환경에 설치된 태그로 변경 가능합니다. (예: qwen2.5:8b-instruct)
 - 진행 상황: 처리한 청크/전체 청크 기반 퍼센트와 ETA를 출력합니다. (비활성화: --no-progress)
"""
import os
import json
import time
import argparse
from typing import List, Dict, Any

try:
	import requests
except ImportError as e:
	raise SystemExit("requests 패키지가 필요합니다. `pip install requests` 후 다시 실행하세요.")


class OllamaClient:
	def __init__(self, host: str = "http://localhost:11434"):
		self.base_url = host.rstrip("/")

	def generate(self, model: str, prompt: str, options: Dict[str, Any] | None = None, stream: bool = False) -> str:
		url = f"{self.base_url}/api/generate"
		payload = {
			"model": model,
			"prompt": prompt,
			"stream": stream,
		}
		if options:
			payload["options"] = options

		resp = requests.post(url, json=payload, timeout=600)
		resp.raise_for_status()

		if stream:
			text = []
			for line in resp.iter_lines(decode_unicode=True):
				if not line:
					continue
				try:
					obj = json.loads(line)
					text.append(obj.get("response", ""))
				except Exception:
					continue
			return "".join(text)
		else:
			obj = resp.json()
			return obj.get("response", "")



def run_chunk_summarization(
	processed_file: str,
	model: str,
	temperature: float = 0.2,
	num_predict: int | None = None,
	limit: int | None = None,
	sleep_sec: float = 0.2,
	print_prompts: bool = False,
    show_progress: bool = True,
    index: int | None = None,
) -> List[Dict[str, Any]]:
	with open(processed_file, "r", encoding="utf-8") as f:
		data = json.load(f)

	outputs: List[Dict[str, Any]] = []
	client = OllamaClient()
	opts = {"temperature": temperature}
	if num_predict is not None:
		opts["num_predict"] = num_predict

	# 데이터 서브셋 구성: index(1-based) 우선, 없으면 limit 적용
	if index is not None:
		if index < 1:
			raise SystemExit("--index 는 1 이상의 정수여야 합니다 (1-based).")
		idx0 = index - 1
		if idx0 >= len(data):
			raise SystemExit(f"--index 값 {index} 가 범위를 벗어났습니다. (데이터 크기: {len(data)})")
		dataset = [data[idx0]]
	else:
		dataset = data[:limit] if limit is not None else data

	# 전체 청크 수 계산 (ETA/퍼센트 산출용)
	total = len(dataset)
	total_chunks = 0
	for sample in dataset:
		prompts: List[str] = sample.get("inputs", [])
		total_chunks += len(prompts)
	processed_chunks = 0
	start_ts = time.time()
	for idx, sample in enumerate(dataset, start=1):
		sample_id = sample.get("id")
		prompts: List[str] = sample.get("inputs", [])
		gold_output = sample.get("outputs")

		prev_summary = ""
		chunk_summaries: List[str] = []  # 누적 요약 단계별 결과 저장

		if len(prompts) == 0:
			final_summary = ""
		else:
			# 1) 청크1만 요약 (워밍업), 저장하지 않음
			filled_prompt = prompts[0].replace("{previous_summary}", "없음")
			if print_prompts:
				prev_len = len("없음")
				print(f"\n--- CHUNK 1 / {len(prompts)} ---")
				print(f"[prev_summary length]: {prev_len}")
				print("[prompt head]:")
				print(filled_prompt[:500])
			s1 = client.generate(model=model, prompt=filled_prompt, options=opts, stream=False).strip()
			time.sleep(sleep_sec)
			processed_chunks += 1
			if show_progress and total_chunks > 0:
				elapsed = time.time() - start_ts
				rate = processed_chunks / elapsed if elapsed > 0 else 0.0
				remaining = max(total_chunks - processed_chunks, 0)
				eta_sec = int(remaining / rate) if rate > 0 else 0
				eta_min = eta_sec // 60
				eta_rem = eta_sec % 60
				percent = (processed_chunks / total_chunks) * 100.0
				print(
					f"Progress: {processed_chunks}/{total_chunks} chunks ({percent:.1f}%) | "
					f"sample {idx}/{total} | chunk 1/{len(prompts)} | ETA {eta_min:02d}:{eta_rem:02d}"
				)

			# 2) 청크2부터 누적 요약 시작
			if len(prompts) >= 2:
				# 요약1 = s1 + 청크2
				filled_prompt = prompts[1].replace("{previous_summary}", s1)
				if print_prompts:
					prev_len = len(s1)
					print(f"\n--- CHUNK 2 / {len(prompts)} ---")
					print(f"[prev_summary length]: {prev_len}")
					print("[prompt head]:")
					print(filled_prompt[:500])
				summary = client.generate(model=model, prompt=filled_prompt, options=opts, stream=False).strip()
				chunk_summaries.append(summary)
				prev_summary = summary
				time.sleep(sleep_sec)
				processed_chunks += 1
				if show_progress and total_chunks > 0:
					elapsed = time.time() - start_ts
					rate = processed_chunks / elapsed if elapsed > 0 else 0.0
					remaining = max(total_chunks - processed_chunks, 0)
					eta_sec = int(remaining / rate) if rate > 0 else 0
					eta_min = eta_sec // 60
					eta_rem = eta_sec % 60
					percent = (processed_chunks / total_chunks) * 100.0
					print(
						f"Progress: {processed_chunks}/{total_chunks} chunks ({percent:.1f}%) | "
						f"sample {idx}/{total} | chunk 2/{len(prompts)} | ETA {eta_min:02d}:{eta_rem:02d}"
					)

				# 이후 청크3..N: 누적 요약 + 다음 청크
				for ci in range(3, len(prompts) + 1):
					prompt = prompts[ci - 1]
					filled_prompt = prompt.replace("{previous_summary}", prev_summary)
					if print_prompts:
						prev_len = len(prev_summary)
						print(f"\n--- CHUNK {ci} / {len(prompts)} ---")
						print(f"[prev_summary length]: {prev_len}")
						print("[prompt head]:")
						print(filled_prompt[:500])
					summary = client.generate(model=model, prompt=filled_prompt, options=opts, stream=False).strip()
					chunk_summaries.append(summary)
					prev_summary = summary
					time.sleep(sleep_sec)
					processed_chunks += 1
					if show_progress and total_chunks > 0:
						elapsed = time.time() - start_ts
						rate = processed_chunks / elapsed if elapsed > 0 else 0.0
						remaining = max(total_chunks - processed_chunks, 0)
						eta_sec = int(remaining / rate) if rate > 0 else 0
						eta_min = eta_sec // 60
						eta_rem = eta_sec % 60
						percent = (processed_chunks / total_chunks) * 100.0
						print(
							f"Progress: {processed_chunks}/{total_chunks} chunks ({percent:.1f}%) | "
							f"sample {idx}/{total} | chunk {ci}/{len(prompts)} | ETA {eta_min:02d}:{eta_rem:02d}"
						)

			final_summary = (chunk_summaries[-1] if len(chunk_summaries) > 0 else s1)

		outputs.append(
			{
				"id": sample_id,
				"chunk_summaries": chunk_summaries,
				"final_summary": final_summary,
				"gold_output": gold_output,
			}
		)

		print(f"[{idx}/{total}] done: {sample_id}")

	return outputs


def main():
	parser = argparse.ArgumentParser(description="Chunked summarization via Ollama (qwen2.5 8B)")
	parser.add_argument(
		"--processed-file",
		default="./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json",
		help="data_processing.py가 생성한 processed JSON 경로",
	)
	parser.add_argument(
		"--model",
		default="qwen2.5:8b",
		help="Ollama 모델 태그 (예: qwen2.5:8b, qwen2.5:7b-instruct 등)",
	)
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--num-predict", type=int, default=None, help="최대 생성 토큰")
	parser.add_argument("--limit", type=int, default=None, help="처리할 샘플 개수 제한")
	parser.add_argument("--index", type=int, default=None, help="요약할 N번째 샘플 (1-based). 지정 시 limit 무시")
	parser.add_argument("--print-prompts", action="store_true", help="치환된 프롬프트(앞 500자)와 이전 요약 길이를 출력")
	parser.add_argument("--no-progress", action="store_true", help="진행률/ETA 출력 비활성화")
	parser.add_argument(
		"--out",
		default="./experiments/geonwoo/summaries/summaries_국회회의록안건별요약_dev_qwen2.5-8b.json",
		help="요약 결과 저장 경로",
	)

	args = parser.parse_args()

	os.makedirs(os.path.dirname(args.out), exist_ok=True)

	results = run_chunk_summarization(
		processed_file=args.processed_file,
		model=args.model,
		temperature=args.temperature,
		num_predict=args.num_predict,
		limit=args.limit,
		print_prompts=args.print_prompts,
        show_progress=not args.no_progress,
        index=args.index,
	)

	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)

	print(f"✅ 요약 결과 저장 완료: {args.out}")


if __name__ == "__main__":
	main()

