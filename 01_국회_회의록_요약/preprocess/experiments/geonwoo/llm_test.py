"""
Ollama(qwen2.5 8B) 기반 청크 요약 실행 스크립트
ollama pull hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M:Q4_K_M
ollama pull hf.co/dnotitia/Llama-DNA-1.0-8B-Instruct-GGUF:Q6_K
ollama pull hf.co/mradermacher/KO-REAson-7B-Q2_5-0831-GGUF:Q6_K

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
	python -u experiments/geonwoo/llm_test.py 
    --processed-file "./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json" 
    --model "hf.co/dnotitia/Llama-DNA-1.0-8B-Instruct-GGUF:Q6_K" 
    --temperature 0.2 
    --num-predict 1024 
    --limit 1 
    --out "./experiments/geonwoo/summaries/blossom8b_test.json"

- 특정 N번째(1-based) 샘플만 실행
	python -u experiments/geonwoo/llm_test.py 
		--processed-file "./experiments/geonwoo/processed/processed_국회회의록안건별요약_dev.json" 
		--model "hf.co/dnotitia/Llama-DNA-1.0-8B-Instruct-GGUF:Q6_K" 
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
import re 

try:
	import requests
except ImportError as e:
	raise SystemExit("requests 패키지가 필요합니다. `pip install requests` 후 다시 실행하세요.")

with open("./experiments/geonwoo/PROMPT.txt", "r", encoding="utf-8") as f:
	PROMPT = f.read()

class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434"):
        self.base_url = host.rstrip("/")

    def generate(self, model: str, prompt: str, options: Dict[str, Any] | None = None, stream: bool = True) -> str:
        # stream=True를 기본값으로 변경
        url = f"{self.base_url}/api/generate"
        
        # Context Window 확장 (Reasoning 모델 대비)
        if options is None:
            options = {}
        if "num_ctx" not in options:
            options["num_ctx"] = 8192 

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream, # 스트리밍 활성화
            "options": options
        }

        try:
            # stream=True일 때는 iter_lines를 써야 하므로 timeout을 넉넉히 주거나 stream 모드에 맞게 처리
            with requests.post(url, json=payload, stream=stream, timeout=1200) as resp:
                resp.raise_for_status()
                
                if stream:
                    full_text = []
                    # 실시간 출력 루프
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            token = obj.get("response", "")
                            
                            # 여기서 실시간으로 화면에 출력!
                            print(token, end="", flush=True)
                            
                            full_text.append(token)
                            
                            if obj.get("done", False):
                                break
                        except Exception:
                            continue
                    
                    print() # 줄바꿈 처리
                    return "".join(full_text)
                else:
                    # stream=False일 경우 (기존 로직)
                    obj = resp.json()
                    return obj.get("response", "")

        except requests.exceptions.Timeout:
            print("\n[Error] 모델 응답 시간 초과")
            return ""
        except Exception as e:
            print(f"\n[Error] Ollama 통신 오류: {e}")
            return ""


def generate_with_retry(
    client: "OllamaClient",
    model: str,
    prompt: str,
    options: Dict[str, Any] | None,
    max_retries: int,
    sleep_sec: float,
) -> str:
    
    # [중요] 모델에게 생각을 강제하는 시스템 프롬프트나 접두어를 붙일 수도 있음
    # 예: prompt = prompt + "\n\nLet's think step by step.\n"
    
    print(f"\n[Model: {model}] 추론 및 생성 시작...")
    
    retry_count = 0
    final_summary = ""
    full_text = client.generate(model=model, prompt=prompt, options=options, stream=True)
        
        # 2. 후처리: <think>...</think> 부분 제거
    final_summary = re.sub(r'<think>.*?</think>', '', full_text, flags=re.DOTALL).strip()
    
    return final_summary.strip()

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
	max_retries: int = 10,
) -> List[Dict[str, Any]]:
	with open(processed_file, "r", encoding="utf-8") as f:
		data = json.load(f)

	outputs: List[Dict[str, Any]] = []
	client = OllamaClient()
	opts = {
    	"temperature": temperature,
    	"num_ctx": 8192,           # 문맥 길이 확보
    	"repeat_penalty": 1.2,     # [중요] 반복 방지 페널티 (보통 1.1 ~ 1.2 추천)
    	"stop": ["[END OF SUMMARY]", "[end of summary]"] # [중요] 강제 종료 단어들
	}

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

		prompts = list(map(
			lambda x: x.replace("{PROMPT}", PROMPT), prompts
		))

		# Optional debug printing is controlled by --print-prompts
		if print_prompts:
			print(f"[debug] prompts count: {len(prompts)}")

		if len(prompts) == 0:
			final_summary = ""
		else:
			# 단일 루프로 청크 1~N 처리 (1은 워밍업, 2~N은 누적)
			s1 = ""
			for ci in range(1, len(prompts) + 1):
				prompt = prompts[ci - 1]
				if ci == 1:
					filled_prompt = prompt.replace("{previous_summary}", "없음")
					filled_prompt = filled_prompt.replace("{step}", ["중간", "최종"][(processed_chunks+1)//len(prompts)])
					if print_prompts:
						prev_len = len("없음")
						print(f"\n--- CHUNK {ci} / {len(prompts)} ---")
						print(f"[prev_summary length]: {prev_len}")
						print("[prompt head]:")
						print(filled_prompt[:500])
				else:
					filled_prompt = prompt.replace("{previous_summary}", prev_summary)
					filled_prompt = filled_prompt.replace("{step}", ["중간", "최종"][(processed_chunks+1)//len(prompts)])
					if print_prompts:
						prev_len = len(prev_summary)
						print(f"\n--- CHUNK {ci} / {len(prompts)} ---")
						print(f"[prev_summary length]: {prev_len}")
						print("[prompt head]:")
						print(filled_prompt[:500])

				summary = generate_with_retry(
					client=client,
					model=model,
					prompt=filled_prompt,
					options=opts,
					max_retries=max_retries,
					sleep_sec=sleep_sec,
				)

				if ci == 1:
					s1 = summary
					prev_summary = summary
				else:
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
		default="hf.co/mradermacher/KO-REAson-7B-Q2_5-0831-GGUF:Q6_K",
		help="Ollama 모델 태그 (예: qwen2.5:8b, qwen2.5:7b-instruct 등)",
	)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--num-predict", type=int, default=None, help="최대 생성 토큰")
	parser.add_argument("--limit", type=int, default=None, help="처리할 샘플 개수 제한")
	parser.add_argument("--index", type=int, default=3, help="요약할 N번째 샘플 (1-based). 지정 시 limit 무시")
	parser.add_argument("--print-prompts", action="store_true", help="치환된 프롬프트(앞 500자)와 이전 요약 길이를 출력")
	parser.add_argument("--no-progress", action="store_true", help="진행률/ETA 출력 비활성화")
	parser.add_argument("--max-retries", type=int, default=10, help="요약문에 금지 패턴(\\n, '요약문')이 포함될 경우 재추론 시도 횟수")
	parser.add_argument(
		"--out",
		default="./experiments/geonwoo/summaries/result.json",
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
		max_retries=args.max_retries,
	)

	with open(args.out, "w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)

	print(f"✅ 요약 결과 저장 완료: {args.out}")


if __name__ == "__main__":
	main()

