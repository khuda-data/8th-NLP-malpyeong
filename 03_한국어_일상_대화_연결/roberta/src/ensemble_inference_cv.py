"""
Cross Validation 모델 Ensemble Inference 스크립트 (개선 버전)
- 선행문 5-fold + 후행문 5-fold = 총 10개 모델 사용
- 각 샘플에 대해 정방향 + 역방향 평가 수행
- Soft Voting (평균 확률) 방식으로 앙상블
"""

import os
import sys
import json
import glob
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Dict, List

from data_loader import TestDialogueDataset, NegationDetector, EmotionAnalyzer
from model import DialogueConnectionModel
from config import Config


def find_cv_models(job_id: str, output_dir: str = './models'):
    """Cross Validation 모델 파일들을 자동으로 찾기
    
    Args:
        job_id: SLURM Job ID (예: '58583')
        output_dir: 모델이 저장된 디렉토리
    
    Returns:
        dict: {
            '선행문': [모델경로1, 모델경로2, ...],
            '후행문': [모델경로1, 모델경로2, ...]
        }
    """
    models = {
        '선행문': [],
        '후행문': []
    }
    
    for position in ['선행문', '후행문']:
        # 패턴: best_model_선행문_fold0_58583.pt
        pattern = os.path.join(output_dir, f'best_model_{position}_fold*_{job_id}.pt')
        model_files = sorted(glob.glob(pattern))
        
        if not model_files:
            print(f"⚠️  경고: {position} 모델을 찾을 수 없습니다.")
            print(f"   패턴: {pattern}")
        else:
            models[position] = model_files
            print(f"✓ {position} 모델 {len(model_files)}개 발견:")
            for mf in model_files:
                print(f"  - {os.path.basename(mf)}")
    
    return models


def load_single_model(model_path: str, model_name: str, tokenizer, device: str):
    """단일 모델 로드"""
    config = Config()
    model = DialogueConnectionModel(model_name=model_name, dropout_rate=config.DROPOUT_RATE)
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model, checkpoint


class ReversedTestDialogueDataset(Dataset):
    """순서를 반대로 한 테스트 데이터셋 클래스
    
    선행문 → 후행문으로, 후행문 → 선행문으로 변환하여 평가
    즉, 문장 순서를 바꿔서 반대 위치 모델로 평가
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 128,
        position_filter: str = None
    ):
        """
        Args:
            data_path: JSON 데이터 파일 경로
            tokenizer: Hugging Face Tokenizer
            max_length: 최대 시퀀스 길이
            position_filter: 특정 발화위치만 필터링
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.position_filter = position_filter
        
        # Special tokens
        self.special_tokens = {
            '선행문': '[PREV]',
            '후행문': '[NEXT]',
            'positive': '[POS]',
            'negative': '[NEG]',
            'neutral': '[NEU]',
            'has_negation': '[NEGT]',
            'no_negation': '[NONEGT]'
        }
        
        # 데이터 로드
        self.samples = self._load_and_process_data(data_path)
    
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """테스트 데이터 로드 및 처리 (순서 반대로)"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        for item in data:
            input_data = item['input']
            target_utterance = input_data['대상발화']
            position = input_data['발화위치']
            
            # 발화위치 필터링
            if self.position_filter and position != self.position_filter:
                continue
            
            # 각 선택지에 대해 샘플 생성
            for choice_num in range(1, 4):
                choice_key = f'발화{choice_num}'
                choice_text = input_data[choice_key]
                
                # 부정 표현 감지
                target_has_neg = NegationDetector.detect(target_utterance)
                choice_has_neg = NegationDetector.detect(choice_text)
                
                # 감정 분석
                target_emotion = EmotionAnalyzer.analyze(target_utterance)
                choice_emotion = EmotionAnalyzer.analyze(choice_text)
                
                # 원본 position을 저장하되, 실제 평가 시에는 반대 position 사용
                samples.append({
                    'id': item['id'],
                    'target_utterance': target_utterance,
                    'choice_text': choice_text,
                    'choice_num': choice_num,
                    'original_position': position,  # 원본 위치
                    'reversed_position': '후행문' if position == '선행문' else '선행문',  # 반대 위치
                    'target_has_negation': target_has_neg,
                    'choice_has_negation': choice_has_neg,
                    'target_emotion': target_emotion,
                    'choice_emotion': choice_emotion
                })
        
        print(f"Loaded {len(samples)} reversed test samples from {data_path}")
        if self.position_filter:
            print(f"Filtered by original position: {self.position_filter}")
        
        return samples
    
    def _create_input_text(self, sample: Dict) -> str:
        """입력 텍스트 생성 (순서 반대로)
        
        선행문인 경우: 선택지 → 대상발화 순서로 배치 (후행문 모델로 평가)
        후행문인 경우: 선택지 → 대상발화 순서로 배치 (선행문 모델로 평가)
        """
        # 반대 위치의 토큰 사용
        position_token = self.special_tokens[sample['reversed_position']]
        
        # 순서를 바꿔서 배치: 원래 choice → target 였던 것을 target → choice로
        # 감정/부정도 순서에 맞춰 교환
        choice_emotion_token = self.special_tokens[sample['choice_emotion']]
        target_emotion_token = self.special_tokens[sample['target_emotion']]
        
        choice_neg_token = (self.special_tokens['has_negation'] 
                           if sample['choice_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        target_neg_token = (self.special_tokens['has_negation'] 
                           if sample['target_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        # 순서 반대: choice가 먼저, target이 나중
        input_text = (
            f"{position_token} {choice_emotion_token} {choice_neg_token} "
            f"{sample['choice_text']} "
            f"{self.tokenizer.sep_token} "
            f"{target_emotion_token} {target_neg_token} "
            f"{sample['target_utterance']}"
        )
        
        return input_text
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 입력 텍스트 생성 (순서 반대)
        input_text = self._create_input_text(sample)
        
        # Tokenization
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'id': sample['id'],
            'choice_num': sample['choice_num'],
            'position': sample['original_position']  # 원본 위치 반환
        }


def ensemble_predict_all_models(
    prev_models: list,
    next_models: list,
    test_path: str,
    tokenizer,
    device: str,
    max_length: int,
    batch_size: int
):
    """모든 모델(선행문 5개 + 후행문 5개)로 앙상블 예측
    
    각 테스트 샘플에 대해:
    1. 선행문 모델 5개로 정방향 평가 (원래 순서)
    2. 후행문 모델 5개로 역방향 평가 (순서 반대)
    또는
    1. 후행문 모델 5개로 정방향 평가 (원래 순서)
    2. 선행문 모델 5개로 역방향 평가 (순서 반대)
    
    총 10개 모델의 평균 확률로 최종 예측
    
    Args:
        prev_models: 선행문 모델 리스트 (5개)
        next_models: 후행문 모델 리스트 (5개)
        test_path: 테스트 데이터 경로
        tokenizer: Tokenizer
        device: 디바이스
        max_length: 최대 시퀀스 길이
        batch_size: 배치 크기
    
    Returns:
        predictions: {id: [(choice_num, avg_score), ...]}
    """
    print(f"\n{'='*60}")
    print(f"10개 모델 앙상블 예측 (선행문 5개 + 후행문 5개)")
    print(f"각 샘플: 정방향 5개 + 역방향 5개 = 총 10개 예측")
    print(f"{'='*60}")
    
    # 각 샘플-선택지별 확률을 저장
    # {sample_id: {choice_num: [prob1, prob2, ..., prob10]}}
    all_predictions = defaultdict(lambda: defaultdict(list))
    
    # 원본 테스트 데이터 로드 (위치 정보 확인용)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 위치별로 샘플 분리
    prev_samples = [s for s in test_data if s['input']['발화위치'] == '선행문']
    next_samples = [s for s in test_data if s['input']['발화위치'] == '후행문']
    
    print(f"\n데이터 분포: 선행문 {len(prev_samples)}개, 후행문 {len(next_samples)}개")
    
    # ========== 선행문 샘플 처리 ==========
    if prev_samples and prev_models:
        print(f"\n{'='*60}")
        print(f"선행문 샘플 처리 ({len(prev_samples)}개)")
        print(f"{'='*60}")
        
        # 1) 선행문 모델 5개로 정방향 평가
        print("\n[1/2] 선행문 모델로 정방향 평가...")
        prev_dataset = TestDialogueDataset(
            test_path, tokenizer, position_filter='선행문', max_length=max_length
        )
        prev_loader = DataLoader(prev_dataset, batch_size=batch_size, shuffle=False)
        
        for model_idx, model in enumerate(prev_models):
            print(f"  선행문 모델 {model_idx + 1}/{len(prev_models)} 평가 중...")
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(prev_loader, desc=f'Prev-Forward M{model_idx+1}', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model.predict_proba(input_ids, attention_mask)
                    scores = outputs[:, 1].cpu().numpy()
                    
                    for i in range(len(scores)):
                        sample_id = batch['id'][i]
                        choice_num = batch['choice_num'][i].item()
                        all_predictions[sample_id][choice_num].append(scores[i])
        
        # 2) 후행문 모델 5개로 역방향 평가 (순서 반대)
        print("\n[2/2] 후행문 모델로 역방향 평가 (순서 반대)...")
        prev_reversed_dataset = ReversedTestDialogueDataset(
            test_path, tokenizer, position_filter='선행문', max_length=max_length
        )
        prev_reversed_loader = DataLoader(prev_reversed_dataset, batch_size=batch_size, shuffle=False)
        
        for model_idx, model in enumerate(next_models):
            print(f"  후행문 모델 {model_idx + 1}/{len(next_models)} 평가 중...")
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(prev_reversed_loader, desc=f'Next-Reversed M{model_idx+1}', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model.predict_proba(input_ids, attention_mask)
                    scores = outputs[:, 1].cpu().numpy()
                    
                    for i in range(len(scores)):
                        sample_id = batch['id'][i]
                        choice_num = batch['choice_num'][i].item()
                        all_predictions[sample_id][choice_num].append(scores[i])
    
    # ========== 후행문 샘플 처리 ==========
    if next_samples and next_models:
        print(f"\n{'='*60}")
        print(f"후행문 샘플 처리 ({len(next_samples)}개)")
        print(f"{'='*60}")
        
        # 1) 후행문 모델 5개로 정방향 평가
        print("\n[1/2] 후행문 모델로 정방향 평가...")
        next_dataset = TestDialogueDataset(
            test_path, tokenizer, position_filter='후행문', max_length=max_length
        )
        next_loader = DataLoader(next_dataset, batch_size=batch_size, shuffle=False)
        
        for model_idx, model in enumerate(next_models):
            print(f"  후행문 모델 {model_idx + 1}/{len(next_models)} 평가 중...")
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(next_loader, desc=f'Next-Forward M{model_idx+1}', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model.predict_proba(input_ids, attention_mask)
                    scores = outputs[:, 1].cpu().numpy()
                    
                    for i in range(len(scores)):
                        sample_id = batch['id'][i]
                        choice_num = batch['choice_num'][i].item()
                        all_predictions[sample_id][choice_num].append(scores[i])
        
        # 2) 선행문 모델 5개로 역방향 평가 (순서 반대)
        print("\n[2/2] 선행문 모델로 역방향 평가 (순서 반대)...")
        next_reversed_dataset = ReversedTestDialogueDataset(
            test_path, tokenizer, position_filter='후행문', max_length=max_length
        )
        next_reversed_loader = DataLoader(next_reversed_dataset, batch_size=batch_size, shuffle=False)
        
        for model_idx, model in enumerate(prev_models):
            print(f"  선행문 모델 {model_idx + 1}/{len(prev_models)} 평가 중...")
            model.eval()
            
            with torch.no_grad():
                for batch in tqdm(next_reversed_loader, desc=f'Prev-Reversed M{model_idx+1}', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model.predict_proba(input_ids, attention_mask)
                    scores = outputs[:, 1].cpu().numpy()
                    
                    for i in range(len(scores)):
                        sample_id = batch['id'][i]
                        choice_num = batch['choice_num'][i].item()
                        all_predictions[sample_id][choice_num].append(scores[i])
    
    # 앙상블: 각 선택지의 평균 확률 계산 (10개 모델)
    print(f"\n{'='*60}")
    print("앙상블 결과 계산 중...")
    print(f"{'='*60}")
    
    ensemble_predictions = defaultdict(list)
    
    for sample_id, choices in all_predictions.items():
        for choice_num, scores in choices.items():
            if len(scores) != 10:
                print(f"⚠️  경고: {sample_id} - 발화{choice_num}의 예측 수가 {len(scores)}개입니다 (예상: 10개)")
            avg_score = np.mean(scores)  # Soft Voting (평균)
            ensemble_predictions[sample_id].append((choice_num, avg_score))
    
    print(f"✓ 10개 모델 앙상블 완료: {len(ensemble_predictions)}개 샘플")
    
    return ensemble_predictions


def select_best_choices(predictions):
    """각 샘플에 대해 가장 높은 점수의 선택지 선택
    
    Args:
        predictions: {id: [(choice_num, score), ...]}
    
    Returns:
        results: {id: '발화1/2/3'}
    """
    results = {}
    
    for sample_id, choices in predictions.items():
        # 점수 기준 정렬
        choices_sorted = sorted(choices, key=lambda x: x[1], reverse=True)
        best_choice_num = choices_sorted[0][0]
        
        results[sample_id] = f'발화{best_choice_num}'
    
    return results


def merge_predictions(prev_predictions, next_predictions):
    """선행문/후행문 예측 결과 병합 (더 이상 사용 안 함 - 호환성 유지용)
    
    Args:
        prev_predictions: {id: '발화X'}
        next_predictions: {id: '발화X'}
    
    Returns:
        merged: {id: '발화X'}
    """
    merged = {}
    
    # 모든 ID 수집
    all_ids = set(prev_predictions.keys()) | set(next_predictions.keys())
    
    for sample_id in all_ids:
        prev_choice = prev_predictions.get(sample_id, None)
        next_choice = next_predictions.get(sample_id, None)
        
        if prev_choice is not None:
            merged[sample_id] = prev_choice
        elif next_choice is not None:
            merged[sample_id] = next_choice
    
    return merged


def create_submission(predictions: dict, test_data: list, output_path: str):
    """제출 파일 생성
    
    Args:
        predictions: {id: '발화X'}
        test_data: 원본 테스트 데이터
        output_path: 출력 파일 경로
    """
    results = []
    
    for sample in test_data:
        sample_id = sample['id']
        predicted_output = predictions.get(sample_id, '발화1')  # 기본값
        
        results.append({
            'id': sample_id,
            'input': sample['input'],
            'output': predicted_output
        })
    
    # 디렉토리 생성 (dirname이 비어있지 않을 때만)
    output_dir = os.path.dirname(output_path)
    if output_dir:  # 빈 문자열이 아닐 때만
        os.makedirs(output_dir, exist_ok=True)
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 제출 파일 저장: {output_path}")
    print(f"  - 총 {len(results)}개 샘플")


def main():
    parser = argparse.ArgumentParser(description='Cross Validation 모델 10개 앙상블 추론 (정방향 + 역방향)')
    parser.add_argument('--job_id', type=str, required=True, help='SLURM Job ID (예: 58583)')
    parser.add_argument('--output', type=str, default=None, help='출력 파일 경로 (기본: submission/ensemble_cv_10models_{job_id}.json)')
    parser.add_argument('--test_path', type=str, default=None, help='테스트 데이터 경로 (기본: config.TEST_PATH)')
    parser.add_argument('--model_dir', type=str, default='./models', help='모델 디렉토리 (기본: ./models)')
    
    args = parser.parse_args()
    
    # Config 로드
    config = Config()
    
    # 출력 경로 설정 및 정규화
    if args.output is None:
        args.output = f'submission/ensemble_cv_10models_{args.job_id}.json'
    
    # 출력 경로를 절대 경로로 변환 (상대 경로인 경우)
    args.output = os.path.abspath(args.output)
    
    # 테스트 데이터 경로 설정
    test_path = args.test_path if args.test_path else config.TEST_PATH
    
    print("="*60)
    print("Cross Validation 10-Model Ensemble Inference")
    print("선행문 5개 + 후행문 5개 = 총 10개 모델")
    print("각 샘플: 정방향 5개 + 역방향 5개 평가")
    print("="*60)
    print(f"\nJob ID: {args.job_id}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Test Data: {test_path}")
    print(f"Output: {args.output}")
    print()
    
    # 디바이스 설정
    device = config.get_device()
    print(f"Device: {device}\n")
    
    # CV 모델 파일 찾기
    print("="*60)
    print("모델 파일 검색")
    print("="*60)
    cv_models = find_cv_models(args.job_id, args.model_dir)
    
    if not cv_models['선행문'] or not cv_models['후행문']:
        print("\n❌ 에러: 선행문 또는 후행문 모델을 찾을 수 없습니다!")
        print(f"   선행문: {len(cv_models['선행문'])}개")
        print(f"   후행문: {len(cv_models['후행문'])}개")
        print(f"\n10개 모델 앙상블을 위해서는 선행문 5개, 후행문 5개가 모두 필요합니다.")
        sys.exit(1)
    
    # Tokenizer 로드
    print("\n" + "="*60)
    print("Tokenizer 로드")
    print("="*60)
    
    if os.path.exists(config.MODEL_NAME) and os.path.isdir(config.MODEL_NAME):
        print(f"로컬에서 로드: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    else:
        print(f"HuggingFace에서 다운로드: {config.MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Special tokens 추가
    tokenizer.add_special_tokens(config.SPECIAL_TOKENS)
    print(f"✓ Tokenizer 로드 완료 (vocab size: {len(tokenizer)})")
    
    # 테스트 데이터 로드
    print("\n" + "="*60)
    print("테스트 데이터 로드")
    print("="*60)
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✓ {len(test_data)}개 샘플 로드")
    
    # 선행문 모델 로드
    print("\n" + "="*60)
    print("선행문 모델 로드")
    print("="*60)
    
    prev_models = []
    for model_path in cv_models['선행문']:
        print(f"\n로딩: {os.path.basename(model_path)}")
        model, checkpoint = load_single_model(model_path, config.MODEL_NAME, tokenizer, device)
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
        print(f"  - Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
        prev_models.append(model)
    
    # 후행문 모델 로드
    print("\n" + "="*60)
    print("후행문 모델 로드")
    print("="*60)
    
    next_models = []
    for model_path in cv_models['후행문']:
        print(f"\n로딩: {os.path.basename(model_path)}")
        model, checkpoint = load_single_model(model_path, config.MODEL_NAME, tokenizer, device)
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
        print(f"  - Val F1: {checkpoint.get('val_f1', 'N/A'):.4f}")
        next_models.append(model)
    
    # 10개 모델 앙상블 예측
    print("\n" + "="*60)
    print("10개 모델 앙상블 예측 시작")
    print("="*60)
    
    ensemble_predictions = ensemble_predict_all_models(
        prev_models=prev_models,
        next_models=next_models,
        test_path=test_path,
        tokenizer=tokenizer,
        device=device,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE
    )
    
    final_predictions = select_best_choices(ensemble_predictions)
    
    # 제출 파일 생성
    print("\n" + "="*60)
    print("제출 파일 생성")
    print("="*60)
    
    create_submission(final_predictions, test_data, args.output)
    
    # 통계 출력
    print("\n" + "="*60)
    print("예측 통계")
    print("="*60)
    
    choice_counts = defaultdict(int)
    for choice in final_predictions.values():
        choice_counts[choice] += 1
    
    for choice in sorted(choice_counts.keys()):
        count = choice_counts[choice]
        percentage = (count / len(final_predictions)) * 100
        print(f"{choice}: {count}개 ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ 10개 모델 앙상블 추론 완료!")
    print("="*60)


if __name__ == '__main__':
    main()
    