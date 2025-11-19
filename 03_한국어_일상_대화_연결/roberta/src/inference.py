"""
추론 및 제출 파일 생성 스크립트
- 학습된 선행문/후행문 모델 로드
- 테스트 데이터에 대한 예측 수행
- 제출 파일 생성
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from collections import defaultdict

from data_loader import TestDialogueDataset
from model import DialogueConnectionModel
from config import Config


def load_model(model_path: str, model_name: str, tokenizer, device: str):
    """모델 로드"""
    print(f"Loading model from: {model_path}")
    
    config = Config()
    model = DialogueConnectionModel(model_name=model_name, dropout_rate=config.DROPOUT_RATE)
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Val Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
    
    return model


def predict_single_position(
    model,
    dataloader,
    device: str
):
    """단일 위치(선행문 또는 후행문)에 대한 예측
    
    Returns:
        predictions: {id: [(choice_num, score), ...]}
    """
    predictions = defaultdict(list)
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 예측
            outputs = model.predict_proba(input_ids, attention_mask)
            
            # 정답일 확률 (클래스 1)
            scores = outputs[:, 1].cpu().numpy()
            
            # 결과 저장
            for i in range(len(scores)):
                sample_id = batch['id'][i]
                choice_num = batch['choice_num'][i].item()
                score = scores[i]
                
                predictions[sample_id].append((choice_num, score))
    
    return predictions


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
    """선행문과 후행문 예측 결과 병합
    
    Args:
        prev_predictions: 선행문 예측 결과
        next_predictions: 후행문 예측 결과
    
    Returns:
        merged: 전체 예측 결과
    """
    merged = {}
    merged.update(prev_predictions)
    merged.update(next_predictions)
    
    return merged


def create_submission_file(
    predictions: dict,
    test_data_path: str,
    output_path: str = 'submission.json'
):
    """제출 파일 생성
    
    Args:
        predictions: {id: '발화1/2/3'}
        test_data_path: 원본 테스트 데이터 경로
        output_path: 출력 파일 경로
    """
    # 원본 테스트 데이터 로드
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 예측 결과 추가
    for item in test_data:
        sample_id = item['id']
        if sample_id in predictions:
            item['output'] = predictions[sample_id]
        else:
            # 예측이 없는 경우 기본값
            item['output'] = '발화1'
            print(f"Warning: No prediction for {sample_id}, using default '발화1'")
    
    # 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Total predictions: {len(test_data)}")


def main():
    """메인 함수"""
    
    # config.py에서 설정 가져오기
    config = Config()
    
    # JOB_ID 서픽스 생성
    job_id_suffix = f"_{config.JOB_ID}" if config.JOB_ID else ""
    
    print("="*60)
    print("대화 연결 모델 추론")
    print("="*60)
    print("\n설정:")
    print(f"  test_path: {config.TEST_PATH}")
    print(f"  model_name: {config.MODEL_NAME}")
    print(f"  max_length: {config.MAX_LENGTH}")
    print(f"  batch_size: {config.BATCH_SIZE}")
    print(f"  prev_model_path: {config.OUTPUT_DIR}/best_model_선행문{job_id_suffix}.pt")
    print(f"  next_model_path: {config.OUTPUT_DIR}/best_model_후행문{job_id_suffix}.pt")
    print(f"  output_path: {config.SUBMISSION_PATH}")
    print()
    
    # 디바이스 설정
    device = config.get_device()
    print(f"Device: {device}\n")
    
    # Tokenizer 로드
    print("Loading tokenizer...")
    model_name = config.MODEL_NAME
    # 로컬 경로인지 확인
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"  Loading from local path: {os.path.abspath(model_name)}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    else:
        print(f"  Downloading from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Special tokens 추가 (학습 시와 동일하게)
    tokenizer.add_special_tokens(config.SPECIAL_TOKENS)
    
    # 1. 선행문 예측
    print("\n" + "="*60)
    print("1. 선행문 예측")
    print("="*60 + "\n")
    
    # 선행문 모델 로드
    prev_model = load_model(
        f"{config.OUTPUT_DIR}/best_model_선행문{job_id_suffix}.pt",
        config.MODEL_NAME,
        tokenizer,
        device
    )
    
    # 선행문 데이터 로드
    print("\nLoading test data (선행문)...")
    prev_dataset = TestDialogueDataset(
        config.TEST_PATH,
        tokenizer,
        max_length=config.MAX_LENGTH,
        position_filter='선행문'
    )
    
    prev_dataloader = DataLoader(
        prev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # 예측
    print("\nPredicting...")
    prev_raw_predictions = predict_single_position(prev_model, prev_dataloader, device)
    prev_predictions = select_best_choices(prev_raw_predictions)
    
    print(f"✓ 선행문 예측 완료: {len(prev_predictions)} samples")
    
    # 메모리 정리
    del prev_model
    torch.cuda.empty_cache()
    
    # 2. 후행문 예측
    print("\n" + "="*60)
    print("2. 후행문 예측")
    print("="*60 + "\n")
    
    # 후행문 모델 로드
    next_model = load_model(
        f"{config.OUTPUT_DIR}/best_model_후행문{job_id_suffix}.pt",
        config.MODEL_NAME,
        tokenizer,
        device
    )
    
    # 후행문 데이터 로드
    print("\nLoading test data (후행문)...")
    next_dataset = TestDialogueDataset(
        config.TEST_PATH,
        tokenizer,
        max_length=config.MAX_LENGTH,
        position_filter='후행문'
    )
    
    next_dataloader = DataLoader(
        next_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    # 예측
    print("\nPredicting...")
    next_raw_predictions = predict_single_position(next_model, next_dataloader, device)
    next_predictions = select_best_choices(next_raw_predictions)
    
    print(f"✓ 후행문 예측 완료: {len(next_predictions)} samples")
    
    # 메모리 정리
    del next_model
    torch.cuda.empty_cache()
    
    # 3. 결과 병합 및 제출 파일 생성
    print("\n" + "="*60)
    print("3. 제출 파일 생성")
    print("="*60 + "\n")
    
    all_predictions = merge_predictions(prev_predictions, next_predictions)
    
    print(f"Total predictions: {len(all_predictions)}")
    print(f"  - 선행문: {len(prev_predictions)}")
    print(f"  - 후행문: {len(next_predictions)}")
    
    # 제출 파일 생성
    create_submission_file(
        all_predictions,
        config.TEST_PATH,
        config.SUBMISSION_PATH
    )
    
    # 예측 분포 출력
    print("\n예측 분포:")
    from collections import Counter
    prediction_counts = Counter(all_predictions.values())
    for label, count in sorted(prediction_counts.items()):
        percentage = count / len(all_predictions) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("✓ 추론 완료!")
    print("="*60)


def evaluate_on_dev():
    """검증 데이터에 대한 평가 (선택사항)"""
    
    # config.py에서 설정 가져오기
    config = Config()
    
    # JOB_ID 서픽스 생성
    job_id_suffix = f"_{config.JOB_ID}" if config.JOB_ID else ""
    
    print("="*60)
    print("검증 데이터 평가")
    print("="*60 + "\n")
    
    device = config.get_device()
    
    # Tokenizer 로드
    model_name = config.MODEL_NAME
    # 로컬 경로인지 확인
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"  Loading from local path: {os.path.abspath(model_name)}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    else:
        print(f"  Downloading from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.add_special_tokens(config.SPECIAL_TOKENS)
    
    # 검증 데이터 로드
    with open(config.DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    # 선행문 예측
    print("선행문 예측 중...")
    prev_model = load_model(f"{config.OUTPUT_DIR}/best_model_선행문{job_id_suffix}.pt", config.MODEL_NAME, tokenizer, device)
    prev_dataset = TestDialogueDataset(
        config.DEV_PATH, tokenizer, 
        max_length=config.MAX_LENGTH, 
        position_filter='선행문'
    )
    prev_dataloader = DataLoader(prev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    prev_raw_predictions = predict_single_position(prev_model, prev_dataloader, device)
    prev_predictions = select_best_choices(prev_raw_predictions)
    del prev_model
    torch.cuda.empty_cache()
    
    # 후행문 예측
    print("후행문 예측 중...")
    next_model = load_model(f"{config.OUTPUT_DIR}/best_model_후행문{job_id_suffix}.pt", config.MODEL_NAME, tokenizer, device)
    next_dataset = TestDialogueDataset(
        config.DEV_PATH, tokenizer, 
        max_length=config.MAX_LENGTH, 
        position_filter='후행문'
    )
    next_dataloader = DataLoader(next_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    next_raw_predictions = predict_single_position(next_model, next_dataloader, device)
    next_predictions = select_best_choices(next_raw_predictions)
    del next_model
    torch.cuda.empty_cache()
    
    # 병합
    all_predictions = merge_predictions(prev_predictions, next_predictions)
    
    # 정답과 비교
    correct = 0
    total = 0
    
    for item in dev_data:
        sample_id = item['id']
        true_label = item['output']
        pred_label = all_predictions.get(sample_id, '발화1')
        
        if true_label == pred_label:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    
    print(f"\n{'='*60}")
    print(f"검증 데이터 정확도: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--eval-dev':
        # 검증 데이터 평가
        evaluate_on_dev()
    else:
        # 테스트 데이터 추론
        main()
