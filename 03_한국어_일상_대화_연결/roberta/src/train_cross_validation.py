"""
5-Fold Cross Validation 학습 스크립트
각 fold마다 모델을 학습하고 평균 성능을 계산
"""

import os
import sys
import json
from config import Config


def train_all_folds():
    """모든 fold에 대해 학습 수행"""
    
    config = Config()
    
    if not config.USE_CROSS_VALIDATION:
        print("❌ 에러: config.py에서 USE_CROSS_VALIDATION = True로 설정해주세요.")
        sys.exit(1)
    
    print("="*60)
    print(f"{config.N_FOLDS}-Fold Cross Validation 학습")
    print("="*60)
    print(f"\nFold 수: {config.N_FOLDS}")
    print(f"기본 데이터: {config.TRAIN_PATH} + {config.DEV_PATH}")
    
    # connect_sentence.json 존재 여부 확인 (config에 절대 경로로 설정됨)
    connect_path = config.CONNECT_SENTENCE_PATH
    if connect_path and connect_path.strip() and os.path.exists(connect_path):
        print(f"추가 데이터: {config.CONNECT_SENTENCE_PATH} ✓")
        print("  → 모든 데이터가 순서를 바꿔서 선행문/후행문 각각에 활용됨")
    else:
        if connect_path and connect_path.strip():
            print(f"추가 데이터: {config.CONNECT_SENTENCE_PATH} (파일 없음)")
        else:
            print(f"추가 데이터: 사용 안 함 (TRAIN_PATH + DEV_PATH만 사용)")
    print()
    
    results = {
        '선행문': [],
        '후행문': []
    }
    
    # 각 fold에 대해 학습
    for fold in range(config.N_FOLDS):
        print("\n" + "="*60)
        print(f"Fold {fold + 1}/{config.N_FOLDS} 시작")
        print("="*60 + "\n")
        
        # 환경변수로 현재 fold 설정
        os.environ['CURRENT_FOLD'] = str(fold)
        
        # train.py를 subprocess로 실행
        import subprocess
        result = subprocess.run(
            [sys.executable, 'train.py'],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ Fold {fold + 1} 학습 실패!")
            continue
        
        # 결과 수집
        job_id_suffix = f"_{config.JOB_ID}" if config.JOB_ID else ""
        
        for position in ['선행문', '후행문']:
            history_path = os.path.join(
                config.OUTPUT_DIR, 
                f'history_{position}_fold{fold}{job_id_suffix}.json'
            )
            
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    best_val_acc = max(history.get('val_accuracy', [0]))
                    best_val_f1 = max(history.get('val_f1', [0]))
                    
                    results[position].append({
                        'fold': fold,
                        'val_accuracy': best_val_acc,
                        'val_f1': best_val_f1
                    })
        
        print(f"\n✓ Fold {fold + 1} 완료!\n")
    
    # 전체 결과 요약
    print("\n" + "="*60)
    print("Cross Validation 결과 요약")
    print("="*60 + "\n")
    
    for position in ['선행문', '후행문']:
        print(f"\n{position}:")
        print("-" * 40)
        
        if not results[position]:
            print("  결과 없음")
            continue
        
        for fold_result in results[position]:
            fold_num = fold_result['fold'] + 1
            acc = fold_result['val_accuracy']
            f1 = fold_result['val_f1']
            print(f"  Fold {fold_num}: Acc={acc:.4f}, F1={f1:.4f}")
        
        # 평균 계산
        avg_acc = sum(r['val_accuracy'] for r in results[position]) / len(results[position])
        avg_f1 = sum(r['val_f1'] for r in results[position]) / len(results[position])
        
        print(f"\n  평균: Acc={avg_acc:.4f}, F1={avg_f1:.4f}")
        print(f"  표준편차: Acc={calculate_std([r['val_accuracy'] for r in results[position]]):.4f}")
    
    # 결과를 JSON으로 저장
    job_id_suffix = f"_{config.JOB_ID}" if config.JOB_ID else ""
    results_path = f'cross_validation_results{job_id_suffix}.json'
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 결과 저장: {results_path}")
    print("\n" + "="*60)


def calculate_std(values):
    """표준편차 계산"""
    import math
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


if __name__ == '__main__':
    # CURRENT_FOLD를 직접 설정하지 말고 이 스크립트가 관리
    train_all_folds()
