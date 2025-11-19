"""
유틸리티 함수 모음
- 학습 히스토리 시각화
- 통계 출력
- 결과 분석
"""

import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def plot_training_history(history_path: str, save_path: str = None):
    """학습 히스토리 시각화
    
    Args:
        history_path: history JSON 파일 경로
        save_path: 이미지 저장 경로 (None이면 화면에 표시)
    """
    # 히스토리 로드
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'학습 히스토리: {history_path}', fontsize=16, fontweight='bold')
    
    # 1. Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss 추이')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy
    axes[0, 1].plot(epochs, history['train_accuracy'], 'b-o', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_accuracy'], 'r-s', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy 추이')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Validation F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-^', label='Val F1', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score 추이')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 통계 요약
    axes[1, 1].axis('off')
    
    # 최고 성능 찾기
    best_epoch = history['val_accuracy'].index(max(history['val_accuracy'])) + 1
    best_val_acc = max(history['val_accuracy'])
    best_val_f1 = history['val_f1'][best_epoch - 1]
    final_val_acc = history['val_accuracy'][-1]
    
    stats_text = f"""
    📊 학습 통계
    
    총 Epoch: {len(epochs)}
    
    ✓ 최고 성능
      - Epoch: {best_epoch}
      - Val Accuracy: {best_val_acc:.4f}
      - Val F1: {best_val_f1:.4f}
    
    ✓ 최종 성능
      - Val Accuracy: {final_val_acc:.4f}
      - Val F1: {history['val_f1'][-1]:.4f}
    
    ✓ Train/Val Gap
      - Loss: {abs(history['train_loss'][-1] - history['val_loss'][-1]):.4f}
      - Accuracy: {abs(history['train_accuracy'][-1] - history['val_accuracy'][-1]):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(prev_history_path: str, next_history_path: str, save_path: str = None):
    """선행문/후행문 모델 비교
    
    Args:
        prev_history_path: 선행문 히스토리 경로
        next_history_path: 후행문 히스토리 경로
        save_path: 이미지 저장 경로
    """
    # 히스토리 로드
    with open(prev_history_path, 'r', encoding='utf-8') as f:
        prev_history = json.load(f)
    
    with open(next_history_path, 'r', encoding='utf-8') as f:
        next_history = json.load(f)
    
    prev_epochs = range(1, len(prev_history['val_accuracy']) + 1)
    next_epochs = range(1, len(next_history['val_accuracy']) + 1)
    
    # Figure 생성
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('선행문 vs 후행문 모델 비교', fontsize=16, fontweight='bold')
    
    # 1. Validation Accuracy 비교
    axes[0].plot(prev_epochs, prev_history['val_accuracy'], 'b-o', 
                label='선행문', linewidth=2)
    axes[0].plot(next_epochs, next_history['val_accuracy'], 'r-s', 
                label='후행문', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy 비교')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Validation F1 비교
    axes[1].plot(prev_epochs, prev_history['val_f1'], 'b-o', 
                label='선행문', linewidth=2)
    axes[1].plot(next_epochs, next_history['val_f1'], 'r-s', 
                label='후행문', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1 Score')
    axes[1].set_title('Validation F1 Score 비교')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 비교 그래프 저장: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # 통계 출력
    print("\n" + "="*60)
    print("모델 성능 비교")
    print("="*60)
    
    prev_best_acc = max(prev_history['val_accuracy'])
    next_best_acc = max(next_history['val_accuracy'])
    
    prev_best_f1 = max(prev_history['val_f1'])
    next_best_f1 = max(next_history['val_f1'])
    
    print(f"\n선행문 모델:")
    print(f"  - 최고 Val Accuracy: {prev_best_acc:.4f}")
    print(f"  - 최고 Val F1: {prev_best_f1:.4f}")
    
    print(f"\n후행문 모델:")
    print(f"  - 최고 Val Accuracy: {next_best_acc:.4f}")
    print(f"  - 최고 Val F1: {next_best_f1:.4f}")
    
    print(f"\n차이:")
    print(f"  - Accuracy 차이: {abs(prev_best_acc - next_best_acc):.4f}")
    print(f"  - F1 차이: {abs(prev_best_f1 - next_best_f1):.4f}")
    
    print("\n" + "="*60)


def analyze_submission(submission_path: str, dev_path: str = None):
    """제출 파일 분석
    
    Args:
        submission_path: 제출 파일 경로
        dev_path: 검증 데이터 경로 (정답 비교용, optional)
    """
    # 제출 파일 로드
    with open(submission_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    print("\n" + "="*60)
    print(f"제출 파일 분석: {submission_path}")
    print("="*60)
    
    # 기본 통계
    total = len(predictions)
    print(f"\n총 예측 샘플 수: {total}")
    
    # 레이블 분포
    from collections import Counter
    label_counts = Counter([item['output'] for item in predictions])
    
    print("\n예측 레이블 분포:")
    for label in ['발화1', '발화2', '발화3']:
        count = label_counts.get(label, 0)
        percentage = count / total * 100
        print(f"  {label}: {count:4d} ({percentage:5.1f}%)")
    
    # 발화위치별 분포
    position_labels = defaultdict(list)
    for item in predictions:
        position = item['input']['발화위치']
        position_labels[position].append(item['output'])
    
    print("\n발화위치별 예측 분포:")
    for position in ['선행문', '후행문']:
        if position in position_labels:
            pos_labels = position_labels[position]
            pos_total = len(pos_labels)
            pos_counts = Counter(pos_labels)
            
            print(f"\n  {position} ({pos_total}개):")
            for label in ['발화1', '발화2', '발화3']:
                count = pos_counts.get(label, 0)
                percentage = count / pos_total * 100 if pos_total > 0 else 0
                print(f"    {label}: {count:4d} ({percentage:5.1f}%)")
    
    # 검증 데이터와 비교 (optional)
    if dev_path:
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        
        # ID로 매핑
        dev_dict = {item['id']: item['output'] for item in dev_data}
        pred_dict = {item['id']: item['output'] for item in predictions}
        
        # 공통 ID 찾기
        common_ids = set(dev_dict.keys()) & set(pred_dict.keys())
        
        if common_ids:
            correct = sum(1 for id in common_ids if dev_dict[id] == pred_dict[id])
            accuracy = correct / len(common_ids) * 100
            
            print(f"\n✓ 검증 데이터 비교:")
            print(f"  - 공통 샘플: {len(common_ids)}")
            print(f"  - 정확도: {accuracy:.2f}% ({correct}/{len(common_ids)})")
    
    print("\n" + "="*60)


from collections import defaultdict


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python utils.py plot <history_path> [save_path]")
        print("  python utils.py compare <prev_history> <next_history> [save_path]")
        print("  python utils.py analyze <submission_path> [dev_path]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'plot':
        history_path = sys.argv[2]
        save_path = sys.argv[3] if len(sys.argv) > 3 else None
        plot_training_history(history_path, save_path)
    
    elif command == 'compare':
        prev_history = sys.argv[2]
        next_history = sys.argv[3]
        save_path = sys.argv[4] if len(sys.argv) > 4 else None
        compare_models(prev_history, next_history, save_path)
    
    elif command == 'analyze':
        submission_path = sys.argv[2]
        dev_path = sys.argv[3] if len(sys.argv) > 3 else None
        analyze_submission(submission_path, dev_path)
    
    else:
        print(f"알 수 없는 명령: {command}")
