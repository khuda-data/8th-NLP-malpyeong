#!/bin/bash
#SBATCH -J diag_cv
#SBATCH --gres=gpu:1
#SBATCH -p batch_ugrad                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 1-0
#SBATCH -o logs/cv_%A.out
#SBATCH -e logs/cv_%A.err

# 작업 정보 출력
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# 로그 디렉토리 생성
mkdir -p logs
mkdir -p models
echo "Models will be saved with Job ID: ${SLURM_JOB_ID}"
echo ""

# Conda 환경 활성화 (base 환경 사용)
source /data/mcladinz/anaconda3/etc/profile.d/conda.sh
conda activate base

# GPU 확인
echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo ""
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo ""

# 데이터 파일 확인
echo "=========================================="
echo "Checking data files..."
echo "=========================================="
for file in dataset/nikluge-2025-일상대화연결.train.json dataset/nikluge-2025-일상대화연결.dev.json dataset/nikluge-2025-일상대화연결.test.json; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (NOT FOUND)"
        exit 1
    fi
done

# connect_sentence.json 파일은 선택적 확인
if [ -f "dataset/connect_sentence.json" ]; then
    echo "✓ dataset/connect_sentence.json (optional)"
else
    echo "ℹ dataset/connect_sentence.json (not configured - will use TRAIN+DEV only)"
fi
echo ""

# Python 스크립트 파일 확인
echo "=========================================="
echo "Checking script files..."
echo "=========================================="
for file in src/data_loader.py src/model.py src/train.py src/train_cross_validation.py src/config.py; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (NOT FOUND)"
        exit 1
    fi
done
echo ""

# config.py에서 USE_CROSS_VALIDATION 확인
echo "=========================================="
echo "Checking configuration..."
echo "=========================================="
cd src
python -c "from config import Config; c = Config(); print(f'USE_CROSS_VALIDATION: {c.USE_CROSS_VALIDATION}'); print(f'N_FOLDS: {c.N_FOLDS}'); print(f'CV_SEED: {c.CV_SEED}')"
cd ..
echo ""

# 5-Fold Cross Validation 학습 시작
echo "=========================================="
echo "Starting 5-Fold Cross Validation..."
echo "=========================================="
echo ""

# JOB_ID 환경변수 설정하여 train_cross_validation.py에 전달
export JOB_ID=${SLURM_JOB_ID}

cd src
python train_cross_validation.py 2>&1 | tee ../logs/cv_${SLURM_JOB_ID}_output.log
TRAIN_EXIT_CODE=$?
cd ..

# 학습 결과 확인
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Cross Validation completed successfully!"
    echo "=========================================="
    echo ""
    
    # 생성된 모델 파일 확인
    echo "Generated model files:"
    ls -lh models/best_model_*fold*_${SLURM_JOB_ID}.pt 2>/dev/null || echo "No model files found"
    echo ""
    
    # CV 결과 파일 확인
    if [ -f "cross_validation_results_${SLURM_JOB_ID}.json" ]; then
        echo "✓ Cross Validation results saved"
        echo ""
        echo "Results summary:"
        python -c "
import json
with open('cross_validation_results_${SLURM_JOB_ID}.json', 'r', encoding='utf-8') as f:
    results = json.load(f)
    for position in ['선행문', '후행문']:
        if position in results and results[position]:
            accs = [r['val_accuracy'] for r in results[position]]
            avg_acc = sum(accs) / len(accs)
            print(f'{position} - Avg Accuracy: {avg_acc:.4f}')
" 2>/dev/null || echo "Could not parse results"
        echo ""
    else
        echo "⚠ Cross Validation results file not found"
    fi
    
    # 앙상블 추론 실행
    echo "=========================================="
    echo "Starting 10-Model Ensemble Inference..."
    echo "=========================================="
    echo ""
    
    # ensemble_inference_cv.py 스크립트 확인
    if [ -f "src/ensemble_inference_cv.py" ]; then
        echo "Running 10-model ensemble inference (Forward + Reversed)"
        echo "  - 선행문 5-fold + 후행문 5-fold = 10 models"
        echo "  - Each sample: Forward 5 + Reversed 5 = 10 predictions"
        echo "Job ID: ${SLURM_JOB_ID}"
        echo "Output: submission/ensemble_cv_10models_${SLURM_JOB_ID}.json"
        echo ""
        
        # 앙상블 추론 실행
        cd src
        python ensemble_inference_cv.py \
            --job_id ${SLURM_JOB_ID} \
            --output ../submission/ensemble_cv_10models_${SLURM_JOB_ID}.json \
            --test_path ../dataset/nikluge-2025-일상대화연결.test.json \
            --model_dir ../models
        
        INFERENCE_EXIT_CODE=$?
        cd ..
        
        if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
            echo ""
            echo "✓ 10-model ensemble inference completed successfully!"
            
            # 제출 파일 확인
            if [ -f "submission/ensemble_cv_10models_${SLURM_JOB_ID}.json" ]; then
                FILE_SIZE=$(du -h "submission/ensemble_cv_10models_${SLURM_JOB_ID}.json" | cut -f1)
                NUM_SAMPLES=$(python -c "import json; data = json.load(open('submission/ensemble_cv_10models_${SLURM_JOB_ID}.json', 'r', encoding='utf-8')); print(len(data))" 2>/dev/null || echo "unknown")
                echo "  - File: submission/ensemble_cv_10models_${SLURM_JOB_ID}.json"
                echo "  - Size: ${FILE_SIZE}"
                echo "  - Samples: ${NUM_SAMPLES}"
                echo ""
                
                # 예측 분포 출력
                echo "Prediction distribution:"
                python -c "
import json
from collections import Counter
with open('submission/ensemble_cv_10models_${SLURM_JOB_ID}.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    outputs = [item['output'] for item in data]
    counter = Counter(outputs)
    total = len(outputs)
    for choice, count in sorted(counter.items()):
        pct = (count / total) * 100
        print(f'  {choice}: {count} ({pct:.1f}%)')
" 2>/dev/null || echo "  Could not analyze predictions"
                echo ""
            else
                echo "  ⚠ Warning: Submission file not found"
            fi
        else
            echo ""
            echo "⚠ Ensemble inference failed (Exit Code: $INFERENCE_EXIT_CODE)"
            echo "  Check logs for details"
        fi
    else
        echo "⚠ src/ensemble_inference_cv.py not found"
        echo "  Skipping ensemble inference"
    fi
    echo ""
    
else
    echo ""
    echo "=========================================="
    echo "Cross Validation failed!"
    echo "=========================================="
    exit 1
fi

# 최종 결과 요약
echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End Time: $(date)"
echo ""
echo "Generated files:"
echo ""
echo "  📁 Models (per fold):"
ls -lh models/best_model_*fold*_${SLURM_JOB_ID}.pt 2>/dev/null | awk '{print "    - "$9" ("$5")"}' | head -n 10
echo ""
echo "  📊 Histories (per fold):"
ls -lh models/history_*fold*_${SLURM_JOB_ID}.json 2>/dev/null | awk '{print "    - "$9" ("$5")"}' | head -n 10
echo ""
echo "  📈 CV Results:"
[ -f "cross_validation_results_${SLURM_JOB_ID}.json" ] && echo "    - cross_validation_results_${SLURM_JOB_ID}.json ($(du -h cross_validation_results_${SLURM_JOB_ID}.json | cut -f1))"
echo ""
echo "  📝 Submission File:"
[ -f "submission/ensemble_cv_10models_${SLURM_JOB_ID}.json" ] && echo "    - submission/ensemble_cv_10models_${SLURM_JOB_ID}.json ($(du -h submission/ensemble_cv_10models_${SLURM_JOB_ID}.json | cut -f1))" || echo "    ⚠ Not generated"
echo ""
echo "  📋 Logs:"
ls -lh logs/cv_${SLURM_JOB_ID}_*.log 2>/dev/null | awk '{print "    - "$9" ("$5")"}'
echo ""
echo "=========================================="
echo "✓ All tasks completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check CV results: cat cross_validation_results_${SLURM_JOB_ID}.json"
echo "  2. Review submission: cat submission/ensemble_cv_10models_${SLURM_JOB_ID}.json"
echo "  3. Submit to competition: submission/ensemble_cv_10models_${SLURM_JOB_ID}.json"
echo ""
echo "📊 Ensemble Strategy:"
echo "  - Total 10 models used (5 선행문 + 5 후행문)"
echo "  - Each sample evaluated by:"
echo "    • 5 forward predictions (original position models)"
echo "    • 5 reversed predictions (opposite position models)"
echo "  - Final prediction: Soft voting (average probability)"
echo ""
