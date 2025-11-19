#!/bin/bash
#SBATCH -J diag_train
#SBATCH --gres=gpu:1
#SBATCH -p batch_ugrad                
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 1-0
#SBATCH -o logs/%A.out
#SBATCH -e logs/%A.err

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
for file in nikluge-2025-일상대화연결.train.json nikluge-2025-일상대화연결.dev.json nikluge-2025-일상대화연결.test.json; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (NOT FOUND)"
        exit 1
    fi
done
echo ""

# Python 스크립트 파일 확인
echo "=========================================="
echo "Checking script files..."
echo "=========================================="
for file in data_loader.py model.py train.py inference.py; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (NOT FOUND)"
        exit 1
    fi
done
echo ""

# 학습 시작
echo "=========================================="
echo "Starting Training..."
echo "=========================================="
echo ""

# JOB_ID 환경변수 설정하여 train.py에 전달
export JOB_ID=${SLURM_JOB_ID}
python train.py 2>&1 | tee logs/train_${SLURM_JOB_ID}_output.log

# 학습 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo ""
    
    # 생성된 모델 파일 확인
    echo "Generated model files:"
    ls -lh models/best_model_*_${SLURM_JOB_ID}.pt 2>/dev/null || echo "No model files found"
    echo ""
    
    # 추론 실행
    echo "=========================================="
    echo "Starting Inference..."
    echo "=========================================="
    echo ""
    
    # JOB_ID 환경변수 설정하여 inference.py에 전달
    export JOB_ID=${SLURM_JOB_ID}
    python inference.py 2>&1 | tee logs/inference_${SLURM_JOB_ID}_output.log
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "Inference completed successfully!"
        echo "=========================================="
        echo ""
        
        # 제출 파일 확인
        if [ -f "submission.json" ]; then
            echo "✓ submission.json created"
            echo "File size: $(du -h submission.json | cut -f1)"
            echo "Sample count: $(python -c "import json; data=json.load(open('submission.json')); print(len(data))")"
        else
            echo "✗ submission.json not found"
        fi
        echo ""
        
        # 검증 데이터로 평가 (선택)
        echo "=========================================="
        echo "Evaluating on dev set..."
        echo "=========================================="
        echo ""
        
        # JOB_ID 환경변수 설정하여 inference.py에 전달
        export JOB_ID=${SLURM_JOB_ID}
        python inference.py --eval-dev 2>&1 | tee logs/eval_${SLURM_JOB_ID}_output.log
        
    else
        echo ""
        echo "=========================================="
        echo "Inference failed!"
        echo "=========================================="
        exit 1
    fi
    
else
    echo ""
    echo "=========================================="
    echo "Training failed!"
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
echo "  Models:"
ls -lh models/best_model_*_${SLURM_JOB_ID}.pt 2>/dev/null | awk '{print "    - "$9" ("$5")"}'
echo "  Histories:"
ls -lh models/history_*_${SLURM_JOB_ID}.json 2>/dev/null | awk '{print "    - "$9" ("$5")"}'
echo "  Submission:"
[ -f "submission.json" ] && echo "    - submission.json ($(du -h submission.json | cut -f1))"
echo ""
echo "Logs saved in logs/ directory:"
ls -lh logs/*_${SLURM_JOB_ID}_*.log 2>/dev/null | awk '{print "  - "$9" ("$5")"}'
echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
