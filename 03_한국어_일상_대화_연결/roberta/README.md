# RoBERTa 한국어 일상 대화 연결 모델

클로드가 적어준 내용입니다 !!!!!!

## 📁 폴더 구조

```
roberta/
├── dataset/                              # 데이터셋 폴더
│   ├── nikluge-2025-일상대화연결.train.json   # 학습 데이터
│   ├── nikluge-2025-일상대화연결.dev.json     # 검증 데이터
│   ├── nikluge-2025-일상대화연결.test.json    # 테스트 데이터
│   ├── connect_sentence.json                # 추가 학습 데이터 (선택)
│   └── df_train_with_backtranslation.json   # 역번역 증강 데이터 (선택)
│
├── klue-roberta-large/                   # 사전학습 모델 (다운로드 필요)
│   └── config.json
│
├── models/                               # 학습된 모델 저장 폴더
│   ├── emoberta-large/                   # 감정 분석 모델 (참고용)
│   └── koelectra-small-v3-nsmc/          # 감정 분석 모델 (참고용)
│
├── src/                                  # 소스 코드
│   ├── config.py                         # 설정 파일 (하이퍼파라미터)
│   ├── data_loader.py                    # 데이터 로더
│   ├── model.py                          # 모델 정의
│   ├── train.py                          # 일반 학습 스크립트
│   ├── train_cross_validation.py         # 5-Fold CV 학습 스크립트
│   ├── inference.py                      # 단일 모델 추론
│   ├── ensemble_inference_cv.py          # 10-모델 앙상블 추론
│   ├── run_ensemble.py                   # 앙상블 실행 유틸
│   └── utils.py                          # 유틸리티 함수
│
├── submission/                           # 제출 파일 저장 폴더
│   ├── ensemble_cv_10models_*.json       # CV 앙상블 결과
│   └── submission_*.json                 # 단일 모델 결과
│
├── logs/                                 # 로그 파일 저장 폴더 (자동 생성)
│
├── run_cross_validation.sh               # 5-Fold CV 학습 + 앙상블 추론 스크립트
├── run_training.sh                       # 일반 학습 + 추론 스크립트
├── requirements.txt                      # Python 패키지 의존성
├── EDA_일상대화연결.ipynb                  # 데이터 탐색 노트북
└── README.md                             # 이 파일
```

## 🚀 사용 방법

### 1️⃣ 사전 준비

#### 1-1. 모델 다운로드
KLUE RoBERTa-large 모델을 다운로드하여 `klue-roberta-large/` 폴더에 배치합니다.

```bash
# Hugging Face에서 다운로드 (예시)
# https://huggingface.co/klue/roberta-large
```

#### 1-2. Python 패키지 설치
```bash
pip install -r requirements.txt
```

**필수 패키지:**
- `torch>=2.0.0` - PyTorch 딥러닝 프레임워크
- `transformers>=4.30.0` - Hugging Face Transformers
- `scikit-learn>=1.3.0` - 데이터 분할 및 평가
- `numpy>=1.24.0` - 수치 연산
- `tqdm>=4.65.0` - 진행 표시줄
- `matplotlib>=3.7.0` - 시각화

#### 1-3. 데이터 준비
`dataset/` 폴더에 다음 파일이 있는지 확인:
- ✅ `nikluge-2025-일상대화연결.train.json` (필수)
- ✅ `nikluge-2025-일상대화연결.dev.json` (필수)
- ✅ `nikluge-2025-일상대화연결.test.json` (필수)
- ⭕ `connect_sentence.json` (선택 - 추가 학습 데이터)

---

### 2️⃣ 학습 및 추론 실행

#### 🎯 방법 1: 5-Fold Cross Validation + 10-모델 앙상블 (권장)

**`run_cross_validation.sh` 실행**

이 스크립트는 다음 작업을 자동으로 수행합니다:
1. **5-Fold Cross Validation 학습** (선행문 5개 + 후행문 5개 = 총 10개 모델)
2. **10-모델 앙상블 추론** (Forward 5개 + Reversed 5개)
3. **결과 파일 생성** (`submission/ensemble_cv_10models_{JOB_ID}.json`)

##### Windows에서 실행 (Git Bash 사용)
```bash
# Git Bash 열기
cd /d/projects/8th-NLP-malpyeong/03_한국어_일상_대화_연결/roberta

# 실행 권한 부여
chmod +x run_cross_validation.sh

# 스크립트 실행
./run_cross_validation.sh
```

##### Windows PowerShell에서 실행
```powershell
# PowerShell에서는 bash 스크립트를 직접 실행할 수 없으므로
# WSL 또는 Git Bash 사용 권장

# 또는 Python 스크립트를 직접 실행
cd src
python train_cross_validation.py
python ensemble_inference_cv.py --job_id manual --output ../submission/ensemble_cv_10models_manual.json
```

##### Linux/SLURM 환경에서 실행 (서버)
```bash
# SLURM Job 제출
sbatch run_cross_validation.sh

# Job 상태 확인
squeue -u $USER

# 로그 확인
tail -f logs/cv_*.out
```

**생성되는 파일:**
- `models/best_model_선행문_fold{0-4}_{JOB_ID}.pt` (5개)
- `models/best_model_후행문_fold{0-4}_{JOB_ID}.pt` (5개)
- `models/history_*_fold*_{JOB_ID}.json` (학습 히스토리)
- `cross_validation_results_{JOB_ID}.json` (CV 결과 요약)
- `submission/ensemble_cv_10models_{JOB_ID}.json` (최종 제출 파일)

**앙상블 전략:**
- 각 테스트 샘플에 대해 10개 모델의 예측 확률을 평균하여 최종 예측
- Forward 예측 (선행문 5개 모델) + Reversed 예측 (후행문 5개 모델)

---

#### 🎯 방법 2: 일반 학습 (단일 모델)

**`run_training.sh` 실행**

선행문/후행문 각각 1개씩만 학습하고 추론합니다.

##### Windows Git Bash
```bash
cd /d/projects/8th-NLP-malpyeong/03_한국어_일상_대화_연결/roberta
chmod +x run_training.sh
./run_training.sh
```

##### Windows PowerShell
```powershell
cd d:\projects\8th-NLP-malpyeong\03_한국어_일상_대화_연결\roberta\src
python train.py
python inference.py
```

##### Linux/SLURM
```bash
sbatch run_training.sh
```

**생성되는 파일:**
- `models/best_model_선행문_{JOB_ID}.pt`
- `models/best_model_후행문_{JOB_ID}.pt`
- `submission/submission.json`

---

### 3️⃣ 설정 변경

`src/config.py` 파일에서 하이퍼파라미터를 수정할 수 있습니다:

```python
class Config:
    # 모델 설정
    MODEL_NAME = '../klue-roberta-large'
    MAX_LENGTH = 128
    DROPOUT_RATE = 0.3
    
    # 학습 파라미터
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 30
    
    # Cross Validation
    USE_CROSS_VALIDATION = True  # True면 CV 사용
    N_FOLDS = 5                  # Fold 개수
    CV_SEED = 42                 # 랜덤 시드
    
    # Early Stopping
    PATIENCE = 5                 # 개선 없으면 조기 종료
    MIN_DELTA = 0.001
```

---

### 4️⃣ 추론만 실행

이미 학습된 모델로 추론만 실행하려면:

```bash
cd src

# 단일 모델 추론
export JOB_ID=12345  # 학습 시 사용한 JOB_ID
python inference.py

# 10-모델 앙상블 추론
python ensemble_inference_cv.py \
    --job_id 12345 \
    --output ../submission/ensemble_cv_10models_12345.json \
    --test_path ../dataset/nikluge-2025-일상대화연결.test.json \
    --model_dir ../models
```

---

## 📊 결과 확인

### Cross Validation 결과
```bash
cat cross_validation_results_{JOB_ID}.json
```

### 제출 파일 확인
```bash
# 샘플 개수 확인
python -c "import json; print(len(json.load(open('submission/ensemble_cv_10models_{JOB_ID}.json'))))"

# 예측 분포 확인
python -c "
import json
from collections import Counter
data = json.load(open('submission/ensemble_cv_10models_{JOB_ID}.json'))
print(Counter([x['output'] for x in data]))
"
```

---

## 🛠️ 문제 해결

### 1. CUDA Out of Memory
- `config.py`에서 `BATCH_SIZE`를 줄이기 (16 → 8)
- `MAX_LENGTH`를 줄이기 (128 → 64)

### 2. 모델 파일을 찾을 수 없음
- `klue-roberta-large/` 폴더에 모델이 있는지 확인
- `config.py`의 `MODEL_NAME` 경로 확인

### 3. 데이터 파일 오류
- `dataset/` 폴더에 필수 파일이 있는지 확인
- JSON 파일 형식이 올바른지 확인

### 4. Windows에서 .sh 파일 실행 안 됨
- **Git Bash 사용** (권장)
- WSL(Windows Subsystem for Linux) 설치 후 사용
- 또는 Python 스크립트를 직접 실행

---

## 📝 추가 정보

### 모델 구조
- **Base Model:** KLUE RoBERTa-large
- **Task:** 이진 분류 (선행문 vs 후행문)
- **Special Tokens:** `[PREV]`, `[NEXT]`, `[POS]`, `[NEG]`, `[NEU]`, `[NEGT]`, `[NONEGT]`
- **입력 형식:** `[PREV] 발화1 [NEXT] 발화2 [POS] [NEGT]`

### Cross Validation 전략
- **Stratified 5-Fold:** 각 fold에서 클래스 비율 유지
- **Independent Training:** 선행문/후행문 독립적으로 학습
- **Ensemble:** 10개 모델의 소프트 보팅 (확률 평균)

### 성능 향상 팁
1. ✅ 5-Fold CV + 앙상블 사용 (가장 중요!)
2. ✅ Label Smoothing 적용 (`LABEL_SMOOTHING = 0.1`)
3. ✅ Dropout 조정 (`DROPOUT_RATE = 0.3`)
4. ✅ Learning Rate Warmup (`WARMUP_RATIO = 0.1`)
5. ⭕ 데이터 증강 (`connect_sentence.json` 사용)
6. ⭕ Gradient Accumulation (큰 배치 효과)

---

## 📮 문의
프로젝트 관련 문의사항은 이슈로 남겨주세요.
