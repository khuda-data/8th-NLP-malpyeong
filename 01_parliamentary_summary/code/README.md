# Parliamentary Summary Project

국회 회의록 요약 프로젝트 코드 저장소

## 디렉토리 구조

```
parliamentary_summary_code/
├── src/                    # 소스 코드
│   ├── main.py            # 메인 진입점
│   ├── paths.py           # 경로 설정 (데이터 경로 참조)
│   ├── config.py          # 전처리 설정
│   ├── preprocessor.py    # 전처리 로직
│   └── ...
├── experiments/            # 실험 코드
│   ├── geonwoo/
│   ├── jinsu/
│   └── seungju/
└── README.md
```

## 데이터 경로

데이터는 별도 디렉토리에 저장됩니다:

- **로컬**: `../parliamentary_summary_data/`
- **Aurora 서버**: `/data/guswns0429/datasets/parliamentary_summary/`

경로는 `src/paths.py`에서 자동으로 감지합니다.

## 사용 방법

### 1. 전처리 실행

```bash
python src/main.py
```

또는 특정 파일만 처리:

```bash
python src/main.py --files parliamentary_summary_dev.json
```

### 2. 데이터 경로 지정

```bash
python src/main.py \
  --input_dir /data/guswns0429/datasets/parliamentary_summary/raw \
  --output_dir /data/guswns0429/datasets/parliamentary_summary/preprocessed
```

## 데이터 구조

데이터는 다음 구조로 저장됩니다:

```
parliamentary_summary_data/
├── raw/                    # 원본 데이터
│   ├── parliamentary_summary_train.json
│   ├── parliamentary_summary_dev.json
│   └── parliamentary_summary_test.json
├── preprocessed/           # 전처리 결과
└── experiments/            # 실험용 데이터
```

## 서버 환경 설정

Aurora 서버에서 사용 시:

1. 코드는 `/data/guswns0429/repos/parliamentary_summary/`에 저장
2. 데이터는 `/data/guswns0429/datasets/parliamentary_summary/`에 저장
3. `paths.py`가 자동으로 경로를 감지합니다

