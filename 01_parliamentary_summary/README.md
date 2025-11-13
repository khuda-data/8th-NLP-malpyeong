# 01_parliamentary_summary

국회 회의록 요약 프로젝트

## 팀 소개

KHUDA NLP 트랙으로 구성된 팀입니다.

- 정유진
- 신진수
- 다니아
- 김건우
- 이승주

## 프로젝트 개요

AI 말평에 제시된 국회 회의록 요약 과제를 해결하는 프로젝트입니다.

**평가지표**: ROUGE-1

## 디렉토리 구조

```
01_parliamentary_summary/
├── code/                    # 코드 저장소
│   ├── src/                # 전처리 소스 코드
│   │   ├── main.py        # 메인 진입점
│   │   ├── paths.py       # 경로 설정 (데이터 경로 참조)
│   │   ├── config.py      # 전처리 설정
│   │   └── ...
│   ├── experiments/        # 개인 실험/개발 폴더
│   │   ├── geonwoo/
│   │   ├── jinsu/
│   │   └── seungju/
│   └── README.md
│
└── data/                   # 데이터 저장소
    ├── raw/                # 원본 데이터
    ├── preprocessed/       # 전처리 결과
    └── experiments/        # 실험용 데이터
```

## 데이터 경로

**로컬/Git 레포 환경**:
- 데이터는 같은 폴더 내 `data/` 디렉토리에 있습니다.
- `code/src/paths.py`가 자동으로 상대 경로를 참조합니다.

**Aurora 서버 환경**:
- 코드: `/data/guswns0429/repos/parliamentary_summary/`
- 데이터: `/data/guswns0429/datasets/parliamentary_summary/`
- `paths.py`가 자동으로 서버 경로를 감지합니다.

## 사용 방법

### 1. 전처리 실행 (로컬/Git 레포)

```bash
cd code
python src/main.py
```

### 2. 서버 환경에서 실행

```bash
# Aurora 서버
cd /data/guswns0429/repos/parliamentary_summary/code
python src/main.py \
  --input_dir /data/guswns0429/datasets/parliamentary_summary/raw \
  --output_dir /data/guswns0429/datasets/parliamentary_summary/preprocessed
```

## 데이터 구조

```
data/
├── raw/                    # 원본 데이터
│   ├── parliamentary_summary_train.json
│   ├── parliamentary_summary_dev.json
│   └── parliamentary_summary_test.json
├── preprocessed/           # 전처리 결과
└── experiments/           # 실험용 데이터
```

자세한 내용은 [code/src/README.md](./code/src/README.md) 참고

