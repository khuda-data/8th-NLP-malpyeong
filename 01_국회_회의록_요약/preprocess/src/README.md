# 국회 회의록 요약 데이터 전처리

## 프로젝트 구조

```
preprocess/
├── src/                       # 소스 코드
│   ├── main.py               # 진입점
│   ├── preprocessor.py       # 메인 전처리 파이프라인
│   ├── boundary_detection.py # 안건 경계 및 결정 발화 탐지
│   ├── filtering.py          # 불용어 제거 및 발화 필터링
│   ├── tagging.py            # 중요도 태깅 (결정, 안건, 쟁점, KEY, IMP)
│   ├── utils.py              # 유틸리티 함수 (역할 정규화 등)
│   └── config.py             # 설정 및 상수
├── data/                     # 원본 데이터
├── output/                   # 전처리 결과물
└── experiments/              # 개인 실험/개발 폴더
    ├── jinsu/
    └── seungju/
```

## 동작 흐름

### 핵심 컨셉
**안건 시작 ID → 결정 발화 탐지 → 다음 안건 ID 전까지 끊어서 안건별 처리**

```
1. 안건 경계 탐지
   └─ sentence_id (안건 시작) → 결정 발화 찾기 → 다음 안건 시작 전까지
   
2. 안건별 전처리
   ├─ 불용어 제거 (짧은 발화, 반응어, 인사 등)
   └─ 태깅 적용 (결정, 안건, 쟁점, KEY, IMP)
   
3. 출력
   └─ [역할] 이름: <태그>발화내용</태그> 형식
```

### 상세 프로세스

#### 1단계: 안건 경계 탐지 (`boundary_detection.py`)
- **시작점**: `sentence_id` (안건 시작 인덱스)
- **종료점**: 우선순위 기반 탐지
  1. 같은 dialogue 내 다음 샘플의 `sentence_id` (100% 정확도)
  2. 다음 안건 시작 패턴 ("의사일정 제N항" + "상정")
  3. 결정 선포 문장 ("가결되었음을 선포" 등)
  4. 회의 종료 문장 ("산회", "마치겠습니다" 등)
  5. Fallback (기본값 50개 발화)
- **경계**: 위 우선순위에 따라 안건의 끝점을 찾아 경계 설정

#### 2단계: 안건별 전처리
- **불용어 제거** (`filtering.py`)
  - 10자 이하 짧은 발화 (의사진행 제외)
  - 단순 반응: "네", "예", "그렇습니다"
  - 인사/예의: "감사합니다", "수고하셨습니다"
  - 숫자 포함 문장은 보존
  - 의사진행 관련 키워드 포함 발화는 보존

- **태깅** (`tagging.py`)
  - `<결정>`: 가결, 부결, 의결, 통과 등
  - `<안건>`: 안건명 (최대 2회)
  - `<쟁점>`: 논란, 문제점, 반대/찬성 근거
  - `<KEY>`: 법령명, 부처명, 날짜, 법조항 (최대 2개)
  - `<IMP>`: 규칙 기반 중요 발화 (전문위원 보고, 법조항 언급, 정부 의견 등)

## 사용 방법

```bash
# preprocess 폴더에서 실행
cd preprocess

# 기본 실행 (data 폴더의 모든 JSON 파일 처리)
python -m src.main

# 특정 파일만 처리
python -m src.main --files 국회회의록안건별요약_dev.json

# 입력/출력 디렉토리 지정
python -m src.main --input_dir /path/to/input --output_dir /path/to/output
```

### 명령줄 옵션
- `--input_dir`: 입력 디렉토리 (기본값: `data`)
- `--output_dir`: 출력 디렉토리 (기본값: `output`)
- `--files`: 처리할 파일명 리스트

## 출력 형식

### 발화 형식
```
[소위원장] 양승조: <결정>의사일정 제1항 법률안을 상정합니다</결정>
[수석전문위원] 김종두: <IMP>법률안의 주요 내용은 다음과 같습니다...</IMP>
```

### JSON 형식
```json
{
  "id": "sample_id",
  "preprocessed_dialogue": "[소위원장] 양승조: <결정>의사일정 제1항...",
  "prompt": "다음은 국회 회의록 안건별 대화입니다...",
  "output": "요약 결과"
}
```

## 입력 데이터 구조

```json
{
  "id": "sample_id",
  "input": {
    "speaker": [
      {"id": "speaker_id", "occupation": "역할"}
    ],
    "conversation": [
      {"speaker": "발화자", "utterance": "발화내용"}
    ],
    "issue": {
      "sentence_id": 10,
      "keyword": "안건키워드",
      "topic": "안건명"
    }
  },
  "output": "요약 결과"
}
```
