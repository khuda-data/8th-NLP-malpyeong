"""
데이터 경로 설정
서버 환경에 맞게 경로를 수정하세요.
"""

import os
from pathlib import Path

# 프로젝트 루트 경로 (코드 위치)
PROJECT_ROOT = Path(__file__).parent.parent

# 데이터 경로 설정
# 현재 구조: 01_parliamentary_summary/code/src/paths.py
# 데이터 위치: 01_parliamentary_summary/data/
if os.path.exists("/data/guswns0429"):
    # Aurora 서버 환경: 데이터는 별도 디렉토리에
    DATA_ROOT = Path("/data/guswns0429/datasets/parliamentary_summary")
else:
    # 로컬/Git 레포 환경: 같은 레벨의 data 폴더
    DATA_ROOT = Path(__file__).parent.parent.parent / "data"

# 데이터 하위 경로
RAW_DATA_DIR = DATA_ROOT / "raw"
PREPROCESSED_DATA_DIR = DATA_ROOT / "preprocessed"
EXPERIMENTS_DATA_DIR = DATA_ROOT / "experiments"
OUTPUT_DIR = DATA_ROOT / "output"

# 출력 경로
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# 디렉토리 생성
for dir_path in [RAW_DATA_DIR, PREPROCESSED_DATA_DIR, EXPERIMENTS_DATA_DIR, 
                 OUTPUT_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 데이터 파일명 (영어)
DATA_FILES = {
    "train": "parliamentary_summary_train.json",
    "dev": "parliamentary_summary_dev.json",
    "test": "parliamentary_summary_test.json",
    "dev_sample": "parliamentary_summary_dev_sample.json",
}

def get_data_path(split: str = "dev") -> Path:
    """데이터 파일 경로 반환"""
    filename = DATA_FILES.get(split, f"parliamentary_summary_{split}.json")
    return RAW_DATA_DIR / filename

def get_preprocessed_path(split: str = "dev") -> Path:
    """전처리된 데이터 파일 경로 반환"""
    filename = f"parliamentary_summary_{split}_preprocessed.json"
    return PREPROCESSED_DATA_DIR / filename

