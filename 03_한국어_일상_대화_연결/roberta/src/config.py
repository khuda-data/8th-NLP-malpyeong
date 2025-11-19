"""
설정 파일
모든 하이퍼파라미터와 경로를 중앙에서 관리
"""

import os


class Config:
    """학습 및 추론 설정"""
    
    # 데이터 경로
    TRAIN_PATH = '../dataset/nikluge-2025-일상대화연결.train.json'
    DEV_PATH = '../dataset/nikluge-2025-일상대화연결.dev.json'
    TEST_PATH = '../dataset/nikluge-2025-일상대화연결.test.json'
    CONNECT_SENTENCE_PATH = '../dataset/connect_sentence.json'  # 추가 학습 데이터 (빈 문자열이면 사용 안 함)
    
    # 모델 설정
    MODEL_NAME = '../klue-roberta-large'  # 로컬 폴더명 (src/ 기준 상대 경로)
    MAX_LENGTH = 128
    DROPOUT_RATE = 0.3
    NUM_LABELS = 2
    
    # 학습 하이퍼파라미터
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 30
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    MAX_GRAD_NORM = 1.0
    
    # Early Stopping
    PATIENCE = 5
    MIN_DELTA = 0.001
    
    # 출력 경로
    OUTPUT_DIR = '../models'
    SUBMISSION_PATH = '../submission/submission.json'
    
    # Cross Validation 설정
    USE_CROSS_VALIDATION = True  # True로 설정하면 K-Fold CV 사용
    N_FOLDS = 5  # Cross Validation fold 수
    CV_SEED = 42  # Cross Validation 랜덤 시드
    
    @property
    def CURRENT_FOLD(self):
        """현재 fold (환경변수 우선, 없으면 0)"""
        return int(os.environ.get('CURRENT_FOLD', '0'))
    
    # Resume 학습 설정 (이어서 학습하려면 경로 지정, 새로 시작하려면 None)
    RESUME_PREV = None  # './models/best_model_선행문_12345.pt'
    RESUME_NEXT = None  # './models/best_model_후행문_12345.pt'
    
    # Special Tokens
    SPECIAL_TOKENS = {
        'additional_special_tokens': [
            '[PREV]',      # 선행문
            '[NEXT]',      # 후행문
            '[POS]',       # 긍정 감정
            '[NEG]',       # 부정 감정
            '[NEU]',       # 중립 감정
            '[NEGT]',      # 부정 표현 있음
            '[NONEGT]'     # 부정 표현 없음
        ]
    }
    
    @property
    def JOB_ID(self):
        """Job ID (환경변수에서 가져오기, 없으면 빈 문자열)"""
        return os.environ.get('JOB_ID', '')
    
    # 디바이스 설정
    @staticmethod
    def get_device():
        """디바이스 자동 선택"""
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionConfig:
    """발화위치별 설정 (필요시 개별 조정)"""
    
    PREV_CONFIG = {
        'position': '선행문',
        'model_path': './models/best_model_선행문.pt',
        'history_path': './models/history_선행문.json'
    }
    
    NEXT_CONFIG = {
        'position': '후행문',
        'model_path': './models/best_model_후행문.pt',
        'history_path': './models/history_후행문.json'
    }


if __name__ == '__main__':
    # 설정 확인
    print("="*60)
    print("현재 설정")
    print("="*60)
    
    config = Config()
    
    print("\n[데이터 경로]")
    print(f"  Train: {config.TRAIN_PATH}")
    print(f"  Dev: {config.DEV_PATH}")
    print(f"  Test: {config.TEST_PATH}")
    
    print("\n[모델 설정]")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Max Length: {config.MAX_LENGTH}")
    print(f"  Dropout: {config.DROPOUT_RATE}")
    
    print("\n[학습 하이퍼파라미터]")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Early Stopping Patience: {config.PATIENCE}")
    
    print("\n[Special Tokens]")
    for token in config.SPECIAL_TOKENS['additional_special_tokens']:
        print(f"  {token}")
    
    print(f"\n[디바이스]")
    print(f"  {config.get_device()}")
    
    print("\n" + "="*60)
