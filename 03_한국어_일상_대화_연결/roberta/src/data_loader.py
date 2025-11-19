"""
데이터 로더 및 전처리 모듈
- Pair-wise 샘플 생성
- 부정 표현 감지
- 감정 분석
- Special Token 추가
"""

import json
import os
import tempfile
import threading
from collections import OrderedDict
from typing import List, Dict, Tuple

from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT_MODEL_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'models', 'koelectra-small-v3-nsmc')
)


class TransformerSentimentAnalyzer:
    """KoELECTRA-NSMC 기반 한국어 문장 감성/부정 표현 분석기.

    - 기본 모델: daekeun-ml/koelectra-small-v3-nsmc
      (로컬 경로: ../models/koelectra-small-v3-nsmc)
    - 2클래스(binary) 로짓을 (positive / negative / neutral) 3값으로 변환해서 반환.
    - 환경변수:
        SENTIMENT_MODEL_NAME    : 모델 경로/이름 (기본은 위 로컬 경로)
        SENTIMENT_MAX_LENGTH    : 최대 토큰 길이 (기본 192)
        NEGATION_PROB_THRESHOLD : 부정(negative) 판정 임계값 (기본 0.45)
        SENTIMENT_CACHE_SIZE    : LRU 캐시 크기
    """

    MODEL_NAME = os.environ.get('SENTIMENT_MODEL_NAME', SENTIMENT_MODEL_DEFAULT)
    MAX_LENGTH = int(os.environ.get('SENTIMENT_MAX_LENGTH', '192'))
    NEGATION_THRESHOLD = float(os.environ.get('NEGATION_PROB_THRESHOLD', '0.45'))
    CACHE_SIZE = int(os.environ.get('SENTIMENT_CACHE_SIZE', '4096'))

    _tokenizer = None
    _model = None
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _cache: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    _lock = threading.Lock()

    # id2label이 "0", "1"이라 키워드에 0/1도 살짝 넣어줌
    LABEL_KEYWORDS = {
        'positive': ['positive', 'pos', '긍정', '1'],
        'negative': ['negative', 'neg', '부정', '0'],
        'neutral' : ['neutral', 'neu', '중립']
    }

    @classmethod
    def _ensure_model(cls):
        if cls._model is not None and cls._tokenizer is not None:
            return
        with cls._lock:
            if cls._model is not None and cls._tokenizer is not None:
                return
            print(f"[Sentiment] Loading model from {cls.MODEL_NAME}")
            cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            cls._model = AutoModelForSequenceClassification.from_pretrained(cls.MODEL_NAME)
            cls._model.to(cls._device)
            cls._model.eval()
            print(f"[Sentiment] num_labels={cls._model.config.num_labels}, id2label={cls._model.config.id2label}")

    @classmethod
    def _normalize_label(cls, raw_label: str, idx: int) -> str:
        """HF id2label 문자열을 'positive' / 'negative' / 'neutral' 중 하나로 정규화."""
        label = (raw_label or '').lower()

        # 1) 키워드 기반 매핑
        for canonical, keywords in cls.LABEL_KEYWORDS.items():
            if any(keyword in label for keyword in keywords):
                return canonical

        # 2) label_0, label_1, ... 형태일 때 인덱스로 매핑
        if label.startswith('label_'):
            try:
                index = int(label.split('_')[-1])
            except ValueError:
                index = idx
            mapping = ['negative', 'neutral', 'positive'] if cls._model.config.num_labels >= 3 else ['negative', 'positive']
            if index < len(mapping):
                return mapping[index]

        # 3) 2클래스 모델일 때 기본값: 0=negative, 1=positive
        if cls._model.config.num_labels == 2:
            return 'negative' if idx == 0 else 'positive'

        # 4) 그 외에는 0=negative, 마지막=positive, 나머지는 neutral
        if idx == 0:
            return 'negative'
        if idx == cls._model.config.num_labels - 1:
            return 'positive'
        return 'neutral'

    @classmethod
    def _cache_get(cls, key: str):
        with cls._lock:
            cached = cls._cache.get(key)
            if cached is not None:
                cls._cache.move_to_end(key)
                return cached.copy()
        return None

    @classmethod
    def _cache_set(cls, key: str, value: Dict[str, float]):
        with cls._lock:
            cls._cache[key] = value.copy()
            if len(cls._cache) > cls.CACHE_SIZE:
                cls._cache.popitem(last=False)

    @classmethod
    def predict_proba(cls, text: str) -> Dict[str, float]:
        """문장 하나에 대해 (positive, negative, neutral) 확률 반환."""
        normalized = (text or '').strip()
        if not normalized:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

        cached = cls._cache_get(normalized)
        if cached is not None:
            return cached

        cls._ensure_model()
        encoded = cls._tokenizer(
            normalized,
            truncation=True,
            padding='max_length',
            max_length=cls.MAX_LENGTH,
            return_tensors='pt'
        )
        encoded = {k: v.to(cls._device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = cls._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[0]

        # === 2클래스(koelectra-nsmc) 전용 처리 ===
        if cls._model.config.num_labels == 2:
            # KoELECTRA NSMC: 0=부정, 1=긍정 (id2label: {"0":"0","1":"1"})
            p_neg_raw = float(probs[0].item())
            p_pos_raw = float(probs[1].item()) if probs.shape[0] > 1 else 1.0 - p_neg_raw

            # pos/neg가 비슷하면 중립 점수를 높여주는 휴리스틱:
            #   delta = |p_pos - p_neg|
            #   delta가 0에 가까울수록 neutral ↑, 0.5 이상이면 사실상 양극단 → neutral 거의 0
            delta = abs(p_pos_raw - p_neg_raw)
            p_neu = max(0.0, 1.0 - 2.0 * delta)  # delta=0 → 1, delta=0.5 → 0, delta>0.5 → 0

            remain = 1.0 - p_neu
            denom = p_pos_raw + p_neg_raw
            if denom > 0:
                p_pos = remain * p_pos_raw / denom
                p_neg = remain * p_neg_raw / denom
            else:
                p_pos = p_neg = remain * 0.5

            scores: Dict[str, float] = {
                'positive': p_pos,
                'negative': p_neg,
                'neutral': p_neu,
            }
        else:
            # === 일반 멀티클래스 모델 처리 (기존 로직) ===
            scores: Dict[str, float] = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            for idx, score in enumerate(probs):
                raw_label = cls._model.config.id2label.get(idx, f'label_{idx}')
                canonical = cls._normalize_label(raw_label, idx)
                scores[canonical] = max(scores[canonical], float(score.item()))

            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}
            else:
                scores['neutral'] = 1.0

        cls._cache_set(normalized, scores)
        return scores.copy()

    @classmethod
    def predict_label(cls, text: str) -> str:
        """가장 확률이 높은 감성 레이블 하나 반환 (positive / negative / neutral)."""
        scores = cls.predict_proba(text)
        return max(scores, key=scores.get)


class NegationDetector:
    """KoELECTRA 감성 모델을 이용한 부정 표현 감지기"""

    @classmethod
    def detect(cls, text: str) -> bool:
        scores = TransformerSentimentAnalyzer.predict_proba(text)
        # negative 확률이 임계값 이상이면 '부정 표현이 있다'로 간주
        return scores['negative'] >= TransformerSentimentAnalyzer.NEGATION_THRESHOLD

    @classmethod
    def count(cls, text: str) -> int:
        return int(cls.detect(text))


class EmotionAnalyzer:
    """감정 레이블/스코어 헬퍼"""

    @classmethod
    def analyze(cls, text: str) -> str:
        return TransformerSentimentAnalyzer.predict_label(text)

    @classmethod
    def get_emotion_score(cls, text: str) -> Tuple[float, float, float]:
        scores = TransformerSentimentAnalyzer.predict_proba(text)
        return scores['positive'], scores['negative'], scores['neutral']


class DialogueDataset(Dataset):
    """대화 연결 데이터셋 클래스 (Pair-wise 방식)"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 128,
        position_filter: str = None  # '선행문' 또는 '후행문'
    ):
        """
        Args:
            data_path: JSON 데이터 파일 경로
            tokenizer: Hugging Face Tokenizer
            max_length: 최대 시퀀스 길이
            position_filter: 특정 발화위치만 필터링 (None이면 전체)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.position_filter = position_filter
        
        # Special tokens
        self.special_tokens = {
            'positive': '[POS]',
            'negative': '[NEG]',
            'neutral': '[NEU]',
            'has_negation': '[NEGT]',
            'no_negation': '[NONEGT]'
        }
        
        # Special tokens를 tokenizer에 추가
        self._add_special_tokens()
        
        # 데이터 로드 및 pair-wise 샘플 생성
        self.samples = self._load_and_process_data(data_path)
    
    def _add_special_tokens(self):
        """Special tokens를 tokenizer에 추가"""
        special_tokens_dict = {
            'additional_special_tokens': list(self.special_tokens.values())
        }
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
        print(f"Added {num_added_tokens} special tokens to tokenizer")
    
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """데이터 로드 및 pair-wise 샘플 생성 (순서 바꿔서 활용)"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        for item in data:
            input_data = item['input']
            target_utterance = input_data['대상발화']
            position = input_data['발화위치']
            
            # 정답 레이블 (train/dev에만 존재)
            correct_label = item.get('output', None)
            
            # 각 선택지에 대해 pair-wise 샘플 생성
            for choice_num in range(1, 4):
                choice_key = f'발화{choice_num}'
                choice_text = input_data[choice_key]
                
                # 레이블: 정답이면 1, 오답이면 0
                if correct_label:
                    label = 1 if correct_label == choice_key else 0
                else:
                    label = -1  # test 데이터
                
                # 부정 표현 감지
                target_has_neg = NegationDetector.detect(target_utterance)
                choice_has_neg = NegationDetector.detect(choice_text)
                
                # 감정 분석
                target_emotion = EmotionAnalyzer.analyze(target_utterance)
                choice_emotion = EmotionAnalyzer.analyze(choice_text)
                
                # 1) 원본 위치 데이터: position_filter와 일치하면 추가
                if self.position_filter is None or position == self.position_filter:
                    samples.append({
                        'id': item['id'],
                        'target_utterance': target_utterance,
                        'choice_text': choice_text,
                        'choice_num': choice_num,
                        'position': position,
                        'label': label,
                        'target_has_negation': target_has_neg,
                        'choice_has_negation': choice_has_neg,
                        'target_emotion': target_emotion,
                        'choice_emotion': choice_emotion
                    })
                
                # 2) 정답인 경우에만 순서를 바꿔서 추가
                # 선행문 학습 시: 후행문 데이터를 순서 바꿔서 선행문으로
                # 후행문 학습 시: 선행문 데이터를 순서 바꿔서 후행문으로
                if label == 1 and self.position_filter:  # 정답이고 position_filter가 지정된 경우
                    opposite_position = '후행문' if position == '선행문' else '선행문'
                    
                    # 반대 위치가 필터와 일치하면 순서를 바꿔서 추가
                    if opposite_position == self.position_filter:
                        # 순서 반대: choice_text와 target_utterance 위치 교환
                        samples.append({
                            'id': item['id'] + '_reversed',
                            'target_utterance': choice_text,  # 순서 바꿈
                            'choice_text': target_utterance,  # 순서 바꿈
                            'choice_num': choice_num,
                            'position': opposite_position,  # 반대 위치로 설정
                            'label': 1,  # 정답 유지
                            'target_has_negation': choice_has_neg,  # 순서 바꿈
                            'choice_has_negation': target_has_neg,  # 순서 바꿈
                            'target_emotion': choice_emotion,  # 순서 바꿈
                            'choice_emotion': target_emotion  # 순서 바꿈
                        })
        
        print(f"Loaded {len(samples)} pair-wise samples from {data_path}")
        if self.position_filter:
            print(f"Filtered by position: {self.position_filter} (including reversed)")
        
        return samples
    
    def _create_input_text(self, sample: Dict) -> str:
        """입력 텍스트 생성 (Special tokens 포함)"""
        # Target emotion token
        target_emotion_token = self.special_tokens[sample['target_emotion']]
        
        # Choice emotion token
        choice_emotion_token = self.special_tokens[sample['choice_emotion']]
        
        # Target negation token
        target_neg_token = (self.special_tokens['has_negation'] 
                           if sample['target_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        # Choice negation token
        choice_neg_token = (self.special_tokens['has_negation'] 
                           if sample['choice_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        # 입력 형식: [CLS] {target_emotion} {target_neg} {대상발화} [SEP] {choice_emotion} {choice_neg} {선택지} [SEP]
        input_text = (
            f"{target_emotion_token} {target_neg_token} "
            f"{sample['target_utterance']} "
            f"{self.tokenizer.sep_token} "
            f"{choice_emotion_token} {choice_neg_token} "
            f"{sample['choice_text']}"
        )
        
        return input_text
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 입력 텍스트 생성
        input_text = self._create_input_text(sample)
        
        # Tokenization
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'id': sample['id'],
            'choice_num': sample['choice_num'],
            'position': sample['position']
        }


class TestDialogueDataset(Dataset):
    """테스트 데이터셋 클래스 (예측용)"""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        max_length: int = 128,
        position_filter: str = None
    ):
        """
        Args:
            data_path: JSON 데이터 파일 경로
            tokenizer: Hugging Face Tokenizer
            max_length: 최대 시퀀스 길이
            position_filter: 특정 발화위치만 필터링
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.position_filter = position_filter
        
        # Special tokens (동일하게 적용)
        self.special_tokens = {
            'positive': '[POS]',
            'negative': '[NEG]',
            'neutral': '[NEU]',
            'has_negation': '[NEGT]',
            'no_negation': '[NONEGT]'
        }
        
        # 데이터 로드
        self.samples = self._load_and_process_data(data_path)
    
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """테스트 데이터 로드 및 처리"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        for item in data:
            input_data = item['input']
            target_utterance = input_data['대상발화']
            position = input_data['발화위치']
            
            # 발화위치 필터링
            if self.position_filter and position != self.position_filter:
                continue
            
            # 각 선택지에 대해 샘플 생성
            for choice_num in range(1, 4):
                choice_key = f'발화{choice_num}'
                choice_text = input_data[choice_key]
                
                # 부정 표현 감지
                target_has_neg = NegationDetector.detect(target_utterance)
                choice_has_neg = NegationDetector.detect(choice_text)
                
                # 감정 분석
                target_emotion = EmotionAnalyzer.analyze(target_utterance)
                choice_emotion = EmotionAnalyzer.analyze(choice_text)
                
                samples.append({
                    'id': item['id'],
                    'target_utterance': target_utterance,
                    'choice_text': choice_text,
                    'choice_num': choice_num,
                    'position': position,
                    'target_has_negation': target_has_neg,
                    'choice_has_negation': choice_has_neg,
                    'target_emotion': target_emotion,
                    'choice_emotion': choice_emotion
                })
        
        print(f"Loaded {len(samples)} test samples from {data_path}")
        if self.position_filter:
            print(f"Filtered by position: {self.position_filter}")
        
        return samples
    
    def _create_input_text(self, sample: Dict) -> str:
        """입력 텍스트 생성 (학습 데이터와 동일)"""
        target_emotion_token = self.special_tokens[sample['target_emotion']]
        choice_emotion_token = self.special_tokens[sample['choice_emotion']]
        
        target_neg_token = (self.special_tokens['has_negation'] 
                           if sample['target_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        choice_neg_token = (self.special_tokens['has_negation'] 
                           if sample['choice_has_negation'] 
                           else self.special_tokens['no_negation'])
        
        input_text = (
            f"{target_emotion_token} {target_neg_token} "
            f"{sample['target_utterance']} "
            f"{self.tokenizer.sep_token} "
            f"{choice_emotion_token} {choice_neg_token} "
            f"{sample['choice_text']}"
        )
        
        return input_text
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 입력 텍스트 생성
        input_text = self._create_input_text(sample)
        
        # Tokenization
        encoding = self.tokenizer(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'id': sample['id'],
            'choice_num': sample['choice_num'],
            'position': sample['position']
        }


def load_data_for_position(
    train_path: str,
    dev_path: str,
    tokenizer,
    position: str,
    max_length: int = 128,
    batch_size: int = 16,
    connect_sentence_path: str = None
):
    """특정 발화위치에 대한 데이터 로더 생성
    
    Args:
        train_path: 학습 데이터 경로
        dev_path: 검증 데이터 경로
        tokenizer: Tokenizer
        position: '선행문' 또는 '후행문'
        max_length: 최대 시퀀스 길이
        batch_size: 배치 크기
        connect_sentence_path: connect_sentence.json 경로 (선택사항)
    
    Returns:
        train_loader, dev_loader
    """
    from torch.utils.data import DataLoader, ConcatDataset
    import tempfile
    
    train_dataset = DialogueDataset(
        train_path, 
        tokenizer, 
        max_length=max_length,
        position_filter=position
    )
    
    # connect_sentence.json 데이터 추가
    if connect_sentence_path and connect_sentence_path.strip() and os.path.exists(connect_sentence_path):
        with open(connect_sentence_path, 'r', encoding='utf-8') as f:
            connect_data = json.load(f)
        
        # connect_sentence.json을 nikluge 형식으로 변환
        converted_data = []
        for idx, item in enumerate(connect_data):
            converted_item = {
                'id': f'connect_{idx}',
                'input': {
                    '대상발화': item['대상발화'],
                    '발화위치': item['발화위치'],
                    '발화1': item['발화'],
                    '발화2': '',
                    '발화3': ''
                },
                'output': '발화1'
            }
            converted_data.append(converted_item)
        
        # 임시 파일로 저장하여 DialogueDataset으로 로드
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', delete=False) as tmp:
            json.dump(converted_data, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        
        connect_dataset = DialogueDataset(
            tmp_path,
            tokenizer,
            max_length=max_length,
            position_filter=position
        )
        
        os.unlink(tmp_path)
        
        # 두 데이터셋 합치기
        train_dataset = ConcatDataset([train_dataset, connect_dataset])
        print(f"✓ connect_sentence.json 데이터 추가 완료")
    else:
        print(f"ℹ connect_sentence.json 사용 안 함 (TRAIN_PATH + DEV_PATH만 사용)")
    
    dev_dataset = DialogueDataset(
        dev_path, 
        tokenizer, 
        max_length=max_length,
        position_filter=position
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows 호환성
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, dev_loader


def load_data_for_cross_validation(
    train_path: str,
    dev_path: str,
    tokenizer,
    position: str,
    fold: int,
    n_folds: int = 5,
    max_length: int = 128,
    batch_size: int = 16,
    seed: int = 42,
    connect_sentence_path: str = None
):
    """K-Fold Cross Validation을 위한 데이터 로더 생성
    
    train_path와 dev_path의 데이터를 합쳐서 n_folds로 나눈 후,
    fold 번째를 validation으로, 나머지를 training으로 사용
    connect_sentence_path가 제공되면 해당 데이터도 함께 로드
    
    Args:
        train_path: 학습 데이터 경로
        dev_path: 검증 데이터 경로  
        tokenizer: Tokenizer
        position: '선행문' 또는 '후행문'
        fold: 현재 fold 번호 (0부터 n_folds-1까지)
        n_folds: 총 fold 개수 (기본값: 5)
        max_length: 최대 시퀀스 길이
        batch_size: 배치 크기
        seed: 랜덤 시드
        connect_sentence_path: connect_sentence.json 경로 (선택사항)
    
    Returns:
        train_loader, dev_loader
    """
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import KFold
    import numpy as np
    
    # Train과 Dev 데이터 모두 로드
    full_dataset_list = []
    
    # Train 데이터 로드
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
        full_dataset_list.extend(train_data)
    
    # Dev 데이터 로드
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
        full_dataset_list.extend(dev_data)
    
    # connect_sentence.json 데이터 로드 및 변환
    if connect_sentence_path and connect_sentence_path.strip() and os.path.exists(connect_sentence_path):
        with open(connect_sentence_path, 'r', encoding='utf-8') as f:
            connect_data = json.load(f)
        
        # connect_sentence.json 형식을 nikluge 형식으로 변환
        # 각 항목이 {'대상발화', '발화위치', '발화'}로 구성됨
        converted_data = []
        for idx, item in enumerate(connect_data):
            converted_item = {
                'id': f'connect_{idx}',
                'input': {
                    '대상발화': item['대상발화'],
                    '발화위치': item['발화위치'],
                    '발화1': item['발화'],  # 정답
                    '발화2': '',  # 오답 (빈 문자열)
                    '발화3': ''   # 오답 (빈 문자열)
                },
                'output': '발화1'  # 발화1이 정답
            }
            converted_data.append(converted_item)
        
        full_dataset_list.extend(converted_data)
        print(f"✓ connect_sentence.json에서 {len(converted_data)}개 샘플 추가 로드")
    else:
        print(f"ℹ connect_sentence.json 사용 안 함 (TRAIN_PATH + DEV_PATH만 사용)")
    
    # 임시 파일로 저장 (DialogueDataset이 파일 경로를 받으므로)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', delete=False) as tmp:
        json.dump(full_dataset_list, tmp, ensure_ascii=False, indent=2)
        tmp_path = tmp.name
    
    # 전체 데이터셋 생성
    full_dataset = DialogueDataset(
        tmp_path,
        tokenizer,
        max_length=max_length,
        position_filter=position
    )
    
    # 임시 파일 삭제
    os.unlink(tmp_path)
    
    # K-Fold 분할
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(full_dataset))
    
    # fold 번째 분할 가져오기
    for i, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if i == fold:
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            break
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Fold {fold + 1}/{n_folds}:")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples: {len(val_idx)}")
    
    return train_loader, val_loader




if __name__ == '__main__':
    from transformers import AutoTokenizer

    print("=== Transformer 기반 감정/부정 태깅 테스트 ===")
    print(f"Sentiment model   : {TransformerSentimentAnalyzer.MODEL_NAME}")
    print(f"Sequence max len  : {TransformerSentimentAnalyzer.MAX_LENGTH}")
    print(f"Negation threshold: {TransformerSentimentAnalyzer.NEGATION_THRESHOLD}")

    tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')

    demo_texts = [
        "너무나도 실망스러운 하루였어",
        "정말 멋진 날이었어",
        "그냥 그런 날이야",
        "싫어하지는 않지만 좋아하지도 않아"
    ]

    print("\n=== 부정 표현/감정 판별 결과 ===")
    for sent in demo_texts:
        scores = TransformerSentimentAnalyzer.predict_proba(sent)
        label = EmotionAnalyzer.analyze(sent)
        has_neg = NegationDetector.detect(sent)
        print(
            f"'{sent}' -> emotion:{label:8s} neg:{has_neg} "
            f"(pos:{scores['positive']:.3f}, neg:{scores['negative']:.3f}, neu:{scores['neutral']:.3f})"
        )
