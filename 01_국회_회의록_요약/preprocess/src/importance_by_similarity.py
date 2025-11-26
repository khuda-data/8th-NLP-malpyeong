"""
SimCSE 기반 요약-발화 유사도로 중요도 판단 모듈

핵심 아이디어:
- 정답 요약(output)과 각 발화(utterance) 간 SimCSE 임베딩 유사도 계산
- 코사인 유사도가 높은 발화 = 요약에 포함될 가능성이 높은 발화 = 중요 발화
"""

import re
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# SimCSE 모델 (전역 변수로 한 번만 로드)
_model = None


def _get_simcse_model():
    """SimCSE 모델을 로드합니다 (한 번만 로드)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # 한국어 SimCSE 모델 사용
            _model = SentenceTransformer('BM-K/KoSimCSE-roberta-multitask')
        except ImportError:
            raise ImportError(
                "sentence-transformers가 설치되지 않았습니다. "
                "pip install sentence-transformers 실행 필요"
            )
        except Exception as e:
            logger.error(f"SimCSE 모델 로드 실패: {e}")
            raise
    return _model


def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 벡터 간 코사인 유사도를 계산합니다.
    
    Args:
        embedding1: 첫 번째 임베딩 벡터
        embedding2: 두 번째 임베딩 벡터
    
    Returns:
        코사인 유사도 (-1.0 ~ 1.0)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def calculate_similarity_scores(
    utterances: List[str],
    summary: str
) -> List[Tuple[str, float]]:
    """
    각 발화와 요약 간 SimCSE 기반 코사인 유사도를 계산합니다.
    
    Args:
        utterances: 발화 리스트
        summary: 정답 요약 텍스트
    
    Returns:
        [(발화, 유사도), ...] 리스트
    """
    if not utterances or not summary:
        return [(utt, 0.0) for utt in utterances]
    
    model = _get_simcse_model()
    
    # 태그 제거 (순수 발화 내용만 비교)
    clean_utterances = [re.sub(r'<[^>]+>', '', utt).strip() for utt in utterances]
    clean_summary = summary.strip()
    
    # 요약 임베딩 계산 (한 번만)
    summary_embedding = model.encode(clean_summary, convert_to_numpy=True)
    
    # 각 발화 임베딩 계산 및 유사도 계산
    results = []
    for i, clean_utt in enumerate(clean_utterances):
        if not clean_utt:
            results.append((utterances[i], 0.0))
            continue
        
        utterance_embedding = model.encode(clean_utt, convert_to_numpy=True)
        similarity = calculate_cosine_similarity(utterance_embedding, summary_embedding)
        results.append((utterances[i], similarity))
    
    return results


def apply_importance_tags_by_similarity(
    utterances: List[str],
    summary: str,
    threshold: float = 0.5
) -> List[str]:
    """
    SimCSE 기반 유사도로 <IMP> 태그를 적용합니다.
    
    Args:
        utterances: 발화 리스트
        summary: 정답 요약 텍스트
        threshold: 중요 발화 임계값 (코사인 유사도)
    
    Returns:
        <IMP> 태그가 적용된 발화 리스트
    """
    if not utterances or not summary:
        return utterances
    
    # 각 발화의 유사도 계산
    similarity_scores = calculate_similarity_scores(utterances, summary)
    
    # 임계값 이상인 발화에 태그 적용
    tagged_utterances = []
    for utterance, (_, similarity) in zip(utterances, similarity_scores):
        # 이미 <결정>이나 <쟁점> 태그가 있으면 <IMP> 불필요
        if '<결정>' in utterance or '<쟁점>' in utterance:
            tagged_utterances.append(utterance)
        elif similarity >= threshold:
            if '<IMP>' not in utterance:
                tagged_utterances.append(f'<IMP>{utterance}</IMP>')
            else:
                tagged_utterances.append(utterance)
        else:
            tagged_utterances.append(utterance)
    
    return tagged_utterances

