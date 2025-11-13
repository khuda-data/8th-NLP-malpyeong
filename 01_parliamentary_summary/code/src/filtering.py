"""
발화 필터링 모듈
"""

import re
import logging
from typing import Dict, List, Optional

try:
    from .config import (
        GENERAL_KEYWORDS, PROCEDURE_KEYWORDS, MIN_UTTERANCES,
        MIN_UTTERANCE_LENGTH, MIN_UTTERANCE_LENGTH_STRICT,
        MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS, GOVERNMENT_ORGS
    )
except ImportError:
    from config import (
        GENERAL_KEYWORDS, PROCEDURE_KEYWORDS, MIN_UTTERANCES,
        MIN_UTTERANCE_LENGTH, MIN_UTTERANCE_LENGTH_STRICT,
        MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS, GOVERNMENT_ORGS
    )

logger = logging.getLogger(__name__)


def extract_keyword_core_terms(keyword: str) -> List[str]:
    """
    keyword에서 핵심 단어를 추출합니다.
    
    예: "학교보건법 일부개정법률안" -> ["학교보건법", "보건법", "학교", "일부개정", "법률안"]
    """
    if not keyword:
        return []
    
    keyword_clean = re.sub(r'\([^)]*\)', '', keyword)
    core_terms = []
    
    # 법률명 패턴
    law_pattern = r'([가-힣]{2,}(?:법|법률|법안))'
    laws = re.findall(law_pattern, keyword_clean)
    core_terms.extend(laws)
    
    # 핵심 명사 추출
    nouns = re.findall(r'[가-힣]{2,}', keyword_clean)
    stop_nouns = {'일부', '개정', '법률', '안건', '의원', '대표', '발의', '관한', '에관한'}
    nouns = [n for n in nouns if n not in stop_nouns and len(n) >= 2]
    core_terms.extend(nouns)
    
    # 중복 제거 및 정렬 (긴 것 우선)
    core_terms = sorted(set(core_terms), key=len, reverse=True)
    
    return core_terms[:5]


def filter_utterances_by_keyword(
    dialogue: List[Dict],
    keyword: Optional[str],
    start_idx: int,
    end_idx: int,
    min_utterances: int = MIN_UTTERANCES
) -> List[Dict]:
    """
    keyword와 관련된 발화만 필터링합니다.
    """
    if not keyword:
        return dialogue[start_idx:end_idx]
    
    is_general = any(gk in keyword for gk in GENERAL_KEYWORDS) or len(keyword) < 5
    core_terms = extract_keyword_core_terms(keyword)
    
    filtered = []
    for i in range(start_idx, end_idx):
        utterance_obj = dialogue[i]
        utterance = utterance_obj.get('utterance', '')
        
        # 1순위: keyword가 정확히 포함된 발화
        if keyword in utterance:
            filtered.append(utterance_obj)
            continue
        
        # 2순위: 핵심 단어가 포함된 발화
        if core_terms and not is_general:
            if any(term in utterance for term in core_terms[:3]):
                filtered.append(utterance_obj)
                continue
        
        # 3순위: 숫자가 포함된 문장은 유지
        if re.search(r'\d', utterance):
            filtered.append(utterance_obj)
            continue
        
        # 4순위: 의사진행 관련 키워드가 있으면 유지
        if any(kw in utterance for kw in PROCEDURE_KEYWORDS[:8]):
            filtered.append(utterance_obj)
            continue
        
        # 5순위: keyword가 일반적인 경우
        if is_general:
            keyword_words = [w for w in keyword.split() if len(w) >= 2]
            if any(word in utterance for word in keyword_words):
                filtered.append(utterance_obj)
                continue
    
    # 필터링 결과가 너무 적으면 완화된 기준 적용
    if len(filtered) < min_utterances:
        logger.warning(f"keyword 필터링 결과가 부족 ({len(filtered)}개 < {min_utterances}개). 안건 경계 내 모든 발화 포함.")
        filtered = dialogue[start_idx:end_idx]
    
    return filtered


def has_important_number(utterance: str) -> bool:
    """
    발화에 요약에 중요한 숫자가 포함되어 있는지 확인합니다.
    """
    # 법조항 패턴
    if re.search(r'제\d+[조항호절]', utterance):
        return True
    
    # 날짜/연도 패턴
    if re.search(r'\d{4}년|\d{1,2}월\s*\d{1,2}일', utterance):
        return True
    
    # 금액 패턴
    if re.search(r'\d+[억만조]', utterance):
        return True
    
    # 의사일정 번호
    if re.search(r'의사일정\s*제\d+항', utterance):
        return True
    
    return False


def should_preserve_for_rouge(utterance: str) -> bool:
    """
    ROUGE-1 향상을 위해 보존해야 할 발화인지 판단합니다.
    """
    utterance_clean = utterance.strip()
    
    if re.search(r'의사일정\s*제\d+항', utterance_clean):
        return True
    
    if re.search(r'[가-힣]+법률안|[가-힣]+법\s*일부개정', utterance_clean):
        return True
    
    if re.search(r'제\d+[조항호절]', utterance_clean):
        return True
    
    if any(kw in utterance_clean for kw in ['의결', '가결', '부결', '결정', '처리', '상정']):
        return True
    
    if any(kw in utterance_clean for kw in ['전문위원', '수석전문위원', '보고', '검토', '의견']):
        if len(utterance_clean) > 20:
            return True
    
    return False


def clean_utterance(utterance: str) -> bool:
    """
    발화가 불필요한지 판단합니다.
    
    Returns:
        True: 제거해야 함, False: 유지해야 함
    """
    utterance_clean = utterance.strip()
    
    # 요약에 중요한 숫자가 포함된 문장은 유지
    if has_important_number(utterance_clean):
        return False
    
    # "...…" 포함 문장 처리
    if '…' in utterance_clean or '...' in utterance_clean:
        if len(utterance_clean) >= MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS:
            return False
        else:
            return True
    
    # 최소 길이 이하 발화 처리
    if len(utterance_clean) <= MIN_UTTERANCE_LENGTH:
        if any(keyword in utterance_clean for keyword in PROCEDURE_KEYWORDS[:8]):
            return False
        if re.search(r'제\d+[조항호절]', utterance_clean):
            return False
        return True
    
    # 단순 반응 표현 제거
    simple_responses = [
        r'^네\s*\.?\s*$', r'^예\s*\.?\s*$', r'^네네\s*\.?\s*$', r'^예예\s*\.?\s*$',
        r'^음\s*\.?\s*$', r'^아\s*\.?\s*$', r'^그렇죠\s*\.?\s*$',
        r'^맞습니다\s*\.?\s*$', r'^그렇습니다\s*\.?\s*$',
    ]
    
    for pattern in simple_responses:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 인사/예의 표현 제거
    greetings = [
        r'^감사합니다\s*\.?\s*$', r'^수고하셨습니다\s*\.?\s*$',
        r'^고생하셨습니다\s*\.?\s*$',
    ]
    
    for pattern in greetings:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 채우기성 표현 제거
    filler_expressions = [
        r'^말씀드렸잖아요\s*\.?\s*$',
        r'^아시다시피\s*\.?\s*$',
        r'^제가 볼 때는요\s*\.?\s*$',
    ]
    
    for pattern in filler_expressions:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # "보고드리겠습니다" 단독은 제거
    if re.match(r'^보고드리겠습니다\s*\.?\s*$', utterance_clean, re.IGNORECASE):
        return True
    
    # 잡담/소음 표현 제거
    noise_expressions = [
        r'^하하\s*\.?\s*$', r'^웃음\s*\.?\s*$',
        r'^잠시만요\s*\.?\s*$', r'^잠깐만요\s*\.?\s*$', r'^잠깐\s*\.?\s*$',
    ]
    
    for pattern in noise_expressions:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 의사 진행 발화는 항상 보존
    if any(keyword in utterance_clean for keyword in PROCEDURE_KEYWORDS):
        return False
    
    # 전문위원/수석전문위원 보고 시작 신호 보존
    report_start_patterns = [
        r'보고드리겠습니다',
        r'보고드리도록 하겠습니다',
        r'설명드리겠습니다',
    ]
    for pattern in report_start_patterns:
        if re.search(pattern, utterance_clean):
            return False
    
    # 책임 있는 발언은 보존
    important_keywords = [
        '정부', '위원', '의견', '쟁점', '근거', '반대', '찬성',
        '제안', '검토', '보고', '설명', '논의', '문제', '사항',
        '전문위원', '수석전문위원'
    ]
    
    if any(keyword in utterance_clean for keyword in important_keywords) and len(utterance_clean) > 5:
        return False
    
    return False

