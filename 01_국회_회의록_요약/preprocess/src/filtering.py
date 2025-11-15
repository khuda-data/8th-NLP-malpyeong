"""
발화 필터링 모듈

불필요한 발화를 제거하고 중요한 발화를 보존합니다.
- 중요한 숫자(법조항, 날짜, 금액 등)가 포함된 발화는 보존
- 의사진행 관련 키워드가 포함된 발화는 보존
- 단순 반응, 인사, 채우기성 표현은 제거
"""

import re
import logging

try:
    from .config import (
        PROCEDURE_KEYWORDS,
        MIN_UTTERANCE_LENGTH,
        MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS
    )
except ImportError:
    from config import (
        PROCEDURE_KEYWORDS,
        MIN_UTTERANCE_LENGTH,
        MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS
    )

logger = logging.getLogger(__name__)


def has_important_number(utterance: str) -> bool:
    """
    발화에 요약에 중요한 숫자가 포함되어 있는지 확인합니다.
    
    확인하는 패턴:
    - 법조항: "제1조", "제2항" 등
    - 날짜/연도: "2024년", "1월 1일" 등
    - 금액: "1억", "100만원" 등
    - 의사일정 번호: "의사일정 제1항" 등
    
    Args:
        utterance: 확인할 발화 텍스트
    
    Returns:
        True: 중요한 숫자가 포함됨, False: 포함되지 않음
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


def clean_utterance(utterance: str) -> bool:
    """
    발화가 불필요한지 판단합니다.
    
    중요한 키워드가 포함된 발화는 길이와 무관하게 보존합니다.
    이를 통해 유의미한 짧은 발화가 제거되는 것을 방지합니다.
    
    Returns:
        True: 제거해야 함, False: 유지해야 함
    """
    utterance_clean = utterance.strip()
    
    # 요약에 중요한 숫자가 포함된 문장은 유지
    if has_important_number(utterance_clean):
        return False
    
    # 의사 진행 발화는 항상 보존 (길이와 무관)
    if any(keyword in utterance_clean for keyword in PROCEDURE_KEYWORDS):
        return False
    
    # 전문위원/수석전문위원 보고 시작 신호 보존 (길이와 무관)
    report_start_patterns = [
        r'보고드리겠습니다',
        r'보고드리도록 하겠습니다',
        r'설명드리겠습니다',
    ]
    for pattern in report_start_patterns:
        if re.search(pattern, utterance_clean):
            return False
    
    # 전문위원/수석전문위원 언급 보존 (길이와 무관)
    if '전문위원' in utterance_clean or '수석전문위원' in utterance_clean:
        return False
    
    # 정부, 찬성, 반대, 의견 등 중요한 키워드 보존 (길이와 무관)
    important_keywords = [
        '정부', '위원', '의견', '쟁점', '근거', '반대', '찬성',
        '제안', '검토', '보고', '설명', '논의', '문제', '사항'
    ]
    if any(keyword in utterance_clean for keyword in important_keywords):
        return False
    
    # "...…" 포함 문장 처리
    if '…' in utterance_clean or '...' in utterance_clean:
        if len(utterance_clean) >= MIN_UTTERANCE_LENGTH_WITH_ELLIPSIS:
            return False
        else:
            return True
    
    # 최소 길이 이하 발화 처리 (중요한 키워드는 이미 위에서 처리됨)
    if len(utterance_clean) <= MIN_UTTERANCE_LENGTH:
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
    
    # "보고드리겠습니다" 단독은 제거 (위에서 보고 시작 신호는 이미 처리됨)
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
    
    # 위에서 처리하지 못한 긴 발화는 유지
    return False

