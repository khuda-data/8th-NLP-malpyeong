"""
안건 경계 탐지 모듈
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .config import (
        PROCLAMATION_PATTERNS, FALLBACK_AGENDA_LENGTH, MAX_AGENDA_LENGTH
    )
except ImportError:
    from config import (
        PROCLAMATION_PATTERNS, FALLBACK_AGENDA_LENGTH, MAX_AGENDA_LENGTH
    )

logger = logging.getLogger(__name__)


def find_next_agenda_start(
    dialogue: List[Dict],
    start_idx: int
) -> Optional[int]:
    """
    다음 안건이 시작되는 위치를 찾는 함수입니다.
    
    예를 들어, 지금 안건이 "의사일정 제1항"이라면
    "의사일정 제2항"이 나오는 위치를 찾는 것입니다.
    
    어떻게 찾나요?
    1. "의사일정 제N항"이라는 말이 나오고
    2. "상정"이라는 말이 있으면 → 무조건 다음 안건!
    3. "상정"이 없어도 → "의결", "가결", "부결" 같은 결정 관련 말이 없으면 → 다음 안건!
    
    Args:
        dialogue: 전체 대화 내용 (발화 리스트)
        start_idx: 지금 안건이 시작된 위치
    
    Returns:
        다음 안건이 시작되는 위치 (없으면 None)
    """
    next_agenda_pattern = r'의사일정\s*제\d+항'
    exclude_keywords = r'의결|가결|부결|결론|회부|부터|까지|내지'
    
    for i in range(start_idx + 1, len(dialogue)):
        utterance = dialogue[i].get('utterance', '')
        if re.search(next_agenda_pattern, utterance):
            # 상정 포함이면 무조건 인정
            if re.search(r'상정', utterance):
                return i
            # 제외 키워드 없으면 인정
            if not re.search(exclude_keywords, utterance):
                return i
    
    return None


def find_agenda_end(
    dialogue: List[Dict],
    start_idx: int
) -> Optional[int]:
    """
    안건이 끝났다는 말을 찾는 함수입니다 (회의 종료와 구분).
    
    안건별 종료 패턴:
    - "이상 보고를 마치겠습니다" → 안건 보고 끝!
    - "이상 보고 마치겠습니다" → 안건 보고 끝!
    - "심사를 마쳤습니다" → 안건 심사 끝!
    - "이상으로 마치겠습니다" → 안건 마무리!
    
    Args:
        dialogue: 전체 대화 내용 (발화 리스트)
        start_idx: 지금 안건이 시작된 위치
    
    Returns:
        안건 종료 문장이 나온 위치 (없으면 None)
    """
    agenda_end_patterns = [
        r'이상\s*(?:보고를|보고)\s*마치겠습니다',  # "이상 보고를 마치겠습니다"
        r'이상\s*보고를\s*마치겠습니다',  # "이상 보고를 마치겠습니다" (공백 없음)
        r'이상\s*보고\s*마치겠습니다',  # "이상 보고 마치겠습니다"
        r'보고를\s*마치겠습니다',  # "보고를 마치겠습니다"
        r'보고\s*마치겠습니다',  # "보고 마치겠습니다"
        r'심사를\s*마쳤습니다',  # "심사를 마쳤습니다"
        r'심의를\s*마치겠습니다',  # "심의를 마치겠습니다" (예시 4에서 발견)
        r'이상으로\s*(?:.*\s*)?마치겠습니다',  # "이상으로 마치겠습니다"
        r'이상\s*(?:.*\s*)?마치겠습니다',  # "이상 마치겠습니다"
    ]
    
    for i in range(start_idx + 1, len(dialogue)):
        utterance = dialogue[i].get('utterance', '')
        # 회의 종료 패턴은 제외 (산회, 회의 종료 등)
        if re.search(r'산회|회의.*종료|회의.*마치', utterance):
            continue
        
        # 안건별 종료 패턴 확인 (더 유연한 매칭)
        # 발화에 패턴이 포함되어 있으면 안건 종료로 판단
        for pattern in agenda_end_patterns:
            if re.search(pattern, utterance):
                return i
        
        # 추가: "보고를 마치겠습니다" 같은 패턴이 발화 중간에 있어도 매칭
        # (예: "이상으로 보고를 마치겠습니다" 같은 변형)
        if re.search(r'보고.*마치겠습니다|마치겠습니다.*보고', utterance):
            return i
    
    return None


def find_meeting_end(
    dialogue: List[Dict],
    start_idx: int
) -> Optional[int]:
    """
    회의가 끝났다는 말을 찾는 함수입니다.
    
    예를 들어:
    - "산회하도록 하겠습니다" → 회의 끝!
    - "오늘 회의는 여기서 마치겠습니다" → 회의 끝!
    - "끝냅시다" → 회의 끝!
    
    이런 말들이 나오면 그 위치를 알려줍니다.
    
    Args:
        dialogue: 전체 대화 내용 (발화 리스트)
        start_idx: 지금 안건이 시작된 위치
    
    Returns:
        회의 종료 문장이 나온 위치 (없으면 None)
    """
    end_patterns = [
        r'산회',
        r'종료',
        r'회의.*종료',
        r'회의.*마치',
        r'회의.*끝',
        r'이것으로.*끝',
        r'이것으로.*마치',
        r'이것으로.*종료'
    ]
    
    for i in range(start_idx + 1, len(dialogue)):
        utterance = dialogue[i].get('utterance', '')
        if any(re.search(pattern, utterance) for pattern in end_patterns):
            return i
    
    return None


def find_proclamation_sentences(
    dialogue: List[Dict],
    exclude_adjournment: bool = True
) -> List[int]:
    """
    결정을 선포하는 문장을 찾는 함수입니다.
    
    결정 발화는 안건의 최종 결정을 나타내는 발화입니다.
    하지만 모든 안건에 결정 발화가 있는 것은 아닙니다 (36.5%만 해당).
    
    우선순위 1: 명확한 선포 패턴 (가장 확실)
    - "가결되었음을 선포합니다" → 가결 결정!
    - "부결되었음을 선포합니다" → 부결 결정!
    - "의결되었음을 선포합니다" → 의결 결정!
    
    우선순위 2: 완료 표현
    - "가결되었습니다" → 가결 완료!
    - "부결되었습니다" → 부결 완료!
    - "의결되었습니다" → 의결 완료!
    - "통과되었습니다" → 통과 완료!
    
    우선순위 3: 유보/회부 결정
    - "유보하도록 하겠습니다" → 유보 결정!
    - "상임위로 회부" → 상임위 회부 결정!
    - "종결하고 의결" → 종결 의결!
    
    제외 조건:
    - "이의 없습니까?" 같은 질문은 제외 (질문은 결정이 아님)
    - "산회를 선포합니다"는 회의 종료이므로 제외
    
    Args:
        dialogue: 전체 대화 내용 (발화 리스트)
        exclude_adjournment: True면 "산회 선포"는 제외 (기본값: True)
    
    Returns:
        결정 선포 문장이 나온 위치들의 리스트
    """
    proclamation_indices = []
    for i, utterance_obj in enumerate(dialogue):
        utterance = utterance_obj.get('utterance', '')
        
        # 산회 선포 제외 (회의 종료이므로)
        if exclude_adjournment and '산회' in utterance:
            continue
        
        # 질문 패턴 제외 (질문은 결정이 아님)
        if re.search(r'이의\s*없습니까|이의\s*없으시지요', utterance):
            continue
        
        # 안건 종료 패턴 제외 (안건 종료는 결정이 아님, 3.5순위에서 별도 처리)
        # "이상 보고를 마치겠습니다", "보고를 마치겠습니다" 등은 안건 종료이지 결정이 아님
        if re.search(r'이상\s*(?:보고를|보고)\s*마치겠습니다|보고를\s*마치겠습니다|보고\s*마치겠습니다|심사를\s*마쳤습니다|심의를\s*마치겠습니다', utterance):
            continue
        
        # 결정 패턴 확인
        for pattern in PROCLAMATION_PATTERNS:
            if re.search(pattern, utterance):
                proclamation_indices.append(i)
                break
    
    return proclamation_indices


def extract_agenda_boundaries(
    dialogue: List[Dict],
    sentence_id: Optional[str] = None,
    next_sentence_id: Optional[str] = None
) -> Tuple[int, int]:
    """
    하나의 안건이 어디서 시작하고 어디서 끝나는지 찾는 함수입니다.
    
    예를 들어, 전체 대화가 100개 발화가 있다면:
    - 안건 1: 발화 10번부터 50번까지
    - 안건 2: 발화 50번부터 80번까지
    - 안건 3: 발화 80번부터 100번까지
    
    이렇게 안건의 시작점과 끝점을 찾아줍니다.
    
    시작점 찾기:
    - sentence_id라는 고유 번호로 시작 발화를 찾습니다.
    
    끝점 찾기 (우선순위 순서대로):
    1순위: 같은 회의에 다음 안건이 있으면 → 그 다음 안건의 시작점이 지금 안건의 끝점!
    
    2순위: "의사일정 제N항"이라는 말이 나오면 → 다음 안건이 시작되는 곳!
           (예: "의사일정 제2항을 상정합니다")
    
    3순위: "가결되었음을 선포합니다" 같은 결정 선포 문장이 나오면 → 그게 끝점!
    
    3.5순위: "이상 보고를 마치겠습니다" 같은 안건별 종료 문장이 나오면 → 그게 끝점!
            (회의 종료와 구분하여 안건만 종료되는 경우 처리)
    
    4순위: "산회하도록 하겠습니다" 같은 회의 종료 문장이 나오면 → 그게 끝점!
    
    5순위: 위 방법들로 못 찾으면 → 기본값으로 50개 발화만 사용
    
    Args:
        dialogue: 전체 대화 내용 (발화 리스트)
        sentence_id: 안건이 시작되는 발화의 고유 번호
        next_sentence_id: 같은 회의에서 다음 안건의 시작 발화 고유 번호 (있으면 최우선 사용!)
    
    Returns:
        (시작점, 끝점) 튜플
        예: (10, 50) → 발화 10번부터 50번까지가 이 안건
    """
    # 1단계: 시작점 찾기
    # sentence_id라는 고유 번호로 안건이 시작되는 발화를 찾습니다.
    start_idx = 0
    if sentence_id:
        for i, utterance_obj in enumerate(dialogue):
            if utterance_obj.get('id') == sentence_id:
                start_idx = i  # 찾았습니다! 이게 시작점!
                break
        else:
            # 못 찾았으면 경고 메시지 출력하고 처음부터 시작
            logger.warning(f"sentence_id '{sentence_id}'를 찾을 수 없음. start_idx=0으로 설정.")
    else:
        logger.warning(f"sentence_id가 없음. start_idx=0으로 설정.")
    
    # 일단 끝점은 전체 대화의 마지막으로 설정 (나중에 더 정확한 위치로 바꿀 예정)
    end_idx = len(dialogue)
    
    # 2단계: 끝점 찾기 (우선순위 순서대로 시도)
    
    # 1순위: 같은 회의에 다음 안건이 있으면 → 그게 가장 정확!
    if next_sentence_id:
        for i, utterance_obj in enumerate(dialogue):
            if utterance_obj.get('id') == next_sentence_id:
                end_idx = i  # 다음 안건이 시작되는 곳이 지금 안건의 끝!
                logger.debug(f"다음 샘플의 sentence_id로 end_idx 찾음: {end_idx}")
                return start_idx, end_idx  # 찾았으니 바로 반환!
    
    # 2순위: "의사일정 제N항" 패턴 찾기
    next_agenda_idx = find_next_agenda_start(dialogue, start_idx)
    if next_agenda_idx:
        end_idx = next_agenda_idx  # 다음 안건이 시작되는 곳!
        logger.debug(f"다음 안건 패턴으로 end_idx 찾음: {end_idx}")
    
    # 3순위: 결정 선포 문장 찾기 ("가결되었음을 선포합니다" 등)
    if end_idx == len(dialogue):  # 아직 끝점을 못 찾았으면
        proclamation_indices = find_proclamation_sentences(dialogue, exclude_adjournment=True)
        for proc_idx in proclamation_indices:
            if proc_idx > start_idx:  # 시작점보다 뒤에 있어야 함
                end_idx = proc_idx + 1  # 선포 문장 다음이 끝점!
                logger.debug(f"선포 문장 발견: {proc_idx}")
                break
    
    # 3.5순위: 안건별 종료 문장 찾기 ("이상 보고를 마치겠습니다" 등)
    # 회의 종료와 구분하여 안건만 종료되는 경우 처리
    agenda_end_used_flag = False  # agenda_end가 사용되었는지 추적 (MAX_AGENDA_LENGTH 제한 방지용)
    if end_idx == len(dialogue):  # 아직 끝점을 못 찾았으면
        agenda_end_idx = find_agenda_end(dialogue, start_idx)
        if agenda_end_idx:
            fallback_limit = start_idx + FALLBACK_AGENDA_LENGTH
            # agenda_end_idx가 Fallback 범위 내에 있으면 사용 (더 정확한 경계)
            if agenda_end_idx <= fallback_limit:
                end_idx = agenda_end_idx + 1
                agenda_end_used_flag = True
                logger.debug(f"안건 종료 문장 발견: {agenda_end_idx} (start_idx={start_idx}, fallback_limit={fallback_limit})")
            else:
                # 너무 멀리 있으면 무시 (다른 안건의 종료일 수 있음)
                logger.debug(f"안건 종료 문장 발견했지만 너무 멀어서 무시: {agenda_end_idx} (start_idx={start_idx}, fallback_limit={fallback_limit})")
        else:
            logger.debug(f"안건 종료 문장을 찾지 못함 (start_idx={start_idx})")
    
    # 4순위: 회의 종료 문장 찾기 ("산회하도록 하겠습니다" 등)
    if end_idx == len(dialogue):  # 아직 끝점을 못 찾았으면
        meeting_end_idx = find_meeting_end(dialogue, start_idx)
        if meeting_end_idx:
            end_idx = meeting_end_idx + 1  # 회의 종료 문장 다음이 끝점!
            logger.debug(f"회의 종료 문장 발견: {meeting_end_idx}")
    
    # 5순위: 위 방법들로 못 찾으면 → 기본값으로 50개 발화만 사용
    if end_idx == len(dialogue):
        logger.warning(f"안건 경계를 찾지 못함. sentence_id 이후 {FALLBACK_AGENDA_LENGTH}개 발화 사용.")
        end_idx = min(start_idx + FALLBACK_AGENDA_LENGTH, len(dialogue))
        # Fallback 사용 전에 한 번 더 agenda_end 체크 (3.5순위에서 놓친 경우 대비)
        if not agenda_end_used_flag:
            agenda_end_idx = find_agenda_end(dialogue, start_idx)
            if agenda_end_idx and agenda_end_idx < end_idx:
                end_idx = agenda_end_idx + 1
                agenda_end_used_flag = True
                logger.debug(f"Fallback 전에 안건 종료 문장 발견: {agenda_end_idx}")
    
    # 안전장치: 끝점이 시작점보다 앞에 있으면 안 됨!
    if end_idx <= start_idx:
        logger.warning(f"end_idx({end_idx})가 start_idx({start_idx})보다 작거나 같음. 수정 중...")
        end_idx = min(start_idx + FALLBACK_AGENDA_LENGTH, len(dialogue))
        if end_idx <= start_idx:
            end_idx = len(dialogue)  # 그래도 안 되면 전체 대화 끝까지
    
    # 안건이 너무 길면 최대 100개 발화로 제한
    # 단, agenda_end나 meeting_end로 찾은 경우는 제한하지 않음 (정확한 경계이므로)
    if end_idx - start_idx > MAX_AGENDA_LENGTH:
        # meeting_end로 찾았는지 확인
        meeting_end_used = False
        if end_idx < len(dialogue):
            meeting_end_idx = find_meeting_end(dialogue, start_idx)
            if meeting_end_idx and end_idx == meeting_end_idx + 1:
                meeting_end_used = True
        
        # agenda_end나 meeting_end로 찾지 않았으면 MAX_AGENDA_LENGTH 제한 적용
        # 단, 제한 적용 전에 한 번 더 agenda_end 체크 (놓친 경우 대비)
        if not agenda_end_used_flag and not meeting_end_used:
            agenda_end_idx = find_agenda_end(dialogue, start_idx)
            if agenda_end_idx and agenda_end_idx < end_idx:
                end_idx = agenda_end_idx + 1
                agenda_end_used_flag = True
                logger.debug(f"MAX_AGENDA_LENGTH 제한 전에 안건 종료 문장 발견: {agenda_end_idx}")
            else:
                end_idx = min(start_idx + MAX_AGENDA_LENGTH, len(dialogue))
                logger.debug(f"MAX_AGENDA_LENGTH 제한으로 end_idx 조정: {end_idx}")
    
    # 마지막으로 대화 길이를 넘지 않도록 확인
    end_idx = min(end_idx, len(dialogue))
    
    return start_idx, end_idx

