"""
안건 경계 탐지 모듈
"""

import re
import os
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .config import (
        NEXT_AGENDA_PATTERNS, PROCLAMATION_PATTERNS, FALLBACK_AGENDA_LENGTH,
        MAX_AGENDA_LENGTH, DECISION_KEYWORDS, FINAL_DECISION_PATTERNS,
        INTERMEDIATE_DECISION_PATTERNS, QUESTION_PATTERNS, SUPPLEMENTARY_PATTERNS
    )
except ImportError:
    from config import (
        NEXT_AGENDA_PATTERNS, PROCLAMATION_PATTERNS, FALLBACK_AGENDA_LENGTH,
        MAX_AGENDA_LENGTH, DECISION_KEYWORDS, FINAL_DECISION_PATTERNS,
        INTERMEDIATE_DECISION_PATTERNS, QUESTION_PATTERNS, SUPPLEMENTARY_PATTERNS
    )

logger = logging.getLogger(__name__)


def find_next_agenda_start(
    dialogue: List[Dict],
    start_idx: int,
    current_keyword: Optional[str] = None
) -> Optional[int]:
    """
    다음 안건의 시작점을 찾습니다.
    
    Args:
        dialogue: 대화 리스트
        start_idx: 현재 안건 시작점
        current_keyword: 현재 안건 키워드 (제외용)
    
    Returns:
        다음 안건 시작점 인덱스 (없으면 None)
    """
    for i in range(start_idx + 1, len(dialogue)):
        utterance = dialogue[i].get('utterance', '')
        for pattern in NEXT_AGENDA_PATTERNS:
            if re.search(pattern, utterance):
                # 현재 안건 키워드가 포함되지 않았는지 확인
                if current_keyword and current_keyword in utterance:
                    continue
                # "다음 제N항부터 제M항까지" 같은 패턴은 현재 안건의 연속이므로 제외
                if re.search(r'다음\s*제\d+항\s*부터', utterance):
                    continue
                # "다음 의사일정 제N항" 패턴이 "의결", "가결", "결론", "회부" 등과 함께 있으면 현재 안건의 결정이므로 제외
                if re.search(r'다음.*의사일정.*제\d+항', utterance):
                    if re.search(r'의결|가결|부결|결론|회부|의결코자|가결하고자|부결하고자', utterance):
                        continue
                # "의사일정 제N항을 상정합니다" 같은 패턴만 다음 안건으로 인정
                if re.search(r'의사일정\s*제\d+항.*상정', utterance):
                    return i
                # "의사일정 제N항" 패턴이 있고 "상정"이 없으면 현재 안건의 연속일 수 있으므로 확인
                if re.search(r'의사일정\s*제\d+항', utterance) and not re.search(r'상정', utterance):
                    # "다음"이 앞에 있고 "의결", "가결", "결론" 등이 없으면 다음 안건으로 인정
                    if re.search(r'다음.*의사일정\s*제\d+항', utterance):
                        if not re.search(r'의결|가결|부결|결론|회부|의결코자|가결하고자|부결하고자', utterance):
                            return i
                        else:
                            continue
                    # "의사일정 제N항"만 있고 "부터", "까지" 같은 연속 표현이 없으면 다음 안건으로 인정
                    if not re.search(r'부터|까지|내지', utterance):
                        return i
                # 기타 패턴은 다음 안건으로 인정
                return i
    
    return None


def find_proclamation_sentences(
    dialogue: List[Dict],
    exclude_adjournment: bool = True
) -> List[int]:
    """
    선포 문장의 인덱스를 찾습니다.
    
    Args:
        dialogue: 대화 리스트
        exclude_adjournment: 산회 선포 제외 여부
    
    Returns:
        선포 문장 인덱스 리스트
    """
    proclamation_indices = []
    for i, utterance_obj in enumerate(dialogue):
        utterance = utterance_obj.get('utterance', '')
        
        # 산회 선포 제외
        if exclude_adjournment and '산회' in utterance:
            continue
        
        for pattern in PROCLAMATION_PATTERNS:
            if re.search(pattern, utterance):
                proclamation_indices.append(i)
                break
    
    return proclamation_indices


def find_decision_with_llm(
    dialogue_segment: List[Dict],
    start_idx: int,
    end_idx: int
) -> Optional[int]:
    """
    LLM을 사용하여 결정 발화를 찾습니다.
    
    Args:
        dialogue_segment: 대화 세그먼트 (start_idx부터 end_idx까지)
        start_idx: 시작 인덱스 (원본 dialogue 기준)
        end_idx: 종료 인덱스 (원본 dialogue 기준)
    
    Returns:
        결정 발화의 인덱스 (원본 dialogue 기준, 없으면 None)
    """
    try:
        # OpenAI API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY가 설정되지 않음. 룰베이스로 대체합니다.")
            return None
        
        # OpenAI 클라이언트 초기화
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except ImportError:
            logger.warning("openai 패키지가 설치되지 않음. 룰베이스로 대체합니다.")
            return None
        
        # 발화 텍스트 준비 (토큰 제한 고려: 마지막 100개 발화만 포함)
        max_utterances = 100
        if len(dialogue_segment) > max_utterances:
            dialogue_segment = dialogue_segment[-max_utterances:]
            start_idx = end_idx - max_utterances
        
        utterances_text = "\n".join([
            f"[{i+start_idx}] [{utt.get('role', '비지정')}] {utt.get('speaker', '')}: {utt.get('utterance', '')[:300]}"
            for i, utt in enumerate(dialogue_segment)
        ])
        
        # 프롬프트 구성
        system_prompt = """국회 회의록에서 안건 결정 발화를 찾아주세요.
규칙:
1. 소위원장 또는 위원장이 내린 결정만 인정
2. 질문("이의 없습니까?" 등)은 결정이 아님
3. 결정 패턴: "가결되었음을 선포", "부결되었습니다", "상임위로 다시 회부", "유보하도록 하겠습니다", "종결 짓겠습니다" 등
4. 여러 결정이 있으면 가장 마지막 결정의 인덱스를 반환
5. 결정이 없으면 -1 반환
응답 형식: 숫자만 (예: 123 또는 -1)"""

        user_prompt = f"""다음 발화 목록에서 안건에 대한 결정 발화를 찾아주세요. 가장 마지막 결정 발화의 인덱스를 반환하세요.

{utterances_text}

결정 발화 인덱스 (없으면 -1):"""
        
        # OpenAI API 호출
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=20
        )
        
        # 응답 파싱
        response_text = response.choices[0].message.content.strip()
        match = re.search(r'-?\d+', response_text)
        if match:
            decision_idx = int(match.group())
            
            if decision_idx == -1:
                return None
            
            # 인덱스 범위 검증
            if start_idx <= decision_idx < end_idx:
                logger.info(f"LLM이 결정 발화 인덱스 {decision_idx} 발견 (범위: [{start_idx}, {end_idx}))")
                return decision_idx
            elif 0 <= decision_idx < len(dialogue_segment):
                adjusted_idx = start_idx + decision_idx
                if start_idx <= adjusted_idx < end_idx:
                    logger.info(f"LLM이 결정 발화 인덱스 {decision_idx} (dialogue_segment 기준) -> {adjusted_idx} (원본 dialogue 기준) 발견")
                    return adjusted_idx
                else:
                    logger.warning(f"인덱스 변환 후 범위 벗어남: {decision_idx} -> {adjusted_idx} (범위: [{start_idx}, {end_idx}))")
                    return None
            else:
                logger.warning(f"LLM이 반환한 인덱스 {decision_idx}가 범위를 벗어남 (범위: [{start_idx}, {end_idx}))")
                return None
        else:
            logger.warning(f"LLM 응답에서 숫자를 찾을 수 없음: {response_text}")
            return None
            
    except Exception as e:
        logger.warning(f"LLM을 사용한 결정 탐색 실패: {e}")
        return None


def _find_decisions_in_range(
    dialogue: List[Dict],
    start_idx: int,
    end_idx: int
) -> List[Tuple[int, int]]:
    """
    지정된 범위에서 결정 후보를 찾습니다.
    
    Returns:
        (인덱스, 우선순위) 튜플 리스트
    """
    decision_candidates = []
    question_indices = []
    
    for i in range(start_idx, end_idx):
        utterance = dialogue[i].get('utterance', '')
        
        # 질문 패턴 제외
        is_question = any(re.search(pattern, utterance) for pattern in QUESTION_PATTERNS)
        if is_question:
            question_indices.append(i)
            continue
        
        if any(kw in utterance for kw in DECISION_KEYWORDS):
            # 1순위 패턴 확인
            for pattern in FINAL_DECISION_PATTERNS:
                if re.search(pattern, utterance):
                    decision_candidates.append((i, 1))
                    break
            else:
                # 2순위 패턴 확인
                for pattern in INTERMEDIATE_DECISION_PATTERNS:
                    if re.search(pattern, utterance):
                        decision_candidates.append((i, 2))
                        break
    
    # 질문 다음의 결정 찾기
    if question_indices:
        for question_idx in reversed(question_indices):
            if question_idx >= end_idx:
                continue
            for j in range(question_idx + 1, min(question_idx + 3, end_idx)):
                next_utterance = dialogue[j].get('utterance', '')
                if any(kw in next_utterance for kw in DECISION_KEYWORDS):
                    for pattern in FINAL_DECISION_PATTERNS:
                        if re.search(pattern, next_utterance):
                            decision_candidates.append((j, 1))
                            break
                    if any(c[0] == j and c[1] == 1 for c in decision_candidates):
                        break
                if any(c[0] > question_idx and c[1] == 1 for c in decision_candidates):
                    break
    
    return decision_candidates


def extract_agenda_boundaries(
    dialogue: List[Dict],
    sentence_id: Optional[str] = None,
    keyword: Optional[str] = None,
    begin: Optional[int] = None,
    end: Optional[int] = None,
    role_map: Optional[Dict[str, str]] = None
) -> Tuple[int, int]:
    """
    안건의 시작과 끝 인덱스를 찾습니다.
    
    우선순위:
    1. 다음 안건 시작점 ("의사일정 제N항" 패턴)
    2. 선포 문장 ("가결되었음을 선포합니다", 산회 제외)
    3. 범위 제한 (FALLBACK_AGENDA_LENGTH)
    
    Args:
        dialogue: 대화 리스트
        sentence_id: 안건 시작점 ID
        keyword: 안건 키워드
        begin: 시작 인덱스 (무시됨)
        end: 종료 인덱스 (무시됨)
        role_map: 발화자 역할 매핑
    
    Returns:
        (start_idx, end_idx) 튜플
    """
    # sentence_id로 시작점 찾기
    start_idx = 0
    if sentence_id:
        for i, utterance_obj in enumerate(dialogue):
            if utterance_obj.get('id') == sentence_id:
                start_idx = i
                break
        else:
            logger.warning(f"sentence_id '{sentence_id}'를 찾을 수 없음. keyword로 시작점 탐색.")
            if keyword:
                for i, utterance_obj in enumerate(dialogue):
                    if keyword in utterance_obj.get('utterance', ''):
                        start_idx = i
                        break
    elif keyword:
        for i, utterance_obj in enumerate(dialogue):
            if keyword in utterance_obj.get('utterance', ''):
                start_idx = i
                break
    
    # sentence_id가 없고 keyword로 찾은 경우에만 이전 맥락 포함
    original_start_idx = start_idx
    if start_idx > 0 and not sentence_id:
        prev_proclamation_indices = find_proclamation_sentences(dialogue[:start_idx], exclude_adjournment=True)
        if prev_proclamation_indices:
            prev_proc_idx = prev_proclamation_indices[-1]
            context_start = prev_proc_idx + 1
            if context_start < start_idx:
                start_idx = context_start
                logger.debug(f"이전 선포 문장({prev_proc_idx}) 이후 맥락 포함: start_idx 조정 {original_start_idx} -> {start_idx}")
    
    end_idx = len(dialogue)
    
    # 1순위: 다음 안건 시작점 찾기
    next_agenda_idx = find_next_agenda_start(dialogue, start_idx, keyword)
    if next_agenda_idx:
        # 결정은 반드시 next_agenda_idx 이전에 있어야 함
        search_end = next_agenda_idx
        decision_candidates = _find_decisions_in_range(
            dialogue,
            max(start_idx, next_agenda_idx - 200),
            search_end
        )
        
        if decision_candidates:
            priority_1_decisions = [c for c in decision_candidates if c[1] == 1]
            if priority_1_decisions:
                decision_idx = max(priority_1_decisions, key=lambda x: x[0])[0]
            else:
                decision_idx = max(decision_candidates, key=lambda x: x[0])[0]
            
            if decision_idx >= next_agenda_idx:
                end_idx = next_agenda_idx
                logger.warning(f"결정({decision_idx})이 다음 안건 시작점({next_agenda_idx}) 이후에 있어 제외")
            else:
                # 결정 이후 부연 설명 확인
                has_supplementary = False
                for i in range(decision_idx + 1, min(decision_idx + 3, next_agenda_idx)):
                    utterance = dialogue[i].get('utterance', '')
                    if any(re.search(pattern, utterance) for pattern in SUPPLEMENTARY_PATTERNS):
                        has_supplementary = True
                        break
                
                end_idx = decision_idx + 1
                if has_supplementary:
                    logger.debug(f"결정({decision_idx}) 이후 부연 설명 발견")
        else:
            # LLM으로 결정 찾기
            dialogue_segment = []
            for i in range(start_idx, next_agenda_idx):
                utt = dialogue[i].copy()
                speaker_raw = utt.get('speaker', '')
                if role_map is not None:
                    role = role_map.get(speaker_raw, '비지정')
                    utt['role'] = role
                else:
                    utt['role'] = '비지정'
                dialogue_segment.append(utt)
            
            llm_decision_idx = find_decision_with_llm(dialogue_segment, start_idx, next_agenda_idx)
            
            if llm_decision_idx is not None:
                end_idx = llm_decision_idx + 1
                logger.info(f"LLM이 결정({llm_decision_idx}) 발견: end_idx 조정 {next_agenda_idx} -> {end_idx}")
            else:
                end_idx = next_agenda_idx
                logger.debug(f"룰베이스 및 LLM으로 결정을 찾지 못함: end_idx = {next_agenda_idx}")
    
    # 2순위: 다음 안건 시작점이 없을 때 결정 찾기
    if end_idx == len(dialogue):
        decision_candidates = _find_decisions_in_range(dialogue, start_idx, len(dialogue))
        
        if decision_candidates:
            priority_1_decisions = [c for c in decision_candidates if c[1] == 1]
            if priority_1_decisions:
                decision_idx = max(priority_1_decisions, key=lambda x: x[0])[0]
            else:
                decision_idx = max(decision_candidates, key=lambda x: x[0])[0]
            end_idx = decision_idx + 1
            logger.debug(f"결정({decision_idx}) 발견 (다음 안건 시작점 없음)")
    
    # 3순위: 선포 문장 찾기
    if end_idx == len(dialogue):
        proclamation_indices = find_proclamation_sentences(dialogue, exclude_adjournment=True)
        for proc_idx in proclamation_indices:
            if proc_idx > start_idx:
                end_idx = proc_idx + 1
                logger.debug(f"선포 문장 발견: {proc_idx}")
                break
    
    # 4순위: 범위 제한
    if end_idx == len(dialogue):
        if len(dialogue) > 0:
            last_utterance = dialogue[-1].get('utterance', '')
            if '산회' in last_utterance and '선포' in last_utterance:
                end_idx = len(dialogue) - 1
                logger.debug(f"'산회를 선포합니다' 발견: end_idx 조정")
        
        if end_idx == len(dialogue):
            logger.warning(f"다음 안건 시작점과 선포 문장을 찾지 못함. sentence_id 이후 {FALLBACK_AGENDA_LENGTH}개 발화 사용.")
            end_idx = min(start_idx + FALLBACK_AGENDA_LENGTH, len(dialogue))
    
    # 안전장치
    if end_idx <= start_idx:
        logger.warning(f"end_idx({end_idx})가 start_idx({start_idx})보다 작거나 같음. 수정 중...")
        end_idx = min(start_idx + FALLBACK_AGENDA_LENGTH, len(dialogue))
        if end_idx <= start_idx:
            end_idx = len(dialogue)
    
    # MAX_AGENDA_LENGTH 제한 (결정이 있으면 완화)
    if end_idx - start_idx > MAX_AGENDA_LENGTH:
        decision_candidates_check = _find_decisions_in_range(
            dialogue,
            start_idx,
            min(start_idx + MAX_AGENDA_LENGTH + 50, len(dialogue))
        )
        
        if decision_candidates_check:
            priority_1_decisions = [c for c in decision_candidates_check if c[1] == 1]
            if priority_1_decisions:
                last_decision_idx = max(priority_1_decisions, key=lambda x: x[0])[0]
            else:
                last_decision_idx = max(decision_candidates_check, key=lambda x: x[0])[0]
            
            if last_decision_idx is not None:
                end_idx = min(last_decision_idx + 1, len(dialogue))
                logger.debug(f"결정 발견으로 MAX_AGENDA_LENGTH 제한 완화: 결정 위치 {last_decision_idx}")
            else:
                end_idx = min(start_idx + MAX_AGENDA_LENGTH, len(dialogue))
        else:
            end_idx = min(start_idx + MAX_AGENDA_LENGTH, len(dialogue))
    
    end_idx = min(end_idx, len(dialogue))
    
    return start_idx, end_idx

