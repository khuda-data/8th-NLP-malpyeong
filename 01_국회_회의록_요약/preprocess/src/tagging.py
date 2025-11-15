"""
국회 회의록 발화에 XML 태그를 적용하는 모듈

주요 기능:
- <결정>: 회의 결정사항 (가결, 부결 등)
- <안건>: 논의 중인 법안명
- <쟁점>: 논란/문제점
- <IMP>: 요약에 중요한 발화
"""

import re
import logging
from typing import Dict, List, Optional

try:
    from .config import (
        GOVERNMENT_ORGS, PROCEDURE_KEYWORDS, MAX_AGENDA_TAGS,
        SIMPLE_PROCEDURE_PATTERNS, SIMPLE_CHAIR_PATTERNS, SIMPLE_END_PATTERNS,
        QUESTION_PATTERNS
    )
except ImportError:
    from config import (
        GOVERNMENT_ORGS, PROCEDURE_KEYWORDS, MAX_AGENDA_TAGS,
        SIMPLE_PROCEDURE_PATTERNS, SIMPLE_CHAIR_PATTERNS, SIMPLE_END_PATTERNS,
        QUESTION_PATTERNS
    )

logger = logging.getLogger(__name__)


def apply_tags(
    utterance: str,
    agenda_title: Optional[str],
    role: Optional[str] = None
) -> str:
    """
    발화에 XML 태그를 적용합니다.
    
    태그는 우선순위에 따라 적용됩니다:
    1순위: <결정> (회의 결정사항)
    2순위: <안건> (법안명)
    3순위: <쟁점> (논란/문제점)
    
    Args:
        utterance: 태그를 적용할 발화 텍스트
        agenda_title: 안건명 (있는 경우)
        role: 발화자 역할 (소위원장, 위원장, 전문위원 등)
    
    Returns:
        XML 태그가 적용된 발화 텍스트
    """
    result = utterance
    
    # ============================================
    # 1순위: <결정> 태그 적용
    # ============================================
    # 소위원장/위원장의 결정 발화만 태깅
    is_chair = role in ['소위원장', '위원장']
    
    # 소위원장만 사용하는 결정 패턴 (예: "다시 의논하기로", "상임위 회부")
    chair_only_patterns = [
        r'([^\.。]*(?:다시.*의논.*하기로|다시.*의논.*하겠습니다|검토.*보고.*하겠습니다|검토.*보고.*드리도록)[^\.。]*)',
        r'([^\.。]*(?:중지하도록|계속.*심사하도록)[^\.。]*)',
        r'([^\.。]*(?:종결.*짓|종결.*지으면|종결.*하겠습니다)[^\.。]*)',
        r'([^\.。]*(?:상임위.*다시.*회부|상임위.*재회부|상임위.*회부.*결론|상임위.*회부.*논의|상임위.*회부.*계속심사|상임위.*회부.*하고자|상임위로.*다시.*회부|상임위로.*재회부|상임위로.*회부.*결론)[^\.。]*)',
    ]
    
    # 일반적인 결정 패턴 (예: "가결되었습니다", "부결되었습니다")
    general_decision_patterns = [
        r'([^\.。]*(?:가결되었습니다|부결되었습니다|의결되었습니다|통과되었습니다|가결되었음을 선포|부결되었음을 선포|의결되었음을 선포)[^\.。]*)',
        r'([^\.。]*(?:가결|부결|통과|보류|확정|재논의)(?:되었습니다|되었음을|하기로|하도록 결정)[^\.。]*)',
        r'([^\.。]*(?:채택합니다|채택하겠습니다|부의합니다)[^\.。]*)',
        r'([^\.。]*(?:원안대로|수정가결|신설하기로|폐지하기로)[^\.。]*)',
        r'([^\.。]*(?:처리하기로 결정|처리하기로 합의|처리하도록 결정)[^\.。]*)',
        r'([^\.。]*(?:유보하도록|유보하기로|유보합니다|유보하겠습니다)[^\.。]*)',
        r'([^\.。]*(?:종결.*통과|통과.*시키도록|최종.*확인.*통과)[^\.。]*)',
    ]
    
    # 질문 형태의 발화는 결정이 아님 (예: "이의 없으십니까?")
    is_question = any(re.search(pattern, result) for pattern in QUESTION_PATTERNS)
    
    # 소위원장 전용 패턴 먼저 확인
    if is_chair and not is_question:
        for pattern in chair_only_patterns:
            matches = list(re.finditer(pattern, result))
            # 뒤에서부터 매칭 (마지막 결정사항이 더 중요)
            for match in reversed(matches):
                matched_text = match.group(1).strip()
                # 이미 다른 태그가 있으면 건너뛰기
                if '<결정>' not in matched_text and '<안건>' not in matched_text and '<쟁점>' not in matched_text:
                    if len(matched_text) > 5:  # 너무 짧은 텍스트는 제외
                        start, end = match.span()
                        result = result[:start] + f'<결정>{matched_text}</결정>' + result[end:]
                        break
            # 하나라도 태깅되면 중단
            if '<결정>' in result:
                break
    
    # 소위원장 전용 패턴에서 찾지 못했으면 일반 결정 패턴 확인
    if '<결정>' not in result and not is_question and is_chair:
        for pattern in general_decision_patterns:
            matches = list(re.finditer(pattern, result))
            for match in reversed(matches):
                matched_text = match.group(1).strip()
                if '<결정>' not in matched_text and '<안건>' not in matched_text and '<쟁점>' not in matched_text:
                    if len(matched_text) > 5:
                        start, end = match.span()
                        result = result[:start] + f'<결정>{matched_text}</결정>' + result[end:]
                        break
            if '<결정>' in result:
                break
    
    # ============================================
    # 2순위: <안건> 태그 적용
    # ============================================
    # 발화에서 법안명을 찾아서 태깅
    extracted_agenda_from_utterance = None
    agenda_title_in_result = agenda_title and agenda_title in result
    
    # 안건명이 없거나 발화에 없으면 발화에서 직접 추출 시도
    if not agenda_title or not agenda_title_in_result:
        # 발화에서 법안명 추출 패턴 (예: "도로교통법 일부개정법률안을 상정")
        agenda_extraction_patterns = [
            r'의사일정\s*제\d+항[^\.。]*?([가-힣\s·및]+(?:법률안|법안|법))\s*을?\s*상정',
            r'([가-힣\s·및]+(?:법률안|법안|법))\s*을?\s*상정',
            r'의사일정\s*제\d+항[^\.。]*?([가-힣\s·및]+(?:법률안|법안|법))',
        ]
        for pattern in agenda_extraction_patterns:
            match = re.search(pattern, result)
            if match:
                extracted_agenda = match.group(1).strip()
                # 너무 짧거나 길면 제외 (5~100자)
                if len(extracted_agenda) > 5 and len(extracted_agenda) < 100:
                    extracted_agenda_from_utterance = extracted_agenda
                    if not agenda_title:
                        agenda_title = extracted_agenda
                    break
    
    # 추출한 안건명을 발화에 태깅
    if extracted_agenda_from_utterance:
        # 정확히 일치하는 경우
        if extracted_agenda_from_utterance in result:
            pattern = re.escape(extracted_agenda_from_utterance)
            matches = list(re.finditer(pattern, result))
            for match in reversed(matches):
                start, end = match.span()
                before = result[:start]
                # 이미 태깅되지 않았고, 다른 태그 안에 있지 않으면 태깅
                if before.count('<안건>') <= before.count('</안건>') and f'<안건>{extracted_agenda_from_utterance}</안건>' not in result:
                    result = result[:start] + f'<안건>{extracted_agenda_from_utterance}</안건>' + result[end:]
                    break
        else:
            # 공백/특수문자 차이 무시하고 매칭 (예: "및" vs "·")
            normalized_extracted = extracted_agenda_from_utterance.replace('및', '·').replace(' ', '')
            normalized_result = result.replace('및', '·').replace(' ', '')
            if normalized_extracted in normalized_result:
                flexible_pattern = extracted_agenda_from_utterance.replace('및', '[및·]').replace('·', '[및·]')
                flexible_pattern = re.escape(flexible_pattern).replace(r'\[및·\]', '[및·]')
                matches = list(re.finditer(flexible_pattern, result))
                for match in reversed(matches):
                    start, end = match.span()
                    matched_text = match.group(0)
                    before = result[:start]
                    if before.count('<안건>') <= before.count('</안건>') and f'<안건>{matched_text}</안건>' not in result:
                        result = result[:start] + f'<안건>{matched_text}</안건>' + result[end:]
                        break
    
    # 안건명의 다양한 표현 형태도 태깅 (예: "법률안" → "법", "일부개정법률안" → "개정안")
    if agenda_title:
        # 안건명의 변형 형태들 생성
        agenda_variants = [
            agenda_title,  # 원본
            agenda_title.replace('일부개정법률안', '개정안'),
            agenda_title.replace('일부개정법률안', '법률안'),
            agenda_title.replace('법률안', '법'),
            re.sub(r'\(.*?\)', '', agenda_title).strip(),  # 괄호 제거
        ]
        
        # 법안명에서 핵심 법명만 추출 (예: "도로교통법 일부개정법률안" → "도로교통법")
        if '일부개정법률안' in agenda_title:
            core_law = agenda_title.split('일부개정법률안')[0].strip()
            if core_law and len(core_law) > 3:
                agenda_variants.append(core_law)
                agenda_variants.append(core_law + '법률안')
        
        # 공백/특수문자 차이 무시한 변형도 추가
        normalized_agenda = agenda_title.replace('및', '·').replace(' ', '')
        normalized_variants = [
            normalized_agenda,
            normalized_agenda.replace('일부개정법률안', '법률안'),
            normalized_agenda.replace('법률안', '법'),
        ]
        agenda_variants.extend(normalized_variants)
        
        normalized_result = result.replace('및', '·').replace(' ', '')
        
        # 최대 MAX_AGENDA_TAGS개까지만 태깅 (과도한 태깅 방지)
        agenda_count = 0
        for variant in agenda_variants:
            if not variant or len(variant) < 3:
                continue
            if agenda_count >= MAX_AGENDA_TAGS:
                break
            
            if f'<안건>{variant}</안건>' in result:
                agenda_count += 1
                continue
            
            normalized_variant = variant.replace('및', '·').replace(' ', '')
            
            if variant in result:
                pattern = re.escape(variant)
                matches = list(re.finditer(pattern, result))
                for match in reversed(matches):
                    start, end = match.span()
                    before = result[:start]
                    if before.count('<안건>') <= before.count('</안건>'):
                        result = result[:start] + f'<안건>{variant}</안건>' + result[end:]
                        agenda_count += 1
                        break
            elif normalized_variant in normalized_result:
                flexible_pattern = variant.replace('및', '[및·]').replace('·', '[및·]')
                flexible_pattern = re.escape(flexible_pattern).replace(r'\[및·\]', '[및·]')
                matches = list(re.finditer(flexible_pattern, result))
                for match in reversed(matches):
                    start, end = match.span()
                    matched_text = match.group(0)
                    before = result[:start]
                    if before.count('<안건>') <= before.count('</안건>'):
                        result = result[:start] + f'<안건>{matched_text}</안건>' + result[end:]
                        agenda_count += 1
                        break
            
            if agenda_count >= MAX_AGENDA_TAGS:
                break
    
    # ============================================
    # 3순위: <쟁점> 태그 적용
    # ============================================
    # 논란/문제점을 나타내는 발화 태깅
    issue_patterns = [
        r'([^\.。]*(?:쟁점은|문제는|우려되는 점은|논란이 되는 부분은|문제점이)[^\.。]*)',
        r'([^\.。]*(?:반대|찬성).*(?:근거|이유|사유)[^\.。]*)',
    ]
    
    for pattern in issue_patterns:
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            matched_text = match.group(1).strip()
            # 너무 짧은 텍스트는 제외 (10자 이상)
            if len(matched_text) > 10:
                # 이미 다른 태그가 있으면 건너뛰기
                if '<쟁점>' not in matched_text and '<결정>' not in matched_text:
                    start, end = match.span()
                    before = result[:start]
                    # 결정 태그 안에 있지 않으면 태깅
                    if before.count('<결정>') <= before.count('</결정>'):
                        result = result[:start] + f'<쟁점>{matched_text}</쟁점>' + result[end:]
                        break
        # 하나라도 태깅되면 중단
        if '<쟁점>' in result:
            break
    
    return result


def is_important_utterance(utterance: str, role: str = '') -> bool:
    """
    발화가 요약에 중요한지 판단합니다.
    
    <IMP> 태그는 요약에 반드시 포함되어야 할 중요한 발화에만 적용됩니다.
    이미 <결정>이나 <쟁점> 태그가 있으면 <IMP>는 불필요합니다.
    
    Args:
        utterance: 판단할 발화 텍스트
        role: 발화자 역할
    
    Returns:
        True: 요약에 중요함, False: 중요하지 않음
    """
    utterance_clean = utterance.strip()
    
    # 이미 <결정>이나 <쟁점> 태그가 있으면 <IMP> 불필요 (중복 방지)
    if '<결정>' in utterance_clean or '<쟁점>' in utterance_clean:
        return False
    
    # 단순 의사진행 발화는 제외 (예: "상정합니다", "보고해 주세요")
    if any(re.search(pattern, utterance_clean) for pattern in SIMPLE_PROCEDURE_PATTERNS):
        return False
    
    # ============================================
    # 1순위: 전문위원/수석전문위원 보고
    # ============================================
    # 전문위원의 보고는 대부분 요약에 중요하지만, 단순 인사는 제외
    if role in ['전문위원', '수석전문위원']:
        # 단순 인사/반응 제외 (예: "수석전문위원입니다", "보고드리겠습니다")
        simple_greetings = [
            r'^수석전문위원입니다\.?$',
            r'^전문위원입니다\.?$',
            r'^보고드리겠습니다\.?$',
            r'^보고를\s*드리겠습니다\.?$',
            r'^보고\s*드리겠습니다\.?$',
            r'^이상\s*보고를\s*마치겠습니다\.?$',
            r'^이상\s*보고\s*마치겠습니다\.?$',
            r'^이상입니다\.?$',
            r'^조문대비표로\s*보고드리겠습니다',
            r'^조문대비표를\s*참고해\s*주시기\s*바랍니다',
            r'^조문대비표로\s*보고드리겠습니다\.?$',
            r'^조문대비표를\s*참고해\s*주시기\s*바랍니다\.?$',
            r'^보고\s*드리겠습니다\.?\s*$',
            r'^보고를\s*드리겠습니다\.?\s*$',
        ]
        if any(re.search(pattern, utterance_clean) for pattern in simple_greetings):
            return False
        
        # 최소 길이: 50자 이상 (너무 짧은 발화는 제외)
        if len(utterance_clean) < 50:
            return False
        
        # 구체적 내용이 있는지 확인 (법안명, 법조항 등)
        # 요약에 실제로 포함될 만한 내용만 중요하다고 판단
        has_content = (
            re.search(r'법률안|법안|법률|법\s*제\d+조|제\d+[조항호절]', utterance_clean) and
            len(utterance_clean) >= 60  # 최소 60자 이상
        )
        
        # 의미 없는 발화 제외 (예: "보고드리겠습니다"만 있는 경우)
        meaningless_patterns = [
            r'^[가-힣\s]{0,10}입니다\.?$',
            r'^[가-힣\s]{0,10}하겠습니다\.?$',
            r'^[가-힣\s]{0,10}드리겠습니다\.?$',
        ]
        if any(re.search(pattern, utterance_clean) for pattern in meaningless_patterns):
            return False
        
        if has_content:
            return True
    
    # ============================================
    # 1.5순위: 위원의 법안 관련 의견
    # ============================================
    # 위원의 발화 중 법안과 관련된 구체적 의견만 중요
    if role == '위원':
        # 최소 길이: 50자 이상
        if len(utterance_clean) < 50:
            return False
        
        # 법안 관련 키워드 확인
        law_related_keywords = ['법안', '법률', '법', '법적', '법제', '법인', '법령', '법규', '법조', '법률안']
        has_law_keyword = any(kw in utterance_clean for kw in law_related_keywords)
        
        # 의견 표현 키워드 확인
        opinion_keywords = ['입장', '의견', '생각', '판단', '주장', '제안', '반대', '찬성', '동의', '부동의']
        has_opinion = any(kw in utterance_clean for kw in opinion_keywords)
        
        # 법안 관련 키워드와 의견 키워드가 모두 있어야 중요하다고 판단
        if has_law_keyword and has_opinion:
            return True
        
        # 법적 근거/지위 등을 언급한 경우도 중요
        legal_status_patterns = [
            r'법적\s*(?:근거|지위|기구|단체|인정|성격)',
            r'법으로\s*(?:명문화|제정|규정|정의)',
            r'법에\s*(?:의해서|따라서|의하여)',
            r'법률안\s*(?:내용|제안|발의)',
        ]
        if any(re.search(pattern, utterance_clean) for pattern in legal_status_patterns):
            if len(utterance_clean) > 15:
                return True
    
    # ============================================
    # 2순위: 법조항 언급
    # ============================================
    # 법조항을 언급한 발화는 중요 (예: "제17조", "제2항")
    if '의사일정' in utterance_clean and re.search(r'의사일정\s*제\d+항', utterance_clean):
        pass  # 의사일정 번호는 제외 (안건 태그에서 처리)
    elif re.search(r'(?:제)?\d+[조항호절]', utterance_clean):
        return True
    
    # ============================================
    # 3순위: 정부/부처 의견
    # ============================================
    # 정부 부처의 의견은 중요하지만, 법안 관련 내용이 있어야 함
    if role in GOVERNMENT_ORGS or any(org in utterance_clean for org in GOVERNMENT_ORGS):
        # 최소 길이: 40자 이상
        if len(utterance_clean) < 40:
            return False
        # 법안 관련 내용이 있어야 중요
        if re.search(r'법률안|법안|법률|법\s*제\d+조|제\d+[조항호절]', utterance_clean):
            return True
    
    # ============================================
    # 4순위: 의사진행 결정 발화
    # ============================================
    # 실제 결정사항을 나타내는 발화 (예: "가결되었습니다")
    important_procedure_patterns = [
        r'가결되었습니다', r'부결되었습니다', r'의결되었습니다', r'통과되었습니다',
        r'가결되었음을\s*선포', r'부결되었음을\s*선포', r'의결되었음을\s*선포',
        r'원안대로', r'수정가결', r'신설하기로', r'폐지하기로',
        r'처리하기로\s*결정', r'처리하기로\s*합의', r'채택합니다', r'부의합니다'
    ]
    if any(re.search(pattern, utterance_clean) for pattern in important_procedure_patterns):
        return True
    
    # 날짜/연도만 언급한 경우는 중요하지 않음 (요약에 거의 포함되지 않음)
    # 따라서 제외
    
    return False


def calculate_rule_based_importance(
    utterances: List[str],
    roles: List[str] = None
) -> List[bool]:
    """
    각 발화가 중요한지 판단하여 True/False 리스트를 반환합니다.
    
    모든 발화가 중요하지 않으면, 첫 번째/마지막 발화를 중요하게 처리합니다.
    (요약에 최소한의 정보는 포함되어야 하므로)
    
    Args:
        utterances: 발화 리스트
        roles: 발화자 역할 리스트
    
    Returns:
        각 발화의 중요도 (True/False) 리스트
    """
    if roles is None:
        roles = [''] * len(utterances)
    
    if len(utterances) != len(roles):
        roles = [''] * len(utterances)
    
    # 각 발화의 중요도 판단
    is_important = []
    for utterance, role in zip(utterances, roles):
        is_important.append(is_important_utterance(utterance, role))
    
    # 모든 발화가 중요하지 않으면, 첫 번째/마지막 발화를 중요하게 처리
    if not any(is_important) and len(utterances) > 0:
        first_role = roles[0] if roles and len(roles) > 0 else ''
        if not any(re.search(pattern, utterances[0]) for pattern in SIMPLE_END_PATTERNS):
            if first_role == '소위원장':
                if any(re.search(pattern, utterances[0]) for pattern in SIMPLE_CHAIR_PATTERNS):
                    pass
                else:
                    is_important[0] = True
            else:
                is_important[0] = True
        
        if len(utterances) > 1:
            last_role = roles[-1] if roles and len(roles) > 0 else ''
            if not any(re.search(pattern, utterances[-1]) for pattern in SIMPLE_END_PATTERNS):
                if last_role == '소위원장':
                    if any(re.search(pattern, utterances[-1]) for pattern in SIMPLE_CHAIR_PATTERNS):
                        pass
                    else:
                        is_important[-1] = True
                else:
                    is_important[-1] = True
    
    return is_important


def apply_importance_tags(
    utterances: List[str],
    roles: List[str] = None,
    min_imp_ratio: float = 0.15,
    max_imp_ratio: float = 0.25
) -> List[str]:
    """
    중요 발화에 <IMP> 태그를 적용합니다.
    
    태그 밀도를 조절하여 전체 발화의 15~25%만 <IMP> 태그를 받도록 합니다.
    전문위원/수석전문위원의 발화는 우선적으로 태깅됩니다.
    
    Args:
        utterances: 발화 리스트
        roles: 발화자 역할 리스트
        min_imp_ratio: 최소 중요 태그 비율 (기본값: 0.15 = 15%)
        max_imp_ratio: 최대 중요 태그 비율 (기본값: 0.25 = 25%)
    
    Returns:
        <IMP> 태그가 적용된 발화 리스트
    """
    is_important = calculate_rule_based_importance(utterances, roles)
    
    # ============================================
    # 1단계: 초기 태깅 후보 선정
    # ============================================
    tagged_utterances = []
    imp_candidates = []  # <IMP> 태그 후보 인덱스
    
    for i, utterance in enumerate(utterances):
        # 이미 <결정>이나 <쟁점> 태그가 있으면 <IMP> 불필요
        has_decision_tag = '<결정>' in utterance
        has_issue_tag = '<쟁점>' in utterance
        
        if has_decision_tag or has_issue_tag:
            tagged_utterances.append(utterance)
            continue
        
        # 중요하다고 판단된 발화를 후보에 추가
        if is_important[i] and '<IMP>' not in utterance:
            if not (utterance.startswith('<') and utterance.endswith('>')):
                tagged_utterances.append(utterance)
                imp_candidates.append(i)
            else:
                tagged_utterances.append(utterance)
        else:
            tagged_utterances.append(utterance)
    
    # ============================================
    # 2단계: 전문위원 발화 우선 처리
    # ============================================
    expert_indices = set()  # 전문위원/수석전문위원 발화 인덱스
    other_candidates = []  # 그 외 발화 인덱스
    
    # 후보 중에서 전문위원 발화 분리
    for idx in imp_candidates:
        role = roles[idx] if roles and idx < len(roles) else ''
        if role in ['전문위원', '수석전문위원']:
            expert_indices.add(idx)
        else:
            other_candidates.append(idx)
    
    # 후보에 없지만 전문위원 발화인 경우도 확인 (일관된 필터링 적용)
    for i, utterance in enumerate(utterances):
        if i in expert_indices:
            continue
        role = roles[i] if roles and i < len(roles) else ''
        if role in ['전문위원', '수석전문위원']:
            has_decision_tag = '<결정>' in utterance
            has_issue_tag = '<쟁점>' in utterance
            if not has_decision_tag and not has_issue_tag:
                # is_important_utterance 함수로 필터링 (일관성 유지)
                if is_important_utterance(utterance, role):
                    expert_indices.add(i)
    
    # 전문위원 발화에 <IMP> 태그 적용 (이미 필터링됨)
    for idx in expert_indices:
        if '<IMP>' not in tagged_utterances[idx]:
            tagged_utterances[idx] = f'<IMP>{tagged_utterances[idx]}</IMP>'
    
    # ============================================
    # 3단계: 나머지 발화에 태그 밀도 조절 적용
    # ============================================
    total_utterances = len(utterances)
    min_imp_count = max(1, int(total_utterances * min_imp_ratio))  # 최소 1개
    max_imp_count = int(total_utterances * max_imp_ratio)
    
    if len(other_candidates) > 0:
        # 중요도 점수 계산 (법조항 > 정부 부처 > 의사진행 키워드)
        priority_scores = []
        for idx in other_candidates:
            utterance = utterances[idx]
            role = roles[idx] if roles and idx < len(roles) else ''
            score = 0
            
            if re.search(r'제\d+[조항호절]', utterance):  # 법조항 언급
                score += 50
            if role in GOVERNMENT_ORGS or any(org in utterance for org in GOVERNMENT_ORGS):  # 정부 부처
                score += 30
            if any(kw in utterance for kw in PROCEDURE_KEYWORDS):  # 의사진행 키워드
                score += 20
            
            priority_scores.append((idx, score))
        
        # 점수 순으로 정렬
        priority_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 전문위원 발화 수를 제외한 나머지 슬롯 계산
        expert_count = len(expert_indices)
        remaining_slots = max(0, max_imp_count - expert_count)
        selected_count = min(max(min_imp_count, len(other_candidates)), remaining_slots)
        selected_indices = {idx for idx, _ in priority_scores[:selected_count]}
        
        # 선택된 발화에 <IMP> 태그 적용
        for idx in selected_indices:
            if '<IMP>' not in tagged_utterances[idx]:
                tagged_utterances[idx] = f'<IMP>{tagged_utterances[idx]}</IMP>'
    
    return tagged_utterances

