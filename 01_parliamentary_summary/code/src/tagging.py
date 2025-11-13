"""
태그 적용 모듈
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .config import (
        GOVERNMENT_ORGS, PROCEDURE_KEYWORDS, MAX_KEY_TAGS, MAX_AGENDA_TAGS,
        SIMPLE_PROCEDURE_PATTERNS, SIMPLE_CHAIR_PATTERNS, SIMPLE_END_PATTERNS,
        QUESTION_PATTERNS
    )
except ImportError:
    from config import (
        GOVERNMENT_ORGS, PROCEDURE_KEYWORDS, MAX_KEY_TAGS, MAX_AGENDA_TAGS,
        SIMPLE_PROCEDURE_PATTERNS, SIMPLE_CHAIR_PATTERNS, SIMPLE_END_PATTERNS,
        QUESTION_PATTERNS
    )

logger = logging.getLogger(__name__)


def extract_key_information(utterance: str) -> Tuple[str, List[str]]:
    """
    발화에서 중요한 정보를 추출합니다.
    """
    key_tokens = []
    
    forbidden_words = {
        '학생', '학교', '의원', '위원', '제도', '문제', '사항', '내용', '부분',
        '정부', '국민', '사회', '국가', '법률', '법안', '안건', '사실', '경우'
    }
    
    # 1순위: 의사일정 번호
    agenda_num_pattern = r'의사일정\s*(제\d+항)'
    agenda_nums = re.findall(agenda_num_pattern, utterance)
    key_tokens.extend(agenda_nums)
    
    # 2순위: 법조항
    clause_pattern = r'(제\d+조|제\d+항|제\d+호|제\d+절)'
    clauses = re.findall(clause_pattern, utterance)
    clauses = [c for c in clauses if c not in agenda_nums]
    key_tokens.extend(clauses)
    
    # 3순위: 법령명/법안명
    law_pattern = r'([가-힣]{2,}(?:법|법률|법안))(?:\s*(?:일부개정|개정|제정|폐지))?'
    laws = re.findall(law_pattern, utterance)
    laws = [law for law in laws if law not in ['법률', '법안'] and len(law) > 3]
    key_tokens.extend(laws)
    
    # 4순위: 정부 부처/기관명
    org_pattern = r'([가-힣]{2,}(?:부|청|위원회|처|원|실|본부|단|관|국))'
    orgs = re.findall(org_pattern, utterance)
    forbidden_orgs = {
        '위원회', '본부', '실', '관', '국', '법안심사소위원회', '소위원회',
        '전문위원회', '이사회', '심의위원회'
    }
    orgs = [org for org in orgs if org not in forbidden_orgs and len(org) > 2]
    orgs = [org for org in orgs if any(gov_org in org for gov_org in GOVERNMENT_ORGS)]
    key_tokens.extend(orgs)
    
    # 5순위: 날짜/연도
    date_pattern = r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}월\s*\d{1,2}일|\d{4}년)'
    dates = re.findall(date_pattern, utterance)
    key_tokens.extend(dates)
    
    # 필터링
    key_tokens = [token for token in key_tokens if token not in forbidden_words]
    general_orgs = ['법안심사소위원회', '소위원회', '전문위원회', '이사회', '심의위원회']
    key_tokens = [token for token in key_tokens if not any(gen_org in token for gen_org in general_orgs)]
    key_tokens = list(set([token.strip() for token in key_tokens if len(token.strip()) > 1]))
    
    # 한 문장에 3개 이상이면 1-2개만 선택
    if len(key_tokens) > 2:
        priority_order = []
        for token in key_tokens:
            if any(x in token for x in ['법', '부', '청', '위원회', '처']):
                priority_order.append(token)
        for token in key_tokens:
            if token not in priority_order and any(x in token for x in ['제', '년', '월', '일']):
                priority_order.append(token)
        
        if priority_order:
            key_tokens = priority_order[:2]
        else:
            key_tokens = key_tokens[:2]
    
    return utterance, key_tokens


def apply_tags(
    utterance: str,
    agenda_title: Optional[str],
    key_tokens: List[str],
    role: Optional[str] = None
) -> str:
    """
    발화에 XML 태그를 적용합니다.
    태그 우선순위: <결정> > <안건> > <쟁점> > <KEY>
    """
    result = utterance
    
    # 1순위: <결정> 태그 (소위원장/위원장만)
    is_chair = role in ['소위원장', '위원장']
    
    chair_only_patterns = [
        r'([^\.。]*(?:다시.*의논.*하기로|다시.*의논.*하겠습니다|검토.*보고.*하겠습니다|검토.*보고.*드리도록)[^\.。]*)',
        r'([^\.。]*(?:중지하도록|계속.*심사하도록)[^\.。]*)',
        r'([^\.。]*(?:종결.*짓|종결.*지으면|종결.*하겠습니다)[^\.。]*)',
        r'([^\.。]*(?:상임위.*다시.*회부|상임위.*재회부|상임위.*회부.*결론|상임위.*회부.*논의|상임위.*회부.*계속심사|상임위.*회부.*하고자|상임위로.*다시.*회부|상임위로.*재회부|상임위로.*회부.*결론)[^\.。]*)',
    ]
    
    general_decision_patterns = [
        r'([^\.。]*(?:가결되었습니다|부결되었습니다|의결되었습니다|통과되었습니다|가결되었음을 선포|부결되었음을 선포|의결되었음을 선포)[^\.。]*)',
        r'([^\.。]*(?:가결|부결|통과|보류|확정|재논의)(?:되었습니다|되었음을|하기로|하도록 결정)[^\.。]*)',
        r'([^\.。]*(?:채택합니다|채택하겠습니다|부의합니다)[^\.。]*)',
        r'([^\.。]*(?:원안대로|수정가결|신설하기로|폐지하기로)[^\.。]*)',
        r'([^\.。]*(?:처리하기로 결정|처리하기로 합의|처리하도록 결정)[^\.。]*)',
        r'([^\.。]*(?:유보하도록|유보하기로|유보합니다|유보하겠습니다)[^\.。]*)',
        r'([^\.。]*(?:종결.*통과|통과.*시키도록|최종.*확인.*통과)[^\.。]*)',
    ]
    
    is_question = any(re.search(pattern, result) for pattern in QUESTION_PATTERNS)
    
    # 소위원장 전용 패턴 적용
    if is_chair and not is_question:
        for pattern in chair_only_patterns:
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
    
    # 일반 결정 패턴 적용
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
    
    # 2순위: <안건> 태그
    extracted_agenda_from_utterance = None
    agenda_title_in_result = agenda_title and agenda_title in result
    if not agenda_title or not agenda_title_in_result:
        agenda_extraction_patterns = [
            r'의사일정\s*제\d+항[^\.。]*?([가-힣\s·및]+(?:법률안|법안|법))\s*을?\s*상정',
            r'([가-힣\s·및]+(?:법률안|법안|법))\s*을?\s*상정',
            r'의사일정\s*제\d+항[^\.。]*?([가-힣\s·및]+(?:법률안|법안|법))',
        ]
        for pattern in agenda_extraction_patterns:
            match = re.search(pattern, result)
            if match:
                extracted_agenda = match.group(1).strip()
                if len(extracted_agenda) > 5 and len(extracted_agenda) < 100:
                    extracted_agenda_from_utterance = extracted_agenda
                    if not agenda_title:
                        agenda_title = extracted_agenda
                    break
    
    # 추출된 안건명 태깅
    if extracted_agenda_from_utterance:
        if extracted_agenda_from_utterance in result:
            pattern = re.escape(extracted_agenda_from_utterance)
            matches = list(re.finditer(pattern, result))
            for match in reversed(matches):
                start, end = match.span()
                before = result[:start]
                if before.count('<안건>') <= before.count('</안건>') and f'<안건>{extracted_agenda_from_utterance}</안건>' not in result:
                    result = result[:start] + f'<안건>{extracted_agenda_from_utterance}</안건>' + result[end:]
                    break
        else:
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
    
    # agenda_title 태깅
    if agenda_title:
        agenda_variants = [
            agenda_title,
            agenda_title.replace('일부개정법률안', '개정안'),
            agenda_title.replace('일부개정법률안', '법률안'),
            agenda_title.replace('법률안', '법'),
            re.sub(r'\(.*?\)', '', agenda_title).strip(),
        ]
        
        if '일부개정법률안' in agenda_title:
            core_law = agenda_title.split('일부개정법률안')[0].strip()
            if core_law and len(core_law) > 3:
                agenda_variants.append(core_law)
                agenda_variants.append(core_law + '법률안')
        
        normalized_agenda = agenda_title.replace('및', '·').replace(' ', '')
        normalized_variants = [
            normalized_agenda,
            normalized_agenda.replace('일부개정법률안', '법률안'),
            normalized_agenda.replace('법률안', '법'),
        ]
        agenda_variants.extend(normalized_variants)
        
        normalized_result = result.replace('및', '·').replace(' ', '')
        
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
    
    # 3순위: <쟁점> 태그
    issue_patterns = [
        r'([^\.。]*(?:쟁점은|문제는|우려되는 점은|논란이 되는 부분은|문제점이)[^\.。]*)',
        r'([^\.。]*(?:반대|찬성).*(?:근거|이유|사유)[^\.。]*)',
    ]
    
    for pattern in issue_patterns:
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            matched_text = match.group(1).strip()
            if len(matched_text) > 10:
                if '<쟁점>' not in matched_text and '<결정>' not in matched_text:
                    start, end = match.span()
                    before = result[:start]
                    if before.count('<결정>') <= before.count('</결정>'):
                        result = result[:start] + f'<쟁점>{matched_text}</쟁점>' + result[end:]
                        break
        if '<쟁점>' in result:
            break
    
    # 4순위: <KEY> 태그
    key_count = 0
    
    for token in key_tokens:
        if key_count >= MAX_KEY_TAGS:
            break
        
        if f'<KEY>{token}</KEY>' in result:
            continue
        
        pattern = re.escape(token)
        matches = list(re.finditer(pattern, result))
        for match in reversed(matches):
            start, end = match.span()
            before = result[:start]
            
            if (before.count('<결정>') > before.count('</결정>') or
                before.count('<안건>') > before.count('</안건>') or
                before.count('<쟁점>') > before.count('</쟁점>')):
                continue
            
            forbidden_in_context = ['학생', '위원', '의원', '제도', '문제']
            if any(fw in token for fw in forbidden_in_context):
                context = result[max(0, start-10):min(len(result), end+10)]
                if not any(x in context for x in ['제', '부', '청', '법']):
                    continue
            
            result = result[:start] + f'<KEY>{token}</KEY>' + result[end:]
            key_count += 1
            break
    
    return result


def is_important_utterance(utterance: str, role: str = '') -> bool:
    """
    발화가 요약에 중요한지 규칙 기반으로 판단합니다.
    """
    utterance_clean = utterance.strip()
    
    # 이미 중요한 태그가 있으면 <IMP> 불필요
    if '<결정>' in utterance_clean or '<쟁점>' in utterance_clean:
        return False
    
    # 단순 의사진행 발화 제외
    if any(re.search(pattern, utterance_clean) for pattern in SIMPLE_PROCEDURE_PATTERNS):
        return False
    
    # 1순위: 전문위원/수석전문위원 보고
    if role in ['전문위원', '수석전문위원']:
        if len(utterance_clean) > 5:
            return True
    
    # 1.5순위: 위원 발화 중 법안 관련 중요한 의견/설명
    if role == '위원':
        law_related_keywords = ['법안', '법률', '법', '법적', '법제', '법인', '법령', '법규', '법조', '법률안']
        has_law_keyword = any(kw in utterance_clean for kw in law_related_keywords)
        
        opinion_keywords = ['입장', '의견', '생각', '판단', '주장', '제안', '반대', '찬성', '동의', '부동의']
        has_opinion = any(kw in utterance_clean for kw in opinion_keywords)
        
        if len(utterance_clean) >= 40 and (has_law_keyword or has_opinion):
            return True
        
        legal_status_patterns = [
            r'법적\s*(?:근거|지위|기구|단체|인정|성격)',
            r'법으로\s*(?:명문화|제정|규정|정의)',
            r'법에\s*(?:의해서|따라서|의하여)',
            r'법률안\s*(?:내용|제안|발의)',
        ]
        if any(re.search(pattern, utterance_clean) for pattern in legal_status_patterns):
            if len(utterance_clean) > 15:
                return True
    
    # 2순위: 법조항 언급
    if '의사일정' in utterance_clean and re.search(r'의사일정\s*제\d+항', utterance_clean):
        pass
    elif re.search(r'(?:제)?\d+[조항호절]', utterance_clean):
        return True
    
    # 3순위: 정부/부처 의견
    if role in GOVERNMENT_ORGS or any(org in utterance_clean for org in GOVERNMENT_ORGS):
        if len(utterance_clean) > 15:
            return True
    
    # 4순위: 의사진행 관련 발화 (실제 결정만)
    important_procedure_patterns = [
        r'가결되었습니다', r'부결되었습니다', r'의결되었습니다', r'통과되었습니다',
        r'가결되었음을\s*선포', r'부결되었음을\s*선포', r'의결되었음을\s*선포',
        r'원안대로', r'수정가결', r'신설하기로', r'폐지하기로',
        r'처리하기로\s*결정', r'처리하기로\s*합의', r'채택합니다', r'부의합니다'
    ]
    if any(re.search(pattern, utterance_clean) for pattern in important_procedure_patterns):
        return True
    
    # 5순위: 날짜/연도 언급
    if re.search(r'\d{4}년', utterance_clean):
        if len(utterance_clean) > 10:
            return True
    
    return False


def calculate_rule_based_importance(
    utterances: List[str],
    roles: List[str] = None
) -> List[bool]:
    """
    규칙 기반으로 중요 문장을 찾습니다.
    """
    if roles is None:
        roles = [''] * len(utterances)
    
    if len(utterances) != len(roles):
        roles = [''] * len(utterances)
    
    is_important = []
    for utterance, role in zip(utterances, roles):
        is_important.append(is_important_utterance(utterance, role))
    
    # 최소한 일부 발화는 중요하게 태깅
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
    min_imp_ratio: float = 0.3,
    max_imp_ratio: float = 0.5
) -> List[str]:
    """
    규칙 기반으로 <IMP> 태그를 적용합니다.
    """
    is_important = calculate_rule_based_importance(utterances, roles)
    
    # 1단계: 중첩 방지 및 초기 태깅
    tagged_utterances = []
    imp_candidates = []
    
    for i, utterance in enumerate(utterances):
        has_decision_tag = '<결정>' in utterance
        has_issue_tag = '<쟁점>' in utterance
        
        if has_decision_tag or has_issue_tag:
            tagged_utterances.append(utterance)
            continue
        
        if is_important[i] and '<IMP>' not in utterance:
            if not (utterance.startswith('<') and utterance.endswith('>')):
                tagged_utterances.append(utterance)
                imp_candidates.append(i)
            else:
                tagged_utterances.append(utterance)
        else:
            tagged_utterances.append(utterance)
    
    # 2단계: 태그 밀도 조절
    expert_indices = set()
    other_candidates = []
    
    for idx in imp_candidates:
        role = roles[idx] if roles and idx < len(roles) else ''
        if role in ['전문위원', '수석전문위원']:
            expert_indices.add(idx)
        else:
            other_candidates.append(idx)
    
    # imp_candidates에 없는 전문위원 발화도 찾아서 태깅
    for i, utterance in enumerate(utterances):
        if i in expert_indices:
            continue
        role = roles[i] if roles and i < len(roles) else ''
        if role in ['전문위원', '수석전문위원']:
            has_decision_tag = '<결정>' in utterance
            has_issue_tag = '<쟁점>' in utterance
            if not has_decision_tag and not has_issue_tag:
                utterance_clean = utterance.strip()
                is_simple = any(re.search(pattern, utterance_clean) for pattern in SIMPLE_PROCEDURE_PATTERNS)
                if not is_simple and len(utterance_clean) > 5:
                    expert_indices.add(i)
    
    # 전문위원 발화는 모두 IMP 태그 적용
    for idx in expert_indices:
        if '<IMP>' not in tagged_utterances[idx]:
            tagged_utterances[idx] = f'<IMP>{tagged_utterances[idx]}</IMP>'
    
    # 나머지 발화에 대해서만 태그 밀도 조절 적용
    total_utterances = len(utterances)
    min_imp_count = max(1, int(total_utterances * min_imp_ratio))
    max_imp_count = int(total_utterances * max_imp_ratio)
    
    if len(other_candidates) > 0:
        priority_scores = []
        for idx in other_candidates:
            utterance = utterances[idx]
            role = roles[idx] if roles and idx < len(roles) else ''
            score = 0
            
            if re.search(r'제\d+[조항호절]', utterance):
                score += 50
            if role in GOVERNMENT_ORGS or any(org in utterance for org in GOVERNMENT_ORGS):
                score += 30
            if any(kw in utterance for kw in PROCEDURE_KEYWORDS):
                score += 20
            
            priority_scores.append((idx, score))
        
        priority_scores.sort(key=lambda x: x[1], reverse=True)
        
        expert_count = len(expert_indices)
        remaining_slots = max(0, max_imp_count - expert_count)
        selected_count = min(max(min_imp_count, len(other_candidates)), remaining_slots)
        selected_indices = {idx for idx, _ in priority_scores[:selected_count]}
        
        for idx in selected_indices:
            if '<IMP>' not in tagged_utterances[idx]:
                tagged_utterances[idx] = f'<IMP>{tagged_utterances[idx]}</IMP>'
    
    return tagged_utterances

