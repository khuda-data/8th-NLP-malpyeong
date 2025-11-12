"""
국회 회의록 요약 데이터셋 전처리 스크립트

이 스크립트는 국립국어원 "국회 회의록 요약" 데이터셋을 전처리합니다.
각 샘플의 participants, agenda, dialogue 정보를 처리하여 
전처리된 대화 텍스트를 생성합니다.
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def load_json_files(data_path: str) -> List[Dict]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        data_path: JSON 파일 경로
        
    Returns:
        샘플 리스트
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def normalize_role(occupation: str) -> str:
    """
    역할(occupation)을 표준화합니다.
    
    Args:
        occupation: 원본 occupation 값
        
    Returns:
        표준화된 역할
    """
    if not occupation:
        return '비지정'
    
    # 호칭어 제거 및 표준화
    role = occupation.strip()
    
    # "님" 제거
    role = re.sub(r'님$', '', role)
    
    # 역할 표준화 매핑
    role_mapping = {
        '위원장': '소위원장',
        '위원장님': '소위원장',
        '소위원장님': '소위원장',
    }
    
    if role in role_mapping:
        role = role_mapping[role]
    
    return role


def normalize_speaker_name(speaker: str) -> str:
    """
    발화자 이름을 정규화합니다 (호칭어 제거, 한글 이름만).
    
    Args:
        speaker: 원본 발화자 이름
        
    Returns:
        정규화된 발화자 이름
    """
    if not speaker:
        return ''
    
    # 호칭어 제거
    name = speaker.strip()
    
    # "님" 제거
    name = re.sub(r'님$', '', name)
    
    # 한글 이름만 추출 (한자, 영문 제거)
    # 한글 이름 패턴: 2-4글자 한글
    korean_name_pattern = r'^[가-힣]{2,4}'
    match = re.match(korean_name_pattern, name)
    if match:
        name = match.group(0)
    else:
        # 한글이 없으면 원본 유지 (예외 처리)
        name = re.sub(r'[^\w가-힣\s]', '', name).strip()
    
    return name


def extract_government_org(occupation: str) -> Optional[str]:
    """
    occupation에서 정부 기관명을 추출합니다.
    
    Args:
        occupation: occupation 값
        
    Returns:
        추출된 기관명 또는 None
    """
    if not occupation:
        return None
    
    # 정부 기관명 리스트 (긴 것부터 매칭)
    government_orgs = [
        '과학기술정보통신부', '농림축산식품부', '문화체육관광부', '산업통상자원부',
        '중소벤처기업부', '기획예산처', '국무조정실', '식품의약품안전처',
        '공정거래위원회', '금융위원회', '국가보훈부', '인사혁신처',
        '행정안전부', '보건복지부', '기획재정부', '국토교통부', '해양수산부',
        '여성가족부', '고용노동부', '환경부', '교육부', '법무부', '외교부',
        '국방부', '법제처'
    ]
    
    # 긴 것부터 매칭 (부분 문자열 매칭 방지)
    for org in government_orgs:
        if org in occupation:
            return org
    
    return None


def infer_role_from_occupation(occupation: str, speaker_id: str) -> str:
    """
    occupation이 없을 때 역할을 추론합니다.
    
    Args:
        occupation: occupation 값 (없을 수 있음)
        speaker_id: 발화자 ID
        
    Returns:
        추론된 역할
    """
    # occupation이 있으면 처리
    if occupation:
        # 먼저 정부 기관명 추출 시도 (예: "보건복지부차관" → "보건복지부")
        org = extract_government_org(occupation)
        if org:
            return org
        
        # 기관명이 없으면 역할 정규화
        normalized = normalize_role(occupation)
        return normalized
    
    # occupation이 없을 때 추론
    # 정부측 인물 판단 (부처명 기반)
    government_orgs = [
        '교육부', '법무부', '외교부', '국방부', '행정안전부', '문화체육관광부',
        '농림축산식품부', '산업통상자원부', '보건복지부', '환경부', '고용노동부',
        '여성가족부', '국토교통부', '해양수산부', '중소벤처기업부', '과학기술정보통신부',
        '기획재정부', '기획예산처', '국무조정실', '법제처', '인사혁신처',
        '공정거래위원회', '금융위원회', '국가보훈부', '식품의약품안전처'
    ]
    
    # 발화자 이름이나 ID에 부처명이 포함되어 있는지 확인
    for org in government_orgs:
        if org in speaker_id:
            # 부처명을 역할로 사용 (예: "교육부")
            return org
    
    # 위원회 구성원으로 추정
    if '위원' in speaker_id or '의원' in speaker_id:
        return '위원'
    
    # 그 외는 비지정
    return '비지정'


def map_speaker_roles(participants: List[Dict]) -> Dict[str, str]:
    """
    발화자 이름을 역할(occupation)로 매핑합니다.
    
    Args:
        participants: 참가자 리스트 (id, occupation 필드 포함)
        
    Returns:
        {발화자_이름: 역할} 딕셔너리
    """
    role_map = {}
    for participant in participants:
        speaker_id = participant.get('id', '')
        occupation = participant.get('occupation', '')
        
        # 역할 추론 및 표준화
        role = infer_role_from_occupation(occupation, speaker_id)
        role_map[speaker_id] = role
    
    return role_map


def extract_agenda_from_dialogue(dialogue: List[Dict]) -> Optional[str]:
    """
    대화 내용에서 안건명을 추출합니다.
    
    Args:
        dialogue: 대화 리스트
        
    Returns:
        추출된 안건명 또는 None
    """
    # 안건 관련 키워드 패턴 (우선순위 순)
    agenda_patterns = [
        r'의사일정\s*제\d+항\s*([^을를까지]+?)\s*(?:일부개정법률안|법률안|안건)',
        r'([^을를까지]+?)\s*(?:일부개정법률안|법률안)\s*을?\s*상정',
        r'안건은\s*([^은는]+?)\s*(?:입니다|입니다\.)',
        r'([가-힣\s·]+법)\s*(?:일부개정|개정|제정|폐지)?\s*(?:법률안|안건)',
        r'상정하(?:겠습니다|도록 하겠습니다|합니다)\s*([가-힣\s·]+법)',
    ]
    
    # 처음 몇 개 발화에서 안건 찾기 (일반적으로 앞부분에 나옴)
    search_range = min(20, len(dialogue))
    
    for i in range(search_range):
        utterance_obj = dialogue[i]
        utterance = utterance_obj.get('utterance', '')
        
        for pattern in agenda_patterns:
            match = re.search(pattern, utterance)
            if match:
                agenda = match.group(1).strip()
                # 불필요한 공백 제거
                agenda = re.sub(r'\s+', ' ', agenda)
                # 너무 짧거나 긴 것은 제외
                if 5 <= len(agenda) <= 100:
                    return agenda
    
    return None


def clean_utterance(utterance: str) -> bool:
    """
    발화가 불필요한지 판단합니다.
    데이터셋 패턴을 반영하여 정보 가치가 없는 발화를 제거합니다.
    
    Args:
        utterance: 발화 텍스트
        
    Returns:
        True면 제거, False면 유지
    """
    utterance_clean = utterance.strip()
    
    # 1. 단순 반응 표현 제거
    simple_responses = [
        r'^네\s*\.?\s*$',
        r'^예\s*\.?\s*$',
        r'^네네\s*\.?\s*$',
        r'^예예\s*\.?\s*$',
        r'^음\s*\.?\s*$',
        r'^아\s*\.?\s*$',
        r'^그렇죠\s*\.?\s*$',
        r'^맞습니다\s*\.?\s*$',
        r'^그렇습니다\s*\.?\s*$',  # 단순 긍정 (내용 없음)
    ]
    
    for pattern in simple_responses:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 2. 인사/예의 표현 제거
    greetings = [
        r'^감사합니다\s*\.?\s*$',
        r'^수고하셨습니다\s*\.?\s*$',
        r'^고생하셨습니다\s*\.?\s*$',
        r'^수고하셨습니다\s*\.?\s*$',
    ]
    
    for pattern in greetings:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 3. 호응성/채우기성 표현 제거 (내용 없이 끝나는 경우)
    filler_expressions = [
        r'^보고드리겠습니다\s*\.?\s*$',  # 내용 없이 끝나는 경우
        r'^보고드리도록 하겠습니다\s*\.?\s*$',
        r'^말씀드렸잖아요\s*\.?\s*$',
        r'^아시다시피\s*\.?\s*$',
        r'^제가 볼 때는요\s*\.?\s*$',
    ]
    
    for pattern in filler_expressions:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 4. 잡담/웃음/소음 표현 제거
    noise_expressions = [
        r'^하하\s*\.?\s*$',
        r'^웃음\s*\.?\s*$',
        r'^잠시만요\s*\.?\s*$',
        r'^잠깐만요\s*\.?\s*$',
        r'^잠깐\s*\.?\s*$',
    ]
    
    for pattern in noise_expressions:
        if re.match(pattern, utterance_clean, re.IGNORECASE):
            return True
    
    # 5. 너무 짧은 발화 (2글자 이하) 제거 (단, 의사진행 관련은 제외)
    if len(utterance_clean) <= 2:
        # 의사진행 관련 키워드가 있으면 유지
        procedure_keywords = ['가결', '부결', '보류', '통과', '상정', '의결', '원안', '수정']
        if not any(keyword in utterance_clean for keyword in procedure_keywords):
            return True
    
    # 6. 삭제 금지 항목 체크 (정보 손실 방지)
    # 의사 진행 발화는 보존
    procedure_keywords = [
        '상정합니다', '상정하겠습니다', '상정하도록 하겠습니다',
        '차기 논의', '지금부터', '정리하겠습니다',
        '이의 없으시지요', '이의 없으십니까', '이의 없습니까',
        '가결되었습니다', '가결되었음을 선포합니다',
        '부결', '보류', '통과', '원안대로', '수정가결', '신설하기로',
        '의결', '합의', '결정', '처리', '채택', '부의'
    ]
    
    # 결정/합의 관련 발화는 보존
    if any(keyword in utterance_clean for keyword in procedure_keywords):
        return False
    
    # 책임 있는 발언 (정부 입장, 위원 의견, 쟁점, 근거)는 보존
    important_keywords = [
        '정부', '위원', '의견', '쟁점', '근거', '반대', '찬성',
        '제안', '검토', '보고', '설명', '논의', '문제', '사항'
    ]
    
    # 중요한 키워드가 있고 발화가 충분히 길면 보존
    if any(keyword in utterance_clean for keyword in important_keywords) and len(utterance_clean) > 5:
        return False
    
    # "동의합니다"는 단독으로 사용될 때는 보존 (결정에 중요)
    # 하지만 다른 내용과 함께 있으면 그대로 유지
    if utterance_clean == '동의합니다' or utterance_clean == '동의합니다.':
        return False  # 결정에 중요한 정보이므로 보존
    
    # "좋습니다", "알겠습니다"는 단독일 때만 제거
    if len(utterance_clean) <= 10:
        trivial_alone = [
            r'^좋습니다\s*\.?\s*$',
            r'^알겠습니다\s*\.?\s*$',
            r'^네\s*,?\s*알겠습니다\s*\.?\s*$',
            r'^예\s*,?\s*알겠습니다\s*\.?\s*$',
        ]
        for pattern in trivial_alone:
            if re.match(pattern, utterance_clean, re.IGNORECASE):
                return True
    
    return False


def extract_key_information(utterance: str) -> Tuple[str, List[str]]:
    """
    발화에서 중요한 정보를 추출합니다.
    <KEY> 태그 오남용 방지를 위해 엄격한 기준 적용.
    
    Args:
        utterance: 발화 텍스트
        
    Returns:
        (수정된_발화, KEY_태그_리스트) 튜플
    """
    key_tokens = []
    
    # 금지 단어 목록 (일반명사, 역할명 등)
    forbidden_words = {
        '학생', '학교', '의원', '위원', '제도', '문제', '사항', '내용', '부분',
        '정부', '국민', '사회', '국가', '법률', '법안', '안건', '사실', '경우'
    }
    
    # 1. 법령명/법안명 (고유명사만)
    # "○○법", "○○법률", "○○법안" 패턴
    law_pattern = r'([가-힣]{2,}(?:법|법률|법안))(?:\s*(?:일부개정|개정|제정|폐지))?'
    laws = re.findall(law_pattern, utterance)
    # 일반적인 "법률", "법안" 단어는 제외
    laws = [law for law in laws if law not in ['법률', '법안'] and len(law) > 3]
    key_tokens.extend(laws)
    
    # 2. 정부 부처/기관명 (고유명사만)
    # "○○부", "○○청", "○○위원회" 등
    org_pattern = r'([가-힣]{2,}(?:부|청|위원회|처|원|실|본부|단|관|국))'
    orgs = re.findall(org_pattern, utterance)
    # 일반적인 단어 및 일반적인 위원회명 제외
    forbidden_orgs = {
        '위원회', '본부', '실', '관', '국', '법안심사소위원회', '소위원회',
        '전문위원회', '이사회', '위원회', '심의위원회'
    }
    orgs = [org for org in orgs if org not in forbidden_orgs and len(org) > 2]
    # 정부 부처명만 선택 (예: 교육부, 법무부 등)
    government_orgs = [
        '교육부', '법무부', '외교부', '국방부', '행정안전부', '문화체육관광부',
        '농림축산식품부', '산업통상자원부', '보건복지부', '환경부', '고용노동부',
        '여성가족부', '국토교통부', '해양수산부', '중소벤처기업부', '과학기술정보통신부',
        '기획재정부', '기획예산처', '국무조정실', '법제처', '인사혁신처',
        '공정거래위원회', '금융위원회', '국가보훈부', '식품의약품안전처',
        '경찰청', '국세청', '관세청', '방위사업청', '소방청'
    ]
    # 정부 부처명만 필터링
    orgs = [org for org in orgs if any(gov_org in org for gov_org in government_orgs)]
    key_tokens.extend(orgs)
    
    # 3. 날짜/연도/시한 (중요한 정보)
    date_pattern = r'(\d{4}년\s*\d{1,2}월\s*\d{1,2}일|\d{1,2}월\s*\d{1,2}일|\d{4}년)'
    dates = re.findall(date_pattern, utterance)
    key_tokens.extend(dates)
    
    # 4. 법조항 (제○조, 제○항, 제○호, 제○절)
    clause_pattern = r'(제\d+조|제\d+항|제\d+호|제\d+절)'
    clauses = re.findall(clause_pattern, utterance)
    key_tokens.extend(clauses)
    
    # 5. 의사일정 번호 (제○항)
    agenda_num_pattern = r'의사일정\s*(제\d+항)'
    agenda_nums = re.findall(agenda_num_pattern, utterance)
    key_tokens.extend(agenda_nums)
    
    # 금지 단어 필터링
    key_tokens = [token for token in key_tokens if token not in forbidden_words]
    
    # 일반적인 위원회명/기관명 제외 (더 확실하게)
    general_orgs = ['법안심사소위원회', '소위원회', '전문위원회', '이사회', '심의위원회']
    key_tokens = [token for token in key_tokens if not any(gen_org in token for gen_org in general_orgs)]
    
    # 중복 제거 및 정리
    key_tokens = list(set([token.strip() for token in key_tokens if len(token.strip()) > 1]))
    
    # 한 문장에 3개 이상이면 1-2개만 선택 (가장 중요한 것만)
    if len(key_tokens) > 2:
        # 법령명/기관명 우선, 그 다음 조항/날짜
        priority_order = []
        for token in key_tokens:
            if any(x in token for x in ['법', '부', '청', '위원회', '처']):
                priority_order.append(token)
        for token in key_tokens:
            if token not in priority_order and any(x in token for x in ['제', '년', '월', '일']):
                priority_order.append(token)
        
        # 우선순위가 있는 것만 선택, 없으면 처음 2개
        if priority_order:
            key_tokens = priority_order[:2]
        else:
            key_tokens = key_tokens[:2]
    
    return utterance, key_tokens


def apply_tags(
    utterance: str,
    agenda_title: Optional[str],
    key_tokens: List[str]
) -> str:
    """
    발화에 XML 태그를 적용합니다 (C단계 개선 버전).
    태그 우선순위: <결정> > <안건> > <쟁점> > <KEY>
    
    Args:
        utterance: 발화 텍스트
        agenda_title: 안건명
        key_tokens: 중요한 토큰 리스트
        
    Returns:
        태그가 적용된 발화
    """
    result = utterance
    
    # 1순위: <결정> 태그 적용
    decision_patterns = [
        r'([^\.。]*(?:가결|부결|통과|보류|의결|합의|결정|상정|처리|채택|부의|원안대로|수정가결|신설|폐지|확정|재논의)[^\.。]*)',
        r'([^\.。]*(?:의결하고자|의결하도록|의결되었습니다|가결되었습니다|가결되었음을 선포)[^\.。]*)',
        r'([^\.。]*(?:이의 없으시지요|이의 없으십니까|이의 없습니까)[^\.。]*)',
    ]
    
    for pattern in decision_patterns:
        matches = list(re.finditer(pattern, result))
        # 뒤에서부터 적용 (인덱스 문제 방지)
        for match in reversed(matches):
            matched_text = match.group(1).strip()
            # 이미 태그가 적용되지 않았고, 다른 태그도 없는 경우
            if '<결정>' not in matched_text and '<안건>' not in matched_text and '<쟁점>' not in matched_text:
                if len(matched_text) > 5:
                    start, end = match.span()
                    result = result[:start] + f'<결정>{matched_text}</결정>' + result[end:]
                    break
        if '<결정>' in result:
            break
    
    # 2순위: <안건> 태그 적용
    if agenda_title:
        # 안건명과 그 변형 찾기
        agenda_variants = [
            agenda_title,
            agenda_title.replace('일부개정법률안', '개정안'),
            agenda_title.replace('일부개정법률안', '법률안'),
            agenda_title.replace('법률안', '법'),
        ]
        
        agenda_count = 0
        for variant in agenda_variants:
            if variant in result and agenda_count < 2:  # 첫 2회만 태깅
                # 이미 태그가 적용되지 않았는지 확인
                if f'<안건>{variant}</안건>' not in result:
                    # <결정> 태그 내부가 아닌지 확인
                    pattern = re.escape(variant)
                    matches = list(re.finditer(pattern, result))
                    # 뒤에서부터 적용
                    for match in reversed(matches):
                        start, end = match.span()
                        # <결정> 태그 내부인지 확인
                        before = result[:start]
                        if before.count('<결정>') > before.count('</결정>'):
                            continue  # <결정> 태그 내부이면 스킵
                        
                        # 태그 적용
                        result = result[:start] + f'<안건>{variant}</안건>' + result[end:]
                        agenda_count += 1
                        break
                if agenda_count >= 2:
                    break
    
    # 3순위: <쟁점> 태그 적용
    issue_keywords = ['쟁점', '문제', '논란', '우려', '비판', '반대', '찬성', '갈등', '이견']
    issue_patterns = [
        r'([^\.。]*(?:쟁점은|문제는|우려되는 점은|논란이 되는 부분은|문제점이)[^\.。]*)',
        r'([^\.。]*(?:반대|찬성).*(?:근거|이유|사유)[^\.。]*)',
    ]
    
    # 패턴 매칭
    for pattern in issue_patterns:
        matches = list(re.finditer(pattern, result))
        # 뒤에서부터 적용
        for match in reversed(matches):
            matched_text = match.group(1).strip()
            if len(matched_text) > 10:
                if '<쟁점>' not in matched_text and '<결정>' not in matched_text:
                    start, end = match.span()
                    # <결정> 태그 내부가 아닌지 확인
                    before = result[:start]
                    if before.count('<결정>') <= before.count('</결정>'):
                        result = result[:start] + f'<쟁점>{matched_text}</쟁점>' + result[end:]
                        break
        if '<쟁점>' in result:
            break
    
    # 키워드 기반 쟁점 태깅 (패턴 매칭 실패 시)
    if '<쟁점>' not in result:
        for keyword in issue_keywords:
            if keyword in result:
                # 문장 단위로 분리
                sentences = re.split(r'([\.。])', result)
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i] if i < len(sentences) else ''
                    if keyword in sentence and len(sentence.strip()) > 10:
                        # <결정> 태그 내부가 아닌지 확인
                        if '<결정>' not in sentence and '<안건>' not in sentence:
                            sentences[i] = f'<쟁점>{sentence.strip()}</쟁점>'
                            result = ''.join(sentences)
                            break
                if '<쟁점>' in result:
                    break
    
    # 4순위: <KEY> 태그 적용 (오남용 방지)
    # 이미 다른 태그로 감싸진 부분은 제외
    key_count = 0
    max_key_tags = 2  # 한 문장에 최대 2개만
    
    for token in key_tokens:
        if key_count >= max_key_tags:
            break
        
        # 이미 태그가 적용된 부분은 제외
        if f'<KEY>{token}</KEY>' in result:
            continue
        
        # 다른 태그 내부인지 확인
        pattern = re.escape(token)
        matches = list(re.finditer(pattern, result))
        # 뒤에서부터 적용
        for match in reversed(matches):
            start, end = match.span()
            before = result[:start]
            
            # 다른 태그 내부인지 확인
            if (before.count('<결정>') > before.count('</결정>') or
                before.count('<안건>') > before.count('</안건>') or
                before.count('<쟁점>') > before.count('</쟁점>')):
                continue  # 다른 태그 내부이면 스킵
            
            # 금지 단어 체크
            forbidden_in_context = ['학생', '위원', '의원', '제도', '문제']
            if any(fw in token for fw in forbidden_in_context):
                # 맥락상 필수인지 확인 (법조항, 기관명 등과 함께 나오는 경우만)
                context = result[max(0, start-10):min(len(result), end+10)]
                if not any(x in context for x in ['제', '부', '청', '법']):
                    continue  # 맥락상 필수가 아니면 스킵
            
            # 태그 적용
            result = result[:start] + f'<KEY>{token}</KEY>' + result[end:]
            key_count += 1
            break
    
    return result


def preprocess_dialogue(
    participants: List[Dict],
    dialogue: List[Dict],
    agenda_title: Optional[str] = None
) -> str:
    """
    대화를 전처리하여 "[역할] 이름: 발화" 형식으로 변환합니다.
    
    Args:
        participants: 참가자 리스트
        dialogue: 대화 리스트
        agenda_title: 안건명 (없으면 대화에서 추출)
        
    Returns:
        전처리된 대화 텍스트 (줄바꿈으로 구분)
    """
    # 발화자 역할 매핑
    role_map = map_speaker_roles(participants)
    
    # 안건명 추출 (제공되지 않은 경우)
    if not agenda_title:
        agenda_title = extract_agenda_from_dialogue(dialogue)
    
    preprocessed_lines = []
    
    for utterance_obj in dialogue:
        speaker_raw = utterance_obj.get('speaker', '')
        utterance = utterance_obj.get('utterance', '')
        
        # A단계: 발화자 이름 정규화
        speaker = normalize_speaker_name(speaker_raw)
        if not speaker:
            speaker = speaker_raw  # 정규화 실패 시 원본 사용
        
        # A단계: 역할 가져오기 및 표준화
        role = role_map.get(speaker_raw, '비지정')
        if role == '비지정':
            # 발화자 이름으로 다시 시도
            role = role_map.get(speaker, '비지정')
        
        # B단계: 불필요한 발화 제거 (유지)
        if clean_utterance(utterance):
            continue
        
        # C단계: 중요한 정보 추출 및 태그 적용 (유지)
        _, key_tokens = extract_key_information(utterance)
        tagged_utterance = apply_tags(utterance, agenda_title, key_tokens)
        
        # A단계: 형식화 (고정 형식 강화)
        # 반드시 "[역할] 이름: 발화" 형식, 콜론 뒤 한 칸 공백
        formatted_line = f"[{role}] {speaker}: {tagged_utterance}"
        preprocessed_lines.append(formatted_line)
    
    return '\n'.join(preprocessed_lines)


def process_single_sample(sample: Dict) -> str:
    """
    단일 샘플을 처리합니다.
    
    Args:
        sample: 샘플 딕셔너리
        
    Returns:
        전처리된 대화 텍스트
    """
    input_data = sample.get('input', {})
    participants = input_data.get('speaker', [])
    dialogue = input_data.get('conversation', [])
    
    # 안건명 추출 시도
    agenda_title = extract_agenda_from_dialogue(dialogue)
    
    # 전처리 수행
    preprocessed_text = preprocess_dialogue(participants, dialogue, agenda_title)
    
    return preprocessed_text


def process_all_files(
    input_dir: str = '.',
    output_dir: str = 'preprocessed',
    file_names: List[str] = None
) -> None:
    """
    모든 데이터셋 파일을 처리합니다.
    
    Args:
        input_dir: 입력 디렉토리
        output_dir: 출력 디렉토리
        file_names: 처리할 파일명 리스트 (None이면 train/dev/test 자동 감지)
    """
    if file_names is None:
        file_names = [
            '국회회의록안건별요약_train.json',
            '국회회의록안건별요약_dev.json',
            '국회회의록안건별요약_test.json'
        ]
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        
        if not os.path.exists(input_path):
            print(f"경고: 파일을 찾을 수 없습니다: {input_path}")
            continue
        
        print(f"처리 중: {file_name}")
        
        # 데이터 로드
        samples = load_json_files(input_path)
        
        # 각 샘플 처리
        preprocessed_samples = []
        for i, sample in enumerate(samples):
            try:
                preprocessed_text = process_single_sample(sample)
                preprocessed_samples.append({
                    'id': sample.get('id', f'sample_{i}'),
                    'preprocessed_dialogue': preprocessed_text
                })
            except Exception as e:
                print(f"경고: 샘플 {i} 처리 중 오류 발생: {e}")
                continue
        
        # 출력 파일명 생성
        base_name = os.path.splitext(file_name)[0]
        output_path_txt = os.path.join(output_dir, f'{base_name}_preprocessed.txt')
        output_path_json = os.path.join(output_dir, f'{base_name}_preprocessed.json')
        
        # TXT 파일로 저장 (각 샘플을 구분자로 분리)
        with open(output_path_txt, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(preprocessed_samples):
                f.write(f"=== Sample {i+1} (ID: {sample['id']}) ===\n")
                f.write(sample['preprocessed_dialogue'])
                f.write('\n\n')
        
        # JSON 파일로 저장
        with open(output_path_json, 'w', encoding='utf-8') as f:
            json.dump(preprocessed_samples, f, ensure_ascii=False, indent=2)
        
        print(f"완료: {len(preprocessed_samples)}개 샘플 처리됨")
        print(f"  - TXT: {output_path_txt}")
        print(f"  - JSON: {output_path_json}")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='국회 회의록 요약 데이터셋 전처리')
    parser.add_argument('--input_dir', type=str, default='.', 
                       help='입력 디렉토리 경로')
    parser.add_argument('--output_dir', type=str, default='preprocessed',
                       help='출력 디렉토리 경로')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='처리할 파일명 리스트 (기본값: train/dev/test 자동 감지)')
    
    args = parser.parse_args()
    
    process_all_files(args.input_dir, args.output_dir, args.files)


if __name__ == '__main__':
    main()

