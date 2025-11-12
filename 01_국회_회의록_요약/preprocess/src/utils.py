"""
유틸리티 함수 모음
"""

import json
import re
import logging
from typing import Dict, List, Optional

try:
    from .config import GOVERNMENT_ORGS
except ImportError:
    from config import GOVERNMENT_ORGS

logger = logging.getLogger(__name__)


def load_json_files(data_path: str) -> List[Dict]:
    """
    JSON 파일을 로드합니다.
    
    Args:
        data_path: JSON 파일 경로
    
    Returns:
        로드된 데이터 리스트
    
    Raises:
        FileNotFoundError: 파일을 찾을 수 없을 때
        json.JSONDecodeError: JSON 파싱 오류
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {data_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON 파싱 오류 ({data_path}): {e}")
        raise


def normalize_role(occupation: str) -> str:
    """역할(occupation)을 표준화합니다."""
    if not occupation:
        return '비지정'
    
    role = occupation.strip()
    role = re.sub(r'님$', '', role)
    
    role_mapping = {
        '위원장': '소위원장',
        '위원장님': '소위원장',
        '소위원장님': '소위원장',
    }
    
    if role in role_mapping:
        role = role_mapping[role]
    
    return role


def normalize_speaker_name(speaker: str) -> str:
    """발화자 이름을 정규화합니다."""
    if not speaker:
        return ''
    
    name = speaker.strip()
    name = re.sub(r'님$', '', name)
    
    korean_name_pattern = r'^[가-힣]{2,4}'
    match = re.match(korean_name_pattern, name)
    if match:
        name = match.group(0)
    else:
        name = re.sub(r'[^\w가-힣\s]', '', name).strip()
    
    return name


def extract_government_org(occupation: str) -> Optional[str]:
    """occupation에서 정부 기관명을 추출합니다."""
    if not occupation:
        return None
    
    for org in GOVERNMENT_ORGS:
        if org in occupation:
            return org
    
    return None


def infer_role_from_occupation(occupation: str, speaker_id: str) -> str:
    """역할을 추론합니다."""
    if occupation:
        org = extract_government_org(occupation)
        if org:
            return org
        return normalize_role(occupation)
    
    # speaker_id에서 정부 기관명 찾기
    for org in GOVERNMENT_ORGS:
        if org in speaker_id:
            return org
    
    if '위원' in speaker_id or '의원' in speaker_id:
        return '위원'
    
    return '비지정'


def map_speaker_roles(participants: List[Dict]) -> Dict[str, str]:
    """발화자 이름을 역할로 매핑합니다."""
    role_map = {}
    for participant in participants:
        speaker_id = participant.get('id', '')
        occupation = participant.get('occupation', '')
        role = infer_role_from_occupation(occupation, speaker_id)
        role_map[speaker_id] = role
    return role_map

