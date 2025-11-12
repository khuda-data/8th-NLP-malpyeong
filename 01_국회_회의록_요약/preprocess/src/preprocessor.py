"""
메인 전처리 파이프라인 모듈
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .config import MIN_UTTERANCE_LENGTH_STRICT
    from .utils import normalize_speaker_name, map_speaker_roles
    from .boundary_detection import extract_agenda_boundaries
    from .filtering import clean_utterance, filter_utterances_by_keyword
    from .tagging import extract_key_information, apply_tags, apply_importance_tags
except ImportError:
    from config import MIN_UTTERANCE_LENGTH_STRICT
    from utils import normalize_speaker_name, map_speaker_roles
    from boundary_detection import extract_agenda_boundaries
    from filtering import clean_utterance, filter_utterances_by_keyword
    from tagging import extract_key_information, apply_tags, apply_importance_tags

logger = logging.getLogger(__name__)


def remove_duplicate_utterances(
    preprocessed_lines: List[str],
    raw_utterances: List[str],
    roles: List[str],
    similarity_threshold: float = 0.8
) -> Tuple[List[str], List[str], List[str]]:
    """
    반복 발화를 제거합니다.
    """
    if len(preprocessed_lines) <= 1:
        return preprocessed_lines, raw_utterances, roles
    
    def extract_content(line: str) -> str:
        match = re.match(r'\[.+\]\s*.+?:\s*(.+)', line)
        if match:
            content = match.group(1)
            content = re.sub(r'<[^>]+>', '', content)
            return content.strip()
        return line.strip()
    
    seen_contents = []
    filtered_lines = []
    filtered_utterances = []
    filtered_roles = []
    
    for i, (line, utterance, role) in enumerate(zip(preprocessed_lines, raw_utterances, roles)):
        content = extract_content(line)
        
        if len(content) < 10:
            filtered_lines.append(line)
            filtered_utterances.append(utterance)
            filtered_roles.append(role)
            continue
        
        is_duplicate = False
        for seen_content in seen_contents:
            seen_words = set(re.findall(r'\w+', seen_content))
            content_words = set(re.findall(r'\w+', content))
            
            if len(seen_words) == 0 or len(content_words) == 0:
                continue
            
            common_words = seen_words & content_words
            similarity = len(common_words) / max(len(seen_words), len(content_words))
            
            if similarity >= similarity_threshold:
                if abs(len(seen_content) - len(content)) / max(len(seen_content), len(content)) < 0.3:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_contents.append(content)
            filtered_lines.append(line)
            filtered_utterances.append(utterance)
            filtered_roles.append(role)
    
    return filtered_lines, filtered_utterances, filtered_roles


def process_utterances(
    dialogue_segment: List[Dict],
    role_map: Dict[str, str],
    agenda_title: Optional[str],
    strict_filtering: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    발화 리스트를 처리하여 전처리된 라인과 원시 발화, 역할 리스트를 반환합니다.
    """
    preprocessed_lines = []
    raw_utterances = []
    roles = []
    
    for utterance_obj in dialogue_segment:
        speaker_raw = utterance_obj.get('speaker', '')
        utterance = utterance_obj.get('utterance', '')
        
        # 발화자 이름 정규화
        speaker = normalize_speaker_name(speaker_raw)
        if not speaker:
            speaker = speaker_raw
        
        # 역할 가져오기
        role = role_map.get(speaker_raw, '비지정')
        if role == '비지정':
            role = role_map.get(speaker, '비지정')
        
        # 불필요한 발화 제거
        if strict_filtering:
            if len(dialogue_segment) <= 5:
                if clean_utterance(utterance) and len(utterance.strip()) <= 5:
                    continue
            else:
                if clean_utterance(utterance):
                    continue
        else:
            if len(utterance.strip()) <= MIN_UTTERANCE_LENGTH_STRICT:
                continue
        
        # 중요한 정보 추출 및 태그 적용
        _, key_tokens = extract_key_information(utterance)
        tagged_utterance = apply_tags(utterance, agenda_title, key_tokens, role)
        
        # 형식화
        formatted_line = f"[{role}] {speaker}: {tagged_utterance}"
        preprocessed_lines.append(formatted_line)
        raw_utterances.append(tagged_utterance)
        roles.append(role)
    
    return preprocessed_lines, raw_utterances, roles


def apply_importance_tags_to_lines(
    preprocessed_lines: List[str],
    raw_utterances: List[str],
    roles: List[str] = None
) -> List[str]:
    """
    규칙 기반 중요도 태그를 라인에 적용합니다.
    """
    if len(raw_utterances) <= 1:
        return preprocessed_lines
    
    important_utterances = apply_importance_tags(raw_utterances, roles)
    
    final_lines = []
    for i, line in enumerate(preprocessed_lines):
        if '<IMP>' in important_utterances[i]:
            if '<IMP>' not in line:
                match = re.match(r'(\[.+\]\s*.+?:\s*)(.+)', line)
                if match:
                    prefix = match.group(1)
                    new_utterance = important_utterances[i]
                    final_lines.append(prefix + new_utterance)
                else:
                    final_lines.append(line)
            else:
                final_lines.append(line)
        else:
            final_lines.append(line)
    
    return final_lines


def preprocess_dialogue_integrated(
    participants: List[Dict],
    dialogue: List[Dict],
    agenda_info: Optional[Dict] = None
) -> str:
    """
    통합 전처리 파이프라인
    
    Args:
        participants: 참가자 리스트
        dialogue: 전체 대화 리스트
        agenda_info: 안건 정보 (sentence_id, keyword, topic, begin, end 등)
    
    Returns:
        전처리된 대화 텍스트
    """
    # 안건 정보 추출
    sentence_id = agenda_info.get('sentence_id') if agenda_info else None
    keyword = agenda_info.get('keyword') if agenda_info else None
    topic = agenda_info.get('topic') if agenda_info else None
    begin = agenda_info.get('begin') if agenda_info else None
    end = agenda_info.get('end') if agenda_info else None
    
    # 안건명 추출
    agenda_title = keyword if keyword else topic
    
    # 발화자 역할 매핑
    role_map = map_speaker_roles(participants)
    
    # 1. 안건 경계 탐지
    start_idx, end_idx = extract_agenda_boundaries(dialogue, sentence_id, keyword, begin, end, role_map)
    
    # 안건 경계 검증
    if start_idx >= end_idx or start_idx < 0 or end_idx > len(dialogue):
        logger.warning(f"잘못된 안건 경계: start_idx={start_idx}, end_idx={end_idx}, dialogue_len={len(dialogue)}")
        start_idx = max(0, start_idx)
        end_idx = min(len(dialogue), max(start_idx + 1, end_idx))
    
    # 2. 안건 경계 내 모든 발화 포함
    filtered_dialogue = dialogue[start_idx:end_idx]
    logger.info(f"안건 경계: {start_idx}~{end_idx} (총 {len(filtered_dialogue)}개 발화 포함, end_idx는 제외)")
    
    # 3. 발화 정제 및 태깅
    preprocessed_lines, raw_utterances, roles = process_utterances(
        filtered_dialogue, role_map, agenda_title, strict_filtering=True
    )
    
    # 3-1. 반복 발화 제거
    preprocessed_lines, raw_utterances, roles = remove_duplicate_utterances(
        preprocessed_lines, raw_utterances, roles, similarity_threshold=0.8
    )
    
    # 4. 규칙 기반 중요도 태깅
    preprocessed_lines = apply_importance_tags_to_lines(preprocessed_lines, raw_utterances, roles)
    
    # 최종 텍스트 조합
    preprocessed_text = '\n'.join(preprocessed_lines)
    
    # 빈 샘플 방지
    if not preprocessed_text.strip():
        logger.warning(f"전처리 결과가 비어있음. 완화된 기준으로 재처리")
        
        preprocessed_lines, raw_utterances, roles = process_utterances(
            dialogue[start_idx:end_idx], role_map, agenda_title, strict_filtering=False
        )
        preprocessed_lines = apply_importance_tags_to_lines(preprocessed_lines, raw_utterances, roles)
        preprocessed_text = '\n'.join(preprocessed_lines)
        
        if not preprocessed_text.strip():
            logger.warning(f"완화된 기준으로도 빈 결과. 최소 정제만 수행.")
            preprocessed_lines = []
            raw_utterances = []
            roles = []
            
            for utterance_obj in dialogue[start_idx:end_idx]:
                speaker_raw = utterance_obj.get('speaker', '')
                utterance = utterance_obj.get('utterance', '').strip()
                
                if len(utterance) <= 1:
                    continue
                
                speaker = normalize_speaker_name(speaker_raw)
                if not speaker:
                    speaker = speaker_raw
                
                role = role_map.get(speaker_raw, '비지정')
                if role == '비지정':
                    role = role_map.get(speaker, '비지정')
                
                # 태그 적용
                _, key_tokens = extract_key_information(utterance)
                tagged_utterance = apply_tags(utterance, agenda_title, key_tokens, role)
                
                formatted_line = f"[{role}] {speaker}: {tagged_utterance}"
                preprocessed_lines.append(formatted_line)
                raw_utterances.append(tagged_utterance)
                roles.append(role)
        
        if len(raw_utterances) > 1:
            preprocessed_lines = apply_importance_tags_to_lines(preprocessed_lines, raw_utterances, roles)
            preprocessed_text = '\n'.join(preprocessed_lines)
            
            if not preprocessed_text.strip():
                logger.error(f"모든 정제 후에도 빈 결과. 원본 발화 그대로 사용.")
                preprocessed_lines = []
                for utterance_obj in dialogue[start_idx:end_idx]:
                    speaker_raw = utterance_obj.get('speaker', '')
                    utterance = utterance_obj.get('utterance', '').strip()
                    if utterance:
                        speaker = normalize_speaker_name(speaker_raw) or speaker_raw
                        role = role_map.get(speaker_raw, '비지정') or role_map.get(speaker, '비지정')
                        preprocessed_lines.append(f"[{role}] {speaker}: {utterance}")
                preprocessed_text = '\n'.join(preprocessed_lines)
    
    return preprocessed_text


def create_prompt_template(preprocessed_dialogue: str, include_tag_explanation: bool = True) -> str:
    """
    LLM 입력을 위한 프롬프트 템플릿을 생성합니다.
    """
    prompt = "다음은 국회 회의록 안건별 대화입니다."
    
    if include_tag_explanation:
        prompt += "\n\n태그의 의미는 다음과 같습니다:\n"
        prompt += "- <IMP>: 요약에 중요한 발화 (전문위원 보고, 핵심 논의 등)\n"
        prompt += "- <결정>: 의사진행 결정사항 (가결, 부결, 의결, 상정 등)\n"
        prompt += "- <KEY>: 중요한 키워드 (법령명, 정부 부처명, 법조항, 날짜, 의사일정 등)\n"
        prompt += "- <안건>: 안건명 (법안명)\n"
        prompt += "- <쟁점>: 논란/문제점/갈등 사항\n"
        
        prompt += "\n요약 시 다음 정보를 반드시 포함해주세요:\n"
        prompt += "- 의사일정 번호 (예: 의사일정 제N항)\n"
        prompt += "- 법안명 (안건명)\n"
        prompt += "- 결정사항 (의결, 가결, 부결 등)\n"
        prompt += "- 전문위원 의견 (있는 경우)\n"
        prompt += "- 법조항 (언급된 경우)\n"
        prompt += "- 쟁점/문제점 (있는 경우)\n"
        prompt += "\n위 태그를 참고하여 핵심 정보를 포함한 요약을 작성해주세요.\n"
    
    prompt += "\n대화:\n"
    prompt += preprocessed_dialogue
    prompt += "\n\n위 대화를 요약해주세요."
    
    return prompt

