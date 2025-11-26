"""
메인 전처리 파이프라인 모듈

전처리 프로세스:
1. 안건 경계 탐지: sentence_id를 기준으로 안건의 시작과 끝을 찾음
2. 발화 정제: 불필요한 발화 제거 및 정제
3. 태깅 적용: <결정>, <안건>, <쟁점> 태그 적용
4. SimCSE 기반 중요도 태깅: <IMP> 태그 적용
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

try:
    from .config import MIN_UTTERANCE_LENGTH_STRICT
    from .utils import normalize_speaker_name, map_speaker_roles
    from .boundary_detection import extract_agenda_boundaries
    from .filtering import clean_utterance
    from .tagging import apply_tags
    from .importance_by_similarity import apply_importance_tags_by_similarity
except ImportError:
    from config import MIN_UTTERANCE_LENGTH_STRICT
    from utils import normalize_speaker_name, map_speaker_roles
    from boundary_detection import extract_agenda_boundaries
    from filtering import clean_utterance
    from tagging import apply_tags
    from importance_by_similarity import apply_importance_tags_by_similarity

logger = logging.getLogger(__name__)


def process_utterances(
    dialogue_segment: List[Dict],
    role_map: Dict[str, str],
    agenda_title: Optional[str],
    strict_filtering: bool = True
) -> Tuple[List[str], List[str], List[str]]:
    """
    발화 리스트를 처리하여 전처리된 라인과 원시 발화, 역할 리스트를 반환합니다.
    
    처리 과정:
    1. 발화자 이름 정규화
    2. 역할 매핑
    3. 불필요한 발화 제거 (strict_filtering=True인 경우)
    4. 태그 적용 (<결정>, <안건>, <쟁점>)
    5. 형식화: [역할] 이름: 발화내용
    
    Args:
        dialogue_segment: 처리할 발화 리스트
        role_map: 발화자 이름 -> 역할 매핑 딕셔너리
        agenda_title: 안건명 (태깅에 사용)
        strict_filtering: True면 엄격한 필터링 적용 (기본값: True)
    
    Returns:
        (전처리된 라인 리스트, 원시 발화 리스트, 역할 리스트) 튜플
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
        
        # 태그 적용
        tagged_utterance = apply_tags(utterance, agenda_title, role)
        
        # 형식화
        formatted_line = f"[{role}] {speaker}: {tagged_utterance}"
        preprocessed_lines.append(formatted_line)
        raw_utterances.append(tagged_utterance)
        roles.append(role)
    
    return preprocessed_lines, raw_utterances, roles


def apply_importance_tags_to_lines(
    preprocessed_lines: List[str],
    raw_utterances: List[str],
    summary: Optional[str] = None,
    threshold: float = 0.5
) -> List[str]:
    """
    SimCSE 기반 유사도로 중요도 태그를 라인에 적용합니다.
    
    Args:
        preprocessed_lines: 전처리된 발화 라인 리스트
        raw_utterances: 원시 발화 리스트 (태그 포함)
        summary: 정답 요약 텍스트 (없으면 규칙 기반으로 fallback)
        threshold: 중요 발화 임계값 (코사인 유사도)
    
    Returns:
        <IMP> 태그가 적용된 전처리된 라인 리스트
    """
    if len(raw_utterances) <= 1:
        return preprocessed_lines
    
    # 요약이 있으면 SimCSE 기반, 없으면 규칙 기반 fallback
    if summary:
        try:
            important_utterances = apply_importance_tags_by_similarity(
                raw_utterances, summary, threshold=threshold
            )
        except Exception as e:
            logger.warning(f"SimCSE 기반 태깅 실패, 규칙 기반으로 fallback: {e}")
            try:
                from .tagging import apply_importance_tags
            except ImportError:
                from tagging import apply_importance_tags
            important_utterances = apply_importance_tags(raw_utterances, roles=None)
    else:
        try:
            from .tagging import apply_importance_tags
        except ImportError:
            from tagging import apply_importance_tags
        important_utterances = apply_importance_tags(raw_utterances, roles=None)
    
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
    agenda_info: Optional[Dict] = None,
    summary: Optional[str] = None
) -> str:
    """
    통합 전처리 파이프라인
    
    Args:
        participants: 참가자 리스트
        dialogue: 전체 대화 리스트
        agenda_info: 안건 정보 (sentence_id, keyword, topic, next_sentence_id 등)
        summary: 정답 요약 텍스트 (SimCSE 기반 중요도 태깅에 사용)
    
    Returns:
        전처리된 대화 텍스트
    """
    # 안건 정보 추출
    sentence_id = agenda_info.get('sentence_id') if agenda_info else None
    keyword = agenda_info.get('keyword') if agenda_info else None
    topic = agenda_info.get('topic') if agenda_info else None
    next_sentence_id = agenda_info.get('next_sentence_id') if agenda_info else None
    
    # 안건명 추출
    agenda_title = keyword if keyword else topic
    
    # 발화자 역할 매핑
    role_map = map_speaker_roles(participants)
    
    # 1. 안건 경계 탐지
    start_idx, end_idx = extract_agenda_boundaries(
        dialogue, sentence_id, next_sentence_id=next_sentence_id
    )
    
    # 안건 경계 검증
    if start_idx >= end_idx or start_idx < 0 or end_idx > len(dialogue):
        logger.warning(f"잘못된 안건 경계: start_idx={start_idx}, end_idx={end_idx}, dialogue_len={len(dialogue)}")
        start_idx = max(0, start_idx)
        end_idx = min(len(dialogue), max(start_idx + 1, end_idx))
    
    # 2. 안건 경계 내 모든 발화 포함
    filtered_dialogue = dialogue[start_idx:end_idx]
    logger.info(f"안건 경계: {start_idx}~{end_idx} (총 {len(filtered_dialogue)}개 발화 포함)")
    
    # 3. 발화 정제 및 태깅
    preprocessed_lines, raw_utterances, roles = process_utterances(
        filtered_dialogue, role_map, agenda_title, strict_filtering=True
    )
    
    # 4. SimCSE 기반 중요도 태깅
    preprocessed_lines = apply_importance_tags_to_lines(preprocessed_lines, raw_utterances, summary=summary)
    
    # 최종 텍스트 조합
    preprocessed_text = '\n'.join(preprocessed_lines)
    
    # 빈 샘플 방지 (간단하게)
    if not preprocessed_text.strip():
        logger.warning(f"전처리 결과가 비어있음. 완화된 기준으로 재처리")
        preprocessed_lines, raw_utterances, roles = process_utterances(
            dialogue[start_idx:end_idx], role_map, agenda_title, strict_filtering=False
        )
        preprocessed_lines = apply_importance_tags_to_lines(preprocessed_lines, raw_utterances, summary=summary)
        preprocessed_text = '\n'.join(preprocessed_lines)
    
    return preprocessed_text


def analyze_summary_length(summary: str) -> Dict[str, int]:
    """
    요약의 길이를 분석합니다.
    
    Args:
        summary: 요약 텍스트
    
    Returns:
        {'char_count': 문자 수, 'word_count': 단어 수, 'sentence_count': 문장 수}
    """
    if not summary:
        return {'char_count': 0, 'word_count': 0, 'sentence_count': 0}
    
    char_count = len(summary)
    word_count = len(summary.split())
    
    # 문장 수 계산 (마침표, 느낌표, 물음표 기준)
    sentence_count = len(re.findall(r'[.!?。！？]\s*', summary))
    if sentence_count == 0 and char_count > 0:
        sentence_count = 1
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count
    }


def create_prompt_template(
    preprocessed_dialogue: str,
    output_summary: Optional[str] = None,
    include_tag_explanation: bool = True
) -> str:
    """
    요약 모델 입력을 위한 프롬프트 템플릿을 생성합니다.
    
    Args:
        preprocessed_dialogue: 전처리된 대화 텍스트
        output_summary: 정답 요약 텍스트 (길이 분석에 사용)
        include_tag_explanation: 태그 설명 포함 여부
    """
    prompt = "다음은 국회 회의록 안건별 대화입니다."
    
    if include_tag_explanation:
        prompt += "\n\n태그의 의미는 다음과 같습니다:\n"
        prompt += "- <IMP>: 요약에 중요한 발화 (전문위원 보고, 핵심 논의 등)\n"
        prompt += "- <결정>: 의사진행 결정사항 (가결, 부결, 의결, 상정 등)\n"
        prompt += "- <안건>: 안건명 (법안명)\n"
        prompt += "- <쟁점>: 논란/문제점/갈등 사항\n"
        
        prompt += "\n【중요】요약 작성 규칙:\n"
        prompt += "1. <IMP> 태그가 있는 발화는 요약에 우선적으로 포함해주세요.\n"
        prompt += "2. <결정> 태그가 있는 발화는 요약에 우선적으로 포함해주세요.\n"
        prompt += "3. <안건> 태그가 있는 내용은 요약에 우선적으로 포함해주세요.\n"
        prompt += "4. <쟁점> 태그가 있는 발화는 요약에 우선적으로 포함해주세요.\n"
        
        prompt += "\n요약 시 다음 정보를 반드시 포함해주세요:\n"
        prompt += "- 의사일정 번호 (예: 의사일정 제N항)\n"
        prompt += "- 법안명 (안건명)\n"
        prompt += "- 결정사항 (의결, 가결, 부결 등)\n"
        prompt += "- 전문위원 의견 (있는 경우)\n"
        prompt += "- 법조항 (언급된 경우)\n"
        prompt += "- 쟁점/문제점 (있는 경우)\n"
        
        # 원본 요약 길이 분석 및 지시
        if output_summary:
            summary_stats = analyze_summary_length(output_summary)
            char_count = summary_stats['char_count']
            word_count = summary_stats['word_count']
            sentence_count = summary_stats['sentence_count']
            
            prompt += f"\n【요약 길이 가이드】\n"
            prompt += f"원본 요약 기준: 약 {char_count}자, {word_count}단어, {sentence_count}문장 수준\n"
            prompt += f"위 길이를 참고하여 비슷한 수준으로 요약해주세요.\n"
    
    prompt += "\n대화:\n"
    prompt += preprocessed_dialogue
    prompt += "\n\n위 대화를 요약해주세요."
    
    return prompt
