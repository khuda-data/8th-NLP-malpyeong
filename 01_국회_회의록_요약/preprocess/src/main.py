"""
메인 진입점

전처리 파이프라인 실행:
1. JSON 파일 로드
2. 같은 dialogue 내 다음 샘플의 sentence_id 매핑 생성
3. 각 샘플에 대해 전처리 수행
4. 결과를 TXT 및 JSON 형식으로 저장
"""

import json
import os
import logging
import argparse
from typing import Dict, List, Optional

from .utils import load_json_files
from .preprocessor import preprocess_dialogue_integrated, create_prompt_template

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _build_next_sentence_id_map(samples: List[Dict]) -> Dict[str, str]:
    """
    같은 dialogue 내 다음 샘플의 sentence_id 매핑을 생성합니다.
    
    같은 dialogue에 여러 안건이 있을 때, 각 안건의 다음 안건 시작점을
    정확하게 찾기 위해 사용됩니다. (1순위 기준)
    
    Returns:
        {sample_id: next_sentence_id} 딕셔너리
        예: {'sample-1': 'SBRW2100000295.1.1.3', 'sample-2': 'SBRW2100000295.1.1.4'}
    """
    next_sentence_id_map = {}
    
    # dialogue별로 그룹화
    dialogue_map = {}
    for sample in samples:
        dialogue = sample.get('input', {}).get('conversation', [])
        if not dialogue:
            continue
        dialogue_key = (len(dialogue), dialogue[0].get('id'), dialogue[-1].get('id'))
        if dialogue_key not in dialogue_map:
            dialogue_map[dialogue_key] = []
        dialogue_map[dialogue_key].append(sample)
    
    # 각 dialogue 내에서 다음 샘플 찾기
    for dialogue_key, dialogue_samples in dialogue_map.items():
        if len(dialogue_samples) <= 1:
            continue
        
        dialogue = dialogue_samples[0].get('input', {}).get('conversation', [])
        
        # 각 샘플의 인덱스 찾기
        sample_indices = []
        for sample in dialogue_samples:
            issue = sample.get('input', {}).get('issue', {})
            sentence_id = issue.get('sentence_id')
            if not sentence_id:
                continue
            
            for i, utt in enumerate(dialogue):
                if utt.get('id') == sentence_id:
                    sample_indices.append({
                        'sample_id': sample.get('id'),
                        'idx': i,
                        'sentence_id': sentence_id
                    })
                    break
        
        sample_indices.sort(key=lambda x: x['idx'])
        
        # 각 샘플에 대해 다음 샘플의 sentence_id 매핑
        for i in range(len(sample_indices) - 1):
            current = sample_indices[i]
            next_sample = sample_indices[i + 1]
            next_sentence_id_map[current['sample_id']] = next_sample['sentence_id']
    
    return next_sentence_id_map


def process_single_sample(sample: Dict, next_sentence_id: Optional[str] = None) -> Dict:
    """
    단일 샘플을 처리합니다.
    
    처리 과정:
    1. 입력 데이터 추출 (participants, dialogue, issue)
    2. 안건 정보 추출 (sentence_id, keyword, topic, next_sentence_id)
    3. 전처리 수행 (경계 탐지, 필터링, 태깅)
    4. 프롬프트 템플릿 생성
    5. 결과 반환
    
    Args:
        sample: 처리할 샘플 딕셔너리
        next_sentence_id: 같은 dialogue 내 다음 샘플의 sentence_id (있으면 최우선 사용)
    
    Returns:
        전처리된 샘플 딕셔너리 (id, preprocessed_dialogue, prompt, output, agenda_info)
    """
    try:
        input_data = sample.get('input', {})
        if not input_data:
            raise ValueError("'input' 필드가 없거나 비어있습니다")
        
        participants = input_data.get('speaker', [])
        dialogue = input_data.get('conversation', [])
        
        if not dialogue:
            logger.warning(f"샘플 {sample.get('id', 'unknown')}: 대화 데이터가 비어있습니다")
        
        # 안건 정보 추출
        issue = input_data.get('issue', {})
        agenda_info = {
            'sentence_id': issue.get('sentence_id'),
            'keyword': issue.get('keyword'),
            'topic': issue.get('topic'),
            'next_sentence_id': next_sentence_id,  # 같은 dialogue 내 다음 샘플의 sentence_id
        }
        
        # output (요약 결과) 추출 (전처리 전에 추출하여 SimCSE 기반 태깅에 사용)
        output = sample.get('output', '')
        
        # 통합 전처리 수행
        preprocessed_text = preprocess_dialogue_integrated(
            participants, dialogue, agenda_info, summary=output
        )
        
        # 프롬프트 템플릿 생성 (output을 전달하여 길이 분석에 사용)
        prompt_text = create_prompt_template(preprocessed_text, output_summary=output, include_tag_explanation=True)
        
        return {
            'id': sample.get('id', ''),
            'preprocessed_dialogue': preprocessed_text,
            'prompt': prompt_text,
            'output': output,
            'agenda_info': agenda_info
        }
    except Exception as e:
        logger.error(f"샘플 처리 중 오류 발생 (ID: {sample.get('id', 'unknown')}): {e}")
        raise


def process_all_files(
    input_dir: str = '.',
    output_dir: str = 'output',
    file_names: Optional[List[str]] = None
) -> None:
    """
    모든 데이터셋 파일을 처리합니다.
    
    처리 과정:
    1. 각 파일 로드
    2. 같은 dialogue 내 다음 샘플의 sentence_id 매핑 생성
    3. 각 샘플 전처리
    4. 결과를 TXT 및 JSON 형식으로 저장
    
    Args:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        file_names: 처리할 파일명 리스트 (None이면 train/dev/test 자동 처리)
    """
    if file_names is None:
        file_names = [
            '국회회의록안건별요약_train.json',
            '국회회의록안건별요약_dev.json',
            '국회회의록안건별요약_test.json'
        ]
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"출력 디렉토리 생성 실패: {e}")
        raise
    
    for file_name in file_names:
        input_path = os.path.join(input_dir, file_name)
        
        if not os.path.exists(input_path):
            logger.warning(f"파일을 찾을 수 없습니다: {input_path}")
            continue
        
        logger.info(f"처리 중: {file_name}")
        
        try:
            samples = load_json_files(input_path)
        except Exception as e:
            logger.error(f"파일 로드 실패 ({file_name}): {e}")
            continue
        
        if not samples:
            logger.warning(f"파일이 비어있습니다: {file_name}")
            continue
        
        # 같은 dialogue 내 다음 샘플의 sentence_id 매핑 생성 (100% 정확도 기준)
        next_sentence_id_map = _build_next_sentence_id_map(samples)
        
        preprocessed_samples = []
        error_count = 0
        
        for i, sample in enumerate(samples):
            try:
                # 같은 dialogue 내 다음 샘플의 sentence_id 가져오기 (최우선 기준)
                next_sentence_id = next_sentence_id_map.get(sample.get('id'))
                result = process_single_sample(sample, next_sentence_id=next_sentence_id)
                preprocessed_samples.append({
                    'id': result['id'],
                    'preprocessed_dialogue': result['preprocessed_dialogue'],
                    'prompt': result.get('prompt', ''),
                    'output': result.get('output', '')
                })
            except Exception as e:
                error_count += 1
                logger.warning(f"샘플 {i} 처리 중 오류 발생 (ID: {sample.get('id', 'unknown')}): {e}")
                continue
        
        if error_count > 0:
            logger.warning(f"{file_name}: {error_count}개 샘플 처리 실패")
        
        if not preprocessed_samples:
            logger.warning(f"{file_name}: 처리된 샘플이 없습니다")
            continue
        
        base_name = os.path.splitext(file_name)[0]
        output_path_txt = os.path.join(output_dir, f'{base_name}_preprocessed.txt')
        output_path_json = os.path.join(output_dir, f'{base_name}_preprocessed.json')
        
        try:
            with open(output_path_txt, 'w', encoding='utf-8') as f:
                for i, sample in enumerate(preprocessed_samples):
                    f.write(f"=== Sample {i+1} (ID: {sample['id']}) ===\n")
                    f.write(sample['preprocessed_dialogue'])
                    f.write('\n\n')
            
            with open(output_path_json, 'w', encoding='utf-8') as f:
                json.dump(preprocessed_samples, f, ensure_ascii=False, indent=2)
            
            logger.info(f"완료: {len(preprocessed_samples)}개 샘플 처리됨")
            logger.info(f"  - TXT: {output_path_txt}")
            logger.info(f"  - JSON: {output_path_json}")
        except OSError as e:
            logger.error(f"파일 저장 실패 ({file_name}): {e}")
            continue


def main():
    """메인 함수"""
    # 기본 입력 디렉토리: 상위 폴더의 data 디렉토리
    default_input_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    parser = argparse.ArgumentParser(description='국회 회의록 요약 데이터셋 통합 전처리')
    parser.add_argument('--input_dir', type=str, default=default_input_dir, 
                       help=f'입력 디렉토리 경로 (기본값: {default_input_dir})')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='출력 디렉토리 경로')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='처리할 파일명 리스트 (기본값: train/dev/test 자동 감지)')
    
    args = parser.parse_args()
    process_all_files(args.input_dir, args.output_dir, args.files)


if __name__ == '__main__':
    main()

