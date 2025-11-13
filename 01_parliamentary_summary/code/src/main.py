"""
메인 진입점
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


def process_single_sample(sample: Dict) -> Dict:
    """
    단일 샘플을 처리합니다.
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
            'begin': issue.get('begin'),
            'end': issue.get('end'),
        }
        
        # 통합 전처리 수행
        preprocessed_text = preprocess_dialogue_integrated(
            participants, dialogue, agenda_info
        )
        
        # output (요약 결과) 추출
        output = sample.get('output', '')
        
        # 프롬프트 템플릿 생성
        prompt_text = create_prompt_template(preprocessed_text, include_tag_explanation=True)
        
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
    input_dir: str = None,
    output_dir: str = None,
    file_names: Optional[List[str]] = None
) -> None:
    """
    모든 데이터셋 파일을 처리합니다.
    """
    from .paths import RAW_DATA_DIR, PREPROCESSED_DATA_DIR, get_data_path
    
    if input_dir is None:
        input_dir = str(RAW_DATA_DIR)
    if output_dir is None:
        output_dir = str(PREPROCESSED_DATA_DIR)
    
    if file_names is None:
        file_names = [
            'parliamentary_summary_train.json',
            'parliamentary_summary_dev.json',
            'parliamentary_summary_test.json'
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
        
        preprocessed_samples = []
        error_count = 0
        
        for i, sample in enumerate(samples):
            try:
                result = process_single_sample(sample)
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
    from .paths import RAW_DATA_DIR, PREPROCESSED_DATA_DIR
    
    parser = argparse.ArgumentParser(description='국회 회의록 요약 데이터셋 통합 전처리')
    parser.add_argument('--input_dir', type=str, default=None, 
                       help=f'입력 디렉토리 경로 (기본값: {RAW_DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=None,
                       help=f'출력 디렉토리 경로 (기본값: {PREPROCESSED_DATA_DIR})')
    parser.add_argument('--files', type=str, nargs='+', default=None,
                       help='처리할 파일명 리스트 (기본값: train/dev/test 자동 감지)')
    
    args = parser.parse_args()
    process_all_files(args.input_dir, args.output_dir, args.files)


if __name__ == '__main__':
    main()

