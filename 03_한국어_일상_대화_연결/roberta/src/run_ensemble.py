"""
Cross Validation 앙상블 추론 간편 실행 스크립트 (로컬용)
"""

import subprocess
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("="*60)
        print("Cross Validation 모델 앙상블 추론")
        print("="*60)
        print("\n사용법:")
        print("  python run_ensemble.py <JOB_ID> [옵션]")
        print("\n예시:")
        print("  python run_ensemble.py 58583")
        print("  python run_ensemble.py 58583 --output my_submission.json")
        print("  python run_ensemble.py 58583 --test_path custom_test.json")
        print("\n설명:")
        print("  JOB_ID: Cross Validation 학습 시 사용한 Job ID")
        print("          (models/best_model_*_fold*_<JOB_ID>.pt 형식의 파일을 찾습니다)")
        print("\n옵션:")
        print("  --output PATH      출력 파일 경로 (기본: submission/ensemble_cv_{JOB_ID}.json)")
        print("  --test_path PATH   테스트 데이터 경로 (기본: config.TEST_PATH)")
        print("  --model_dir PATH   모델 디렉토리 (기본: ./models)")
        print("="*60)
        sys.exit(1)
    
    # ensemble_inference_cv.py 실행
    args = sys.argv[1:]
    
    print("\n" + "="*60)
    print("앙상블 추론 시작")
    print("="*60)
    print(f"Job ID: {args[0]}")
    print()
    
    cmd = [sys.executable, 'ensemble_inference_cv.py'] + args
    
    result = subprocess.run(cmd)
    
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
