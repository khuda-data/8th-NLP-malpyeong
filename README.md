# 8th-NLP-malpyeong

전체 프로젝트 소개

## 프로젝트 구성

- [01_국회_회의록_요약](./01_국회_회의록_요약/) - 국회 회의록 요약 프로젝트
- [02_그림_기반_문장_생성](./02_그림_기반_문장_생성/) - 그림 기반 문장 생성 프로젝트
- [03_한국어_일상_대화_연결](./03_한국어_일상_대화_연결/) - 한국어 일상 대화 연결 프로젝트

## 참고 자료

- [docs](./docs/) - 참고자료, 공유하면 좋은 자료 논문 첨부 해주시면 감사할게요~

## Git 브랜치 전략

### 브랜치 구조

- `main` - 메인 브랜치 (최종 배포용)
- `dev/parlsum` - 국회 회의록 요약 팀 개발 브랜치
- `dev/img2txt` - 그림 기반 문장 생성 팀 개발 브랜치
- `dev/kochats` - 한국어 일상 대화 연결 팀 개발 브랜치

### 작업 흐름

1. 각 팀원들은 자신의 팀 dev 브랜치에서 작업
   ```bash
   git checkout dev/parlsum 
   ```

2. 작업 완료 후 팀 dev 브랜치에 merge
   ```bash
   git checkout dev/parlsum
   git merge feature/your-feature
   ```

3. 최종적으로 main 브랜치에 merge
   ```bash
   git checkout main
   git merge dev/parlsum
   ```

