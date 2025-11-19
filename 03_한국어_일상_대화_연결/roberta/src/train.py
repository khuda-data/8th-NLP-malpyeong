"""
학습 스크립트
- 선행문/후행문 모델 각각 학습
- Early Stopping, Learning Rate Scheduler 포함
- Validation 평가 및 Best Model 저장
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

from data_loader import DialogueDataset, load_data_for_position, load_data_for_cross_validation
from model import DialogueConnectionModel, save_model
from config import Config


class EarlyStopping:
    """Early Stopping 클래스"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: 개선이 없을 때 기다리는 epoch 수
            min_delta: 개선으로 간주할 최소 변화량
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Returns:
            should_stop: True면 학습 중단
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def compute_metrics(predictions, labels):
    """평가 지표 계산"""
    # -1 레이블 제거 (test 데이터)
    mask = labels != -1
    predictions = predictions[mask]
    labels = labels[mask]
    
    if len(labels) == 0:
        return {}
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """1 epoch 학습"""
    model.train()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Training Epoch {epoch}')
    
    for batch in progress_bar:
        # 데이터 이동
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # 통계 수집
        total_loss += loss.item()
        all_predictions.extend(outputs['predictions'].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Progress bar 업데이트
        progress_bar.set_postfix({'loss': loss.item()})
    
    # 평균 손실 및 메트릭 계산
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


def validate(model, dataloader, device, epoch):
    """검증 평가"""
    model.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Validation Epoch {epoch}')
    
    with torch.no_grad():
        for batch in progress_bar:
            # 데이터 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # 통계 수집
            total_loss += loss.item()
            all_predictions.extend(outputs['predictions'].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Progress bar 업데이트
            progress_bar.set_postfix({'loss': loss.item()})
    
    # 평균 손실 및 메트릭 계산
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(np.array(all_predictions), np.array(all_labels))
    
    return avg_loss, metrics


def train_model_for_position(
    position: str,
    train_path: str,
    dev_path: str,
    model_name: str = 'klue/roberta-large',
    max_length: int = 128,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 30,
    patience: int = 5,
    output_dir: str = './models',
    device: str = None,
    resume_from: str = None
):
    """특정 발화위치에 대한 모델 학습
    
    Args:
        position: '선행문' 또는 '후행문'
        train_path: 학습 데이터 경로
        dev_path: 검증 데이터 경로
        model_name: Pretrained 모델 이름
        max_length: 최대 시퀀스 길이
        batch_size: 배치 크기
        learning_rate: 학습률
        num_epochs: 최대 epoch 수
        patience: Early stopping patience
        output_dir: 모델 저장 디렉토리
        device: 디바이스
        resume_from: 이어서 학습할 체크포인트 경로 (예: './models/best_model_선행문.pt')
    
    Returns:
        best_model_path: 최고 성능 모델 경로
    """
    # Config에서 JOB_ID 가져오기
    from config import Config
    cfg = Config()
    
    # 디바이스 설정
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"Training model for: {position}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Tokenizer 로드
    print("Loading tokenizer...")
    # 로컬 경로인지 확인
    if os.path.exists(model_name) and os.path.isdir(model_name):
        print(f"  Loading from local path: {os.path.abspath(model_name)}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    else:
        print(f"  Downloading from HuggingFace: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 데이터 로드
    print("Loading data...")
    
    # Cross Validation 사용 여부 확인
    if cfg.USE_CROSS_VALIDATION:
        print(f"Using {cfg.N_FOLDS}-Fold Cross Validation (Fold {cfg.CURRENT_FOLD + 1}/{cfg.N_FOLDS})")
        
        # connect_sentence.json 경로 설정 (이미 config에 절대 경로로 설정됨)
        connect_path = cfg.CONNECT_SENTENCE_PATH
        if connect_path and connect_path.strip() and os.path.exists(connect_path):
            print(f"Using additional data: {connect_path}")
        else:
            if connect_path and connect_path.strip():
                print(f"Warning: connect_sentence.json not found at {connect_path}")
            else:
                print(f"connect_sentence.json not configured (using TRAIN_PATH + DEV_PATH only)")
            connect_path = None
        
        train_loader, dev_loader = load_data_for_cross_validation(
            train_path, dev_path, tokenizer, position, 
            fold=cfg.CURRENT_FOLD,
            n_folds=cfg.N_FOLDS,
            max_length=max_length, 
            batch_size=batch_size,
            seed=cfg.CV_SEED,
            connect_sentence_path=connect_path
        )
    else:
        print("Using standard train/dev split")
        
        # connect_sentence.json 경로 설정 (이미 config에 절대 경로로 설정됨)
        connect_path = cfg.CONNECT_SENTENCE_PATH
        if connect_path and connect_path.strip() and os.path.exists(connect_path):
            print(f"Using additional data: {connect_path}")
        else:
            if connect_path and connect_path.strip():
                print(f"Warning: connect_sentence.json not found at {connect_path}")
            else:
                print(f"connect_sentence.json not configured (using TRAIN_PATH + DEV_PATH only)")
            connect_path = None
        
        train_loader, dev_loader = load_data_for_position(
            train_path, dev_path, tokenizer, position, max_length, batch_size,
            connect_sentence_path=connect_path
        )
    
    # 모델 초기화
    print("Initializing model...")
    # 로컬 경로인지 확인
    if os.path.exists(model_name) and os.path.isdir(model_name):
        model = DialogueConnectionModel(model_name=model_name, dropout_rate=0.3, local_model_path=model_name)
    else:
        model = DialogueConnectionModel(model_name=model_name, dropout_rate=0.3)
    
    # Special tokens에 맞게 embedding 확장
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    model.to(device)
    
    # Optimizer 및 Scheduler 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 체크포인트에서 이어서 학습
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        print(f"\n{'='*60}")
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"{'='*60}\n")
        
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        print(f"  - Previous val accuracy: {checkpoint.get('val_accuracy', 0):.4f}")
        print(f"  - Previous val F1: {checkpoint.get('val_f1', 0):.4f}")
        print(f"  - Resuming from epoch {start_epoch}\n")
    
    # Early stopping 초기화
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 학습 기록
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    best_val_accuracy = 0.0
    best_model_path = None
    
    # 학습 시작
    print("\nStarting training...")
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # 학습
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        
        # 검증
        val_loss, val_metrics = validate(model, dev_loader, device, epoch)
        
        # 결과 출력
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_metrics.get('accuracy', 0):.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics.get('accuracy', 0):.4f}, Val F1: {val_metrics.get('f1', 0):.4f}")
        
        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_metrics.get('accuracy', 0))
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics.get('accuracy', 0))
        history['val_f1'].append(val_metrics.get('f1', 0))
        
        # Best model 저장
        val_accuracy = val_metrics.get('accuracy', 0)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # 파일명에 JOB_ID, FOLD 포함
            job_id_suffix = f"_{cfg.JOB_ID}" if cfg.JOB_ID else ""
            fold_suffix = f"_fold{cfg.CURRENT_FOLD}" if cfg.USE_CROSS_VALIDATION else ""
            best_model_path = os.path.join(
                output_dir, 
                f'best_model_{position}{fold_suffix}{job_id_suffix}.pt'
            )
            save_model(
                model, optimizer, epoch, best_model_path,
                val_accuracy=val_accuracy,
                val_f1=val_metrics.get('f1', 0),
                val_loss=val_loss
            )
            print(f"✓ Best model saved! (Val Acc: {val_accuracy:.4f})")
        
        # Early stopping 체크
        if early_stopping(val_accuracy, epoch):
            print(f"\n⚠ Early stopping triggered at epoch {epoch}")
            print(f"Best epoch was {early_stopping.best_epoch} with accuracy {early_stopping.best_score:.4f}")
            break
    
    # 학습 기록 저장
    job_id_suffix = f"_{cfg.JOB_ID}" if cfg.JOB_ID else ""
    fold_suffix = f"_fold{cfg.CURRENT_FOLD}" if cfg.USE_CROSS_VALIDATION else ""
    history_path = os.path.join(output_dir, f'history_{position}{fold_suffix}{job_id_suffix}.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Training completed for {position}")
    print(f"Best model: {best_model_path}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"{'='*60}\n")
    
    return best_model_path


def main():
    """메인 함수 - 선행문과 후행문 모델을 각각 학습"""
    
    # config.py에서 설정 가져오기
    config = Config()
    
    print("="*60)
    print("대화 연결 모델 학습")
    print("="*60)
    print("\n설정:")
    print(f"  train_path: {config.TRAIN_PATH}")
    print(f"  dev_path: {config.DEV_PATH}")
    print(f"  model_name: {config.MODEL_NAME}")
    print(f"  max_length: {config.MAX_LENGTH}")
    print(f"  batch_size: {config.BATCH_SIZE}")
    print(f"  learning_rate: {config.LEARNING_RATE}")
    print(f"  num_epochs: {config.NUM_EPOCHS}")
    print(f"  patience: {config.PATIENCE}")
    print(f"  output_dir: {config.OUTPUT_DIR}")
    print(f"  resume_prev: {config.RESUME_PREV}")
    print(f"  resume_next: {config.RESUME_NEXT}")
    print()
    
    # GPU 체크
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠ CPU 모드로 실행됩니다.\n")
    
    # 1. 선행문 모델 학습
    print("\n" + "="*60)
    print("1. 선행문 모델 학습 시작")
    print("="*60)
    
    prev_model_path = train_model_for_position(
        position='선행문',
        train_path=config.TRAIN_PATH,
        dev_path=config.DEV_PATH,
        model_name=config.MODEL_NAME,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        output_dir=config.OUTPUT_DIR,
        device=device,
        resume_from=config.RESUME_PREV
    )
    
    # 2. 후행문 모델 학습
    print("\n" + "="*60)
    print("2. 후행문 모델 학습 시작")
    print("="*60)
    
    next_model_path = train_model_for_position(
        position='후행문',
        train_path=config.TRAIN_PATH,
        dev_path=config.DEV_PATH,
        model_name=config.MODEL_NAME,
        max_length=config.MAX_LENGTH,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        num_epochs=config.NUM_EPOCHS,
        patience=config.PATIENCE,
        output_dir=config.OUTPUT_DIR,
        device=device,
        resume_from=config.RESUME_NEXT
    )
    
    # 최종 결과
    print("\n" + "="*60)
    print("✓ 모든 학습 완료!")
    print("="*60)
    print(f"\n선행문 모델: {prev_model_path}")
    print(f"후행문 모델: {next_model_path}")
    print("\n추론을 위해 inference.py를 실행하세요.")


if __name__ == '__main__':
    main()
