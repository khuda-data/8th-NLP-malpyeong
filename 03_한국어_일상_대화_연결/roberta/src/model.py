"""
대화 연결 예측 모델 클래스
- klue/roberta-large 기반
- 선행문/후행문 별도 모델
- Binary classification (pair-wise)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict


class DialogueConnectionModel(nn.Module):
    """대화 연결 예측 모델 (Binary Classification)"""
    
    def __init__(
        self,
        model_name: str = 'klue/roberta-large',
        dropout_rate: float = 0.3,
        num_labels: int = 2,  # 0: 오답, 1: 정답
        local_model_path: str = None  # 로컬 모델 경로 (선택)
    ):
        """
        Args:
            model_name: Pretrained 모델 이름 또는 로컬 경로
            dropout_rate: Dropout 비율
            num_labels: 분류 클래스 수 (binary: 2)
            local_model_path: 로컬에 다운로드한 모델 경로 (우선순위)
        """
        super(DialogueConnectionModel, self).__init__()
        
        import os
        
        # 로컬 경로가 지정되면 우선 사용
        model_path = local_model_path if local_model_path else model_name
        
        # 로컬 파일 존재 여부 확인
        use_local = os.path.exists(model_path) and os.path.isdir(model_path)
        
        # RoBERTa 모델 로드
        if use_local:
            print(f"  Loading model from local path: {os.path.abspath(model_path)}")
            self.config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            self.roberta = AutoModel.from_pretrained(model_path, local_files_only=True)
        else:
            print(f"  Downloading model from HuggingFace: {model_path}")
            self.config = AutoConfig.from_pretrained(model_path)
            self.roberta = AutoModel.from_pretrained(model_path)
        
        # Special tokens에 대한 임베딩 확장은 tokenizer에서 처리됨
        # resize_token_embeddings는 training script에서 수행
        
        hidden_size = self.config.hidden_size  # RoBERTa-large: 1024
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Label smoothing을 위한 설정
        self.num_labels = num_labels
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] (optional, for training)
        
        Returns:
            Dict containing:
                - loss (if labels provided)
                - logits
                - predictions
        """
        # RoBERTa forward
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token representation 사용
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Dropout & Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        
        # Predictions
        predictions = torch.argmax(logits, dim=-1)  # [batch_size]
        
        result = {
            'logits': logits,
            'predictions': predictions
        }
        
        # Loss 계산 (학습 시에만)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
            loss = loss_fct(logits, labels)
            result['loss'] = loss
        
        return result
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """확률값 반환 (softmax 적용)
        
        Returns:
            probabilities: [batch_size, num_labels]
        """
        with torch.no_grad():
            outputs = self(input_ids, attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
        
        return probabilities


class DialogueConnectionEnsemble:
    """선행문/후행문 모델 앙상블 클래스"""
    
    def __init__(
        self,
        prev_model: DialogueConnectionModel,
        next_model: DialogueConnectionModel,
        device: str = 'cuda'
    ):
        """
        Args:
            prev_model: 선행문 예측 모델
            next_model: 후행문 예측 모델
            device: 디바이스
        """
        self.prev_model = prev_model.to(device)
        self.next_model = next_model.to(device)
        self.device = device
        
        self.prev_model.eval()
        self.next_model.eval()
    
    def predict(
        self,
        samples: list,
        tokenizer
    ) -> Dict[str, str]:
        """테스트 샘플에 대한 예측 수행
        
        Args:
            samples: 테스트 샘플 리스트 (id별로 그룹화된 데이터)
            tokenizer: Tokenizer
        
        Returns:
            predictions: {id: '발화1/2/3'}
        """
        predictions = {}
        
        for sample_group in samples:
            sample_id = sample_group['id']
            position = sample_group['position']
            choices = sample_group['choices']  # 3개 선택지
            
            # 모델 선택
            model = self.prev_model if position == '선행문' else self.next_model
            
            # 각 선택지에 대한 확률 계산
            choice_scores = []
            
            for choice_data in choices:
                input_ids = choice_data['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = choice_data['attention_mask'].unsqueeze(0).to(self.device)
                
                # 확률 예측
                with torch.no_grad():
                    probabilities = model.predict_proba(input_ids, attention_mask)
                    # 정답일 확률 (클래스 1)
                    score = probabilities[0, 1].item()
                    choice_scores.append(score)
            
            # 가장 높은 점수의 선택지 선택
            best_choice_idx = choice_scores.index(max(choice_scores))
            predictions[sample_id] = f'발화{best_choice_idx + 1}'
        
        return predictions


def load_model_for_position(
    model_path: str,
    model_name: str = 'klue/roberta-large',
    device: str = 'cuda'
) -> DialogueConnectionModel:
    """저장된 모델 로드
    
    Args:
        model_path: 모델 체크포인트 경로
        model_name: Pretrained 모델 이름
        device: 디바이스
    
    Returns:
        model: 로드된 모델
    """
    model = DialogueConnectionModel(model_name=model_name)
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def save_model(
    model: DialogueConnectionModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    **kwargs
):
    """모델 저장
    
    Args:
        model: 모델
        optimizer: Optimizer
        epoch: 현재 epoch
        save_path: 저장 경로
        **kwargs: 추가 정보 (accuracy, loss 등)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    # 모델 테스트
    print("=== 모델 초기화 테스트 ===")
    
    model = DialogueConnectionModel(
        model_name='klue/roberta-large',
        dropout_rate=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Forward pass 테스트
    print("\n=== Forward Pass 테스트 ===")
    batch_size = 4
    seq_len = 64
    
    dummy_input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    dummy_labels = torch.randint(0, 2, (batch_size,))
    
    outputs = model(
        input_ids=dummy_input_ids,
        attention_mask=dummy_attention_mask,
        labels=dummy_labels
    )
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Predictions: {outputs['predictions']}")
    
    # Predict proba 테스트
    print("\n=== Predict Proba 테스트 ===")
    probas = model.predict_proba(dummy_input_ids, dummy_attention_mask)
    print(f"Probabilities shape: {probas.shape}")
    print(f"Sample probabilities:\n{probas[:2]}")
