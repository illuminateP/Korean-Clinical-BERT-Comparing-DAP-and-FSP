import torch
import torch.nn as nn
from transformers import BertForPreTraining

class BaselineBertForPreTraining(BertForPreTraining):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        
        # 1. 부모 클래스 초기화 및 기본 가중치 공유(Tying) 적용
        # 이 시점까지는 embeddings와 decoder가 연결되어 있음
        self.post_init()
        # 토크나이저 할당
        self.tokenizer = tokenizer

        # ----------------------------------------------------------------
        # 2. 공정한 비교를 위한 Weight Tying 강제 해제
        # HanVec 모델이 구조상 Weight Tying을 할 수 없으므로, 
        # Baseline도 동일하게 Decoder를 독립적인 Linear 레이어로 교체함
        # ----------------------------------------------------------------
        self.cls.predictions.decoder = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False
        )
        
        # 3. 바이어스 재설정 (0으로 초기화 및 연결)
        self.cls.predictions.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.cls.predictions.decoder.bias = self.cls.predictions.bias

        # NSP Head는 가중치를 입력 임베딩과 공유하지 않으므로 초기화할 필요 X
        # 4. MLM 가중치 초기화
        self._init_weights(self.cls.predictions.decoder)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, next_sentence_label=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            next_sentence_label=next_sentence_label,
            **kwargs
        )