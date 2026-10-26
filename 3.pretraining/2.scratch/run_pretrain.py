#!/usr/bin/env python3
"""
run_pretrain.py - KCI 토크나이저 기반 BERT Pretraining from Scratch

실험 T2: KCI 논문 초록으로 학습된 토크나이저 + 전체 말뭉치 MLM+NSP 학습
기존 2.scratch/run_pretrain_from_scratch.py와 완벽히 동일한 학습 조건.

Usage:
    python 2.run_pretrain.py
    python 2.run_pretrain.py --epochs 20 --batch_size 16 --lr 1e-4
"""

import os
import sys
import io
import csv
import time
import datetime
import argparse
from typing import Any, Dict, List

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForPreTraining,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")


# ==========================================
# [설정]
# ==========================================
MODEL_NAME = "klue/bert-base"  # 아키텍처(config)만 참조, 가중치는 사용하지 않음
TOKENIZER_DIR = os.path.join(DATA, "2.preprocessing", "2.scratch", "tokenizer")
DATASET_DIR = os.path.join(DATA, "2.preprocessing", "2.scratch", "datasets")
OUTPUT_DIR = os.path.join(DATA, "3.pretraining", "2.scratch")
MLM_PROBABILITY = 0.15
SEED = 42


# ==========================================
# [콜백] 학습 로그를 CSV 파일로 저장
# ==========================================
class TrainingLogger(TrainerCallback):
    def __init__(self, filepath, resuming=False):
        self.filepath = filepath
        # 체크포인트에서 이어 학습하면 기존 곡선 뒤에 덧붙이고,
        # 처음부터 학습하면 이전 실행의 곡선을 지우고 새로 쓴다.
        self.resuming = resuming
        self.fieldnames = [
            "step", "epoch", "loss", "learning_rate",
            "eval_loss", "eval_mlm_accuracy", "eval_nsp_accuracy",
        ]

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if not self.resuming:
                # 신규 학습: 이전 실행의 곡선을 비우고 헤더부터 다시 쓴다.
                with open(self.filepath, mode="w", newline="") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
                return

            should_write_header = (
                not os.path.exists(self.filepath)
                or os.path.getsize(self.filepath) == 0
            )
            if should_write_header:
                with open(self.filepath, mode="a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            if not any(k in logs for k in ["loss", "eval_loss"]):
                return
            row = {
                "step": state.global_step,
                "epoch": round(state.epoch, 4) if state.epoch else 0,
                "loss": logs.get("loss", ""),
                "learning_rate": logs.get("learning_rate", ""),
                "eval_loss": logs.get("eval_loss", ""),
                "eval_mlm_accuracy": logs.get("eval_mlm_accuracy", ""),
                "eval_nsp_accuracy": logs.get("eval_nsp_accuracy", ""),
            }
            with open(self.filepath, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)


# ==========================================
# [콜백] FTDC (Friendly Time Display Callback)
# ==========================================
class FTDC(TrainerCallback):
    def __init__(self, total_steps=None):
        self.total_steps = total_steps
        self.last_log_time = time.time()
        self.last_log_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs and "loss" in logs:
            current_time = time.time()
            current_step = state.global_step

            step_diff = current_step - self.last_log_step
            time_diff = current_time - self.last_log_time

            seconds_per_step = time_diff / step_diff if step_diff > 0 else 0

            max_steps = (
                state.max_steps if self.total_steps is None else self.total_steps
            )
            remaining_steps = max_steps - current_step
            eta_seconds = int(remaining_steps * seconds_per_step)

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            eta_str = str(datetime.timedelta(seconds=eta_seconds))
            progress = (current_step / max_steps) * 100 if max_steps > 0 else 0

            epoch = logs.get("epoch", 0)
            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)

            print(f"\n [FTDC] {now_str}")
            print(
                f" [FTDC] [Epoch {epoch:.2f}] Step {current_step}/{max_steps} ({progress:.1f}%)"
            )
            print(f" [FTDC] Loss: {loss:.4f} | LR: {lr:.2e}")
            print(
                f" [FTDC] 구간 속도: {1/seconds_per_step:.2f} it/s | 예상 남은 시간: {eta_str}"
            )

            self.last_log_time = current_time
            self.last_log_step = current_step


# ==========================================
# [Data Collator] MLM + NSP
# ==========================================
class DataCollatorForBertPreTraining:
    """MLM 동적 마스킹 + NSP 레이블을 함께 처리하는 Data Collator"""

    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        nsp_labels = [f["next_sentence_label"] for f in features]
        special_tokens_masks = [f["special_tokens_mask"] for f in features]

        batch = {
            key: torch.tensor([f[key] for f in features], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }

        # ---- MLM 마스킹 ----
        labels = batch["input_ids"].clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_mask = torch.tensor(special_tokens_masks, dtype=torch.bool)
        probability_matrix.masked_fill_(special_mask, value=0.0)

        padding_mask = batch["attention_mask"] == 0
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK], 10% random, 10% keep
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        batch["input_ids"][indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            "[MASK]"
        )

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        batch["input_ids"][indices_random] = random_words[indices_random]

        batch["labels"] = labels
        batch["next_sentence_label"] = torch.tensor(nsp_labels, dtype=torch.long)

        return batch


# ==========================================
# [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="T2: KCI 토크나이저 기반 BERT Pretraining from Scratch"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=5000)
    args = parser.parse_args()

    # 1. KCI 기반 커스텀 토크나이저 로드
    if not os.path.exists(TOKENIZER_DIR):
        print(f"ERROR: 토크나이저를 찾을 수 없습니다: {TOKENIZER_DIR}")
        print("       2.preprocessing/1.pretrain_data/3.build_dataset_scratch.py 를 먼저 실행하세요.")
        sys.exit(1)

    print(f"[1] KCI 기반 토크나이저 로드: {TOKENIZER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    print(f"    Vocab size: {len(tokenizer):,}")

    # 2. BERT 모델 초기화 (from scratch, random weights)
    config = BertConfig.from_pretrained(MODEL_NAME)
    config.vocab_size = len(tokenizer)
    model = BertForPreTraining(config)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[2] 모델 초기화 완료 (from scratch)")
    print(f"    파라미터 수: {param_count:,}")

    # 3. 전처리된 데이터셋 로드
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: 데이터셋을 찾을 수 없습니다: {DATASET_DIR}")
        print("       2.preprocessing/1.pretrain_data/3.build_dataset_scratch.py 를 먼저 실행하세요.")
        sys.exit(1)

    print(f"[3] 데이터셋 로드: {DATASET_DIR}")
    datasets = load_from_disk(DATASET_DIR)
    print(f"    Train: {len(datasets['train']):,}  |  Val: {len(datasets['test']):,}")

    # 4. Data Collator
    data_collator = DataCollatorForBertPreTraining(
        tokenizer=tokenizer,
        mlm_probability=MLM_PROBABILITY,
    )

    # 5. 학습 설정 (scratch와 동일)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        torch_compile=True,
        dataloader_num_workers=8,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
    )

    # 6. 콜백 생성
    num_gpus = max(1, torch.cuda.device_count())
    effective_batch = args.batch_size * num_gpus
    total_train_steps = (len(datasets["train"]) // effective_batch) * args.epochs

    # resume 여부를 로거가 알아야 하므로 체크포인트 탐색을 먼저 수행한다.
    last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    csv_logger = TrainingLogger(
        filepath=os.path.join(OUTPUT_DIR, "training_logs.csv"),
        resuming=last_checkpoint is not None,
    )
    time_callback = FTDC(total_steps=total_train_steps)

    # 7. Trainer 생성 & 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=data_collator,
        callbacks=[csv_logger, time_callback],
    )

    print(
        f"\n[4] 학습 시작 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})"
    )
    
    if last_checkpoint is not None:
        print(f"저장된 체크포인트({last_checkpoint})에서 학습을 이어서 진행합니다.")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("체크포인트가 존재하지 않아 처음부터 학습을 시작합니다.")
        trainer.train()

    # 8. 최종 모델 저장
    model_dir = os.path.join(OUTPUT_DIR, "models")
    tokenizer_dir = os.path.join(OUTPUT_DIR, "tokenizer")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"\n[완료] Best 모델 저장: {model_dir}")
    print(f"[완료] 토크나이저 저장: {tokenizer_dir}")


if __name__ == "__main__":
    main()
