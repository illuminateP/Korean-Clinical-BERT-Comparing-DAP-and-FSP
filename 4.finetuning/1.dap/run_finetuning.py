"""
Baseline 모델 의료 파인튜닝 스크립트 (RTX 3090 24GB 하드코딩 및 검증 로직 추가)
- Tasks: KorMedQA (MCQA), KBMC (NER - 7 Labels)
- KorMedQA: datasets/saved/KorMedQA/{subset}/ (HuggingFace Dataset)
- KBMC: datasets/saved/KBMC/ (사전 분할된 HuggingFace Dataset)

Usage:
    python run_finetuning_baseline.py
    python run_finetuning_baseline.py --tasks KorMedQA --kormedqa_subset nurse
"""

import os
import sys
# 1번 GPU 사용
TARGET_GPU_ID = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = TARGET_GPU_ID

import csv
import torch
import json
import argparse
import random
import numpy as np

from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from sklearn.metrics import f1_score as sklearn_f1_score
from seqeval.metrics import classification_report as seqeval_report, f1_score as seqeval_f1
from transformers import (
    BertConfig,
    AutoTokenizer,
    BertForMultipleChoice,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    TrainerCallback,
    set_seed as transformers_set_seed
)

# modeling.py 임포트 (가중치 로드용)
try:
    from modeling import BaselineBertForPreTraining
except ImportError:
    print("[Error] modeling.py 모듈을 불러올 수 없습니다. 경로를 확인해주세요.")

# ==========================================
# [설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

DEFAULT_OUTPUT_ROOT = os.path.join(DATA, "4.finetuning", "1.dap")
DEFAULT_MODEL_PATH = os.path.join(DATA, "3.pretraining", "1.dap", "models")
SAVED_DATASETS_DIR = os.path.join(DATA, "2.preprocessing", "finetune", "saved")


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)


# ==========================================
# [콜백] Training Logger (CSV)
# ==========================================
class TrainingLogger(TrainerCallback):
    def __init__(self, filepath):
        self.filepath = filepath
        self.fieldnames = [
            'step', 'epoch', 'loss', 'learning_rate', 'eval_loss',
            'eval_accuracy', 'eval_f1'
        ]

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            if not os.path.exists(self.filepath) or os.path.getsize(self.filepath) == 0:
                with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                    writer.writeheader()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            if not any(k in logs for k in ['loss', 'eval_loss']): return
            row = {k: logs.get(k, '') for k in self.fieldnames}
            row['step'] = state.global_step
            if not row.get('epoch') and state.epoch is not None:
                row['epoch'] = state.epoch
            with open(self.filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
                writer.writerow(row)


# ==========================================
# [Task 정의 & HPARAMS (RTX 3090 24GB 최적화)]
# ==========================================
TASK_INFO = {
    "KorMedQA": {"type": "mcqa", "num_labels": 5, "dir_name": "KorMedQA"},
    "KBMC":     {"type": "ner",  "num_labels": 4, "dir_name": "KBMC"},
}

KBMC_TAG_MAP = {
    "Body": 0,
    "Disease": 1,
    "Treatment": 2,
    "O": 3,
}
NER_ID_TO_TAG = {
    0: "Body", 1: "Disease", 2: "Treatment", 3: "O"
}

# 평가 라이브러리(seqeval)가 연속된 태그를 하나의 개체로 병합하도록 I- 접두사 부여
NER_ID_TO_SEQEVAL = {
    0: "I-Body", 1: "I-Disease", 2: "I-Treatment", 3: "O"
}

HPARAMS = {
    "KorMedQA": {
        "max_epochs": 20,
        "batch_size": 4,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-05,
        "dropout_prob": 0.1,
        "warmup_ratio": 0.1,
        "max_seq_len": 512,
        "tb_interval": 100,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-6,
    },
    "KBMC": {
        "max_epochs": 5,
        "batch_size": 32,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1e-04,
        "dropout_prob": 0.1,
        "warmup_ratio": 0.1,
        "max_seq_len": 128,
        "tb_interval": 50,
        "weight_decay": 0.01,
        "adam_epsilon": 1e-6,
    },
}

KORMEDQA_SUBSETS = ["doctor", "nurse", "pharm", "dentist"]


# ==========================================
# [데이터 로드] KorMedQA (Multiple Choice 구조 적용 및 검증 로직 추가)
# ==========================================
def load_kormedqa_single(tokenizer, max_seq_len, subset, split_name, num_labels):
    split_path = os.path.join(SAVED_DATASETS_DIR, "KorMedQA", subset, split_name)
    if not os.path.isdir(split_path):
        return None

    raw_ds = load_from_disk(split_path)

    first_sentences = []
    second_sentences = []
    labels = []

    for sample in raw_ds:
        q = sample["question"]
        options = [
            sample.get("A") or "",
            sample.get("B") or "",
            sample.get("C") or "",
            sample.get("D") or "",
            sample.get("E") or ""
        ]
        
        # 선지 개수와 설정된 num_labels 불일치 시 에러 발생
        if len(options) != num_labels:
            raise ValueError(f"[오류] 데이터의 선지 개수({len(options)})와 설정된 num_labels({num_labels})가 일치하지 않습니다.")

        first_sentences.extend([q] * num_labels)
        second_sentences.extend(options)
        labels.append(sample["answer"] - 1)

    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len
    )

    processed_features = {
        k: [v[i : i + num_labels] for i in range(0, len(v), num_labels)]
        for k, v in tokenized.items()
    }
    
    processed_features["labels"] = labels

    ds = Dataset.from_dict(processed_features)
    ds.set_format("torch")
    return ds


def load_kormedqa(tokenizer, max_seq_len, num_labels=5, seed=42):
    print("[Data] KorMedQA: Loading and Merging ALL subsets for Joint Training")
    
    train_datasets = []
    dev_datasets = []
    
    for subset in KORMEDQA_SUBSETS:
        train_ds = load_kormedqa_single(tokenizer, max_seq_len, subset, "train", num_labels)
        dev_ds = load_kormedqa_single(tokenizer, max_seq_len, subset, "dev", num_labels)
        
        if train_ds is not None:
            train_datasets.append(train_ds)
        if dev_ds is not None:
            dev_datasets.append(dev_ds)

    # 모든 서브셋 데이터 병합 및 무작위 셔플링
    merged_train = concatenate_datasets(train_datasets).shuffle(seed=seed)
    merged_dev = concatenate_datasets(dev_datasets).shuffle(seed=seed)
    
    print(f"  [train] {len(merged_train)} joint samples loaded")
    print(f"  [dev] {len(merged_dev)} joint samples loaded")
    
    return DatasetDict({
        "train": merged_train,
        "dev": merged_dev
    })

# ==========================================
# [데이터 로드] KBMC (NER)
# ==========================================
def _split_at_entity_boundary(word_info, max_content_len):
    """개체(Entity)가 잘리지 않는 경계에서 word_info를 분할하여 청크 리스트로 반환."""
    chunks = []
    start = 0

    while start < len(word_info):
        cumulative = 0
        end = start
        for i in range(start, len(word_info)):
            wt_len = len(word_info[i][0])
            if cumulative + wt_len > max_content_len:
                break
            cumulative += wt_len
            end = i + 1

        if end >= len(word_info):
            chunks.append(word_info[start:])
            break

        if end == start:
            end = start + 1
            chunks.append(word_info[start:end])
            start = end
            continue

        safe_end = end
        for j in range(end, start, -1):
            if j >= len(word_info):
                continue
            tag_name = NER_ID_TO_TAG[word_info[j][1]]
            if tag_name == 'O' or tag_name.endswith('-B'):
                safe_end = j
                break

        if safe_end <= start:
            safe_end = end

        chunks.append(word_info[start:safe_end])
        start = safe_end

    return chunks


def load_kbmc(tokenizer, max_seq_len):
    # 3.convert_kbmc.py 가 생성한 4-label IO 체계 데이터를 사용한다
    saved_dir = os.path.join(SAVED_DATASETS_DIR, "KBMC")
    print(f"[Data] KBMC: Loading from saved dataset with word_ids mapping")

    splits = {}
    for split_name in ["train", "valid"]:
        split_path = os.path.join(saved_dir, split_name)
        if not os.path.isdir(split_path): continue

        raw_ds = load_from_disk(split_path)

        all_input_ids, all_attention_masks, all_labels = [], [], []

        for sample in raw_ds:
            words = sample["Sentence"].split()
            tags = sample["Tags"].split()
            tag_ids = [KBMC_TAG_MAP[t] for t in tags]

            # 띄어쓰기 기준으로 토크나이저에 입력 (문맥 보존)
            tokenized = tokenizer(
                words,
                is_split_into_words=True,
                max_length=max_seq_len,
                padding="max_length",
                truncation=True
            )

            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            word_ids = tokenized.word_ids()

            # word_ids를 이용한 서브워드 라벨 정렬
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    # [CLS], [SEP], [PAD] 토큰
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # 새로운 단어의 첫 번째 서브워드에 정답 라벨 부여
                    label_ids.append(tag_ids[word_idx])
                else:
                    # 같은 단어에서 파생된 후속 서브워드에도 동일 라벨 부여
                    label_ids.append(tag_ids[word_idx])
                    
                previous_word_idx = word_idx

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(label_ids)

        ds = Dataset.from_dict({
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        })
        ds.set_format("torch")
        splits[split_name] = ds
        print(f"  [{split_name}] {len(ds)} samples loaded")

    return DatasetDict(splits)


# ==========================================
# [메트릭] Compute Metrics
# ==========================================
def get_compute_metrics(task_type):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task_type == "mcqa":
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": (predictions == labels).mean(),
                "f1": float(sklearn_f1_score(labels, predictions, average='macro', zero_division=0)),
            }
        elif task_type == "ner":
            predictions = np.argmax(predictions, axis=2)
            true_preds_seq, true_labels_seq = [], []
            for pred, label in zip(predictions, labels):
                pred_tags, label_tags = [], []
                for p, l in zip(pred, label):
                    if l != -100:
                        pred_tags.append(NER_ID_TO_SEQEVAL[int(p)])
                        label_tags.append(NER_ID_TO_SEQEVAL[int(l)])
                true_preds_seq.append(pred_tags)
                true_labels_seq.append(label_tags)
            return {
                "f1": seqeval_f1(true_labels_seq, true_preds_seq, average='macro', zero_division=0),
            }
        return {}
    return compute_metrics


def evaluate_kbmc(preds):
    predictions = np.argmax(preds.predictions, axis=2)
    labels = preds.label_ids

    true_preds_seq, true_labels_seq = [], []
    for pred, label in zip(predictions, labels):
        pred_tags, label_tags = [], []
        for p, l in zip(pred, label):
            if l != -100:
                pred_tags.append(NER_ID_TO_SEQEVAL[int(p)])
                label_tags.append(NER_ID_TO_SEQEVAL[int(l)])
        true_preds_seq.append(pred_tags)
        true_labels_seq.append(label_tags)

    report = seqeval_report(true_labels_seq, true_preds_seq, output_dict=True, zero_division=0)

    metrics = {}
    for entity_type in ['Body', 'Disease', 'Treatment']:
        if entity_type in report:
            metrics[f'{entity_type}_precision'] = round(report[entity_type]['precision'] * 100, 2)
            metrics[f'{entity_type}_recall'] = round(report[entity_type]['recall'] * 100, 2)
            metrics[f'{entity_type}_f1'] = round(report[entity_type]['f1-score'] * 100, 2)

    metrics['macro_f1'] = round(report['macro avg']['f1-score'] * 100, 2)
    metrics['micro_f1'] = round(report['micro avg']['f1-score'] * 100, 2)

    return metrics


def evaluate_kormedqa_all_subsets(trainer, tokenizer, max_seq_len, num_labels):
    metrics = {}
    all_correct, all_total = 0, 0

    for subset in KORMEDQA_SUBSETS:
        test_ds = load_kormedqa_single(tokenizer, max_seq_len, subset, "test", num_labels)
        if test_ds is None: continue

        preds = trainer.predict(test_ds)
        pred_indices = np.argmax(preds.predictions, axis=1)
        labels = preds.label_ids

        correct = int((pred_indices == labels).sum())
        total = len(labels)
        acc = round(float(correct / total * 100), 2) if total > 0 else 0.0

        metrics[f"{subset}_acc"] = acc
        all_correct += correct
        all_total += total
        print(f"  -> {subset}: {acc}% ({correct}/{total})")

    metrics["accuracy"] = round(float(all_correct / all_total * 100), 2) if all_total > 0 else 0.0
    print(f"  -> overall: {metrics['accuracy']}% ({all_correct}/{all_total})")

    return metrics


# ==========================================
# [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Medical Fine-tuning (Baseline)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Path to pretrained checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["KorMedQA", "KBMC"],
                        help="Tasks to run (default: KorMedQA KBMC)")
    args = parser.parse_args()

    os.makedirs(DEFAULT_OUTPUT_ROOT, exist_ok=True)
    experiment_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))

    SEED = 42
    set_all_seeds(SEED)

    # 3.pretraining 은 토크나이저를 가중치 폴더의 형제로 저장한다
    #   data/3.pretraining/<실험>/models      <- 가중치 (--model_path)
    #   data/3.pretraining/<실험>/tokenizer   <- 토크나이저
    # 과거 구조처럼 models/tokenizer 안에 둔 경우도 있으므로 둘 다 찾는다.
    candidates = [
        os.path.join(args.model_path, "tokenizer"),
        os.path.join(os.path.dirname(args.model_path), "tokenizer"),
    ]
    tokenizer_path = next((p for p in candidates if os.path.isdir(p)), None)
    if tokenizer_path is None:
        print("ERROR: 토크나이저를 찾을 수 없습니다. 다음 경로를 확인했습니다:")
        for p in candidates:
            print(f"  - {p}")
        print("       3.pretraining 단계를 먼저 완료하세요.")
        sys.exit(1)

    print(f"[Tokenizer] {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    all_final_results = {}

    for task_name in args.tasks:
        if task_name not in TASK_INFO: continue
        info = TASK_INFO[task_name]
        task_config = HPARAMS[task_name]

        print(f"\n{'='*60}\n[DEBUG] Start Fine-tuning: {task_name}\n{'='*60}")

        num_epochs = task_config["max_epochs"]
        batch_size = task_config["batch_size"]
        gradient_accumulation_steps = task_config["gradient_accumulation_steps"]
        learning_rate = float(task_config["learning_rate"])
        warmup_ratio = task_config["warmup_ratio"]
        max_seq_len = task_config["max_seq_len"]
        tb_interval = task_config["tb_interval"]
        dropout_prob = task_config["dropout_prob"]
        weight_decay = task_config["weight_decay"]
        adam_epsilon = task_config["adam_epsilon"]

        print(f"[HPARAMS] Applied: Task={task_name}, Epochs={num_epochs}, BS={batch_size}, LR={learning_rate}, GA={gradient_accumulation_steps}, TB={tb_interval}")
        print(" >> Task Hyperparameters:")
        for key, value in task_config.items():
            print(f"    - {key:<28}: {value}")

        # 데이터 로드
        if task_name == "KorMedQA":
            dataset = load_kormedqa(tokenizer, max_seq_len, info["num_labels"], SEED)
        elif task_name == "KBMC":
            dataset = load_kbmc(tokenizer, max_seq_len)
        
        train_ds = dataset.get("train")
        valid_ds = dataset.get("dev") if task_name == "KorMedQA" else dataset.get("valid")

        if train_ds is None:
            raise RuntimeError(f"[Error] {task_name}의 train 데이터셋을 불러올 수 없습니다.")
        if valid_ds is None:
            raise RuntimeError(f"[Error] {task_name}의 valid/dev 데이터셋을 불러올 수 없습니다.")

        print(f" - Train: {len(train_ds)}")
        print(f" - Valid: {len(valid_ds)}")

        # 명시적 BertConfig 및 모델 초기화
        print(f"[Model] Loading BaselineBertForPreTraining from {args.model_path}")
        model_config = BertConfig.from_pretrained(args.model_path, num_labels=info["num_labels"])
        model_config.hidden_dropout_prob = dropout_prob
        model_config.attention_probs_dropout_prob = dropout_prob

        # BaseBert에서 인코더 가중치 이식
        pretrained_base = BaselineBertForPreTraining.from_pretrained(args.model_path, config=model_config)

        if info["type"] == "ner":
            model = BertForTokenClassification(config=model_config)
            model.bert = pretrained_base.bert
        elif info["type"] == "mcqa":
            model = BertForMultipleChoice(config=model_config)
            model.bert = pretrained_base.bert

        data_collator = DefaultDataCollator()
        compute_metrics_fn = get_compute_metrics(info["type"])

        output_dir = os.path.join(DEFAULT_OUTPUT_ROOT, info["dir_name"])
        log_file = os.path.join(output_dir, "training_log.csv")

        training_args = TrainingArguments(
            output_dir=output_dir,
            seed=SEED,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1" if task_name == "KBMC" else "eval_accuracy",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=tb_interval,
            torch_compile=False,
            fp16=False,
            bf16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[TrainingLogger(log_file)]
        )

        print(f"[Train] Starting training for {task_name}...")
        trainer.train()

        print(f"[Eval] Final evaluation for {task_name}...")
        if task_name == "KorMedQA":
            final_metrics = evaluate_kormedqa_all_subsets(trainer, tokenizer, max_seq_len, info["num_labels"])
            final_eval_name = "test (all subsets)"
        elif task_name == "KBMC":
            final_preds = trainer.predict(valid_ds)
            final_metrics = evaluate_kbmc(final_preds)
            final_eval_name = "valid"

        best_model_dir = os.path.join(output_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)
        trainer.save_model(best_model_dir)

        TOKENIZER_DIR = os.path.join(output_dir, "tokenizer")
        os.makedirs(TOKENIZER_DIR, exist_ok=True)
        tokenizer.save_pretrained(TOKENIZER_DIR)

        task_result_data = {
            "experiment_name": experiment_name,
            "dataset_name": task_name,
            "eval_set": final_eval_name,
            "metrics": final_metrics,
        }
        task_metrics_path = os.path.join(output_dir, "final_metrics.json")
        with open(task_metrics_path, "w", encoding="utf-8") as f:
            json.dump(task_result_data, f, indent=4, ensure_ascii=False)

        all_final_results[task_name] = {
            "eval_set": final_eval_name,
            "train_subset": "Joint (All Subsets)" if task_name == "KorMedQA" else None,
            "metrics": final_metrics,
        }

    if all_final_results:
        json_path = os.path.join(DEFAULT_OUTPUT_ROOT, "final_results.json")
        output = {
            "experiment": experiment_name,
            "tasks": all_final_results,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()