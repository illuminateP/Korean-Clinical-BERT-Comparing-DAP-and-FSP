#!/usr/bin/env python3
"""
3.build_dataset_scratch.py - KCI 초록 추출 + 도메인 토크나이저 학습 + 말뭉치 전처리

실험 T2: KCI 한국어 논문 초록을 추출하여 별도 텍스트 파일로 저장한 뒤,
해당 파일로만 토크나이저를 학습하고 전체 cleaned_corpus.txt를 대상으로 MLM+NSP 데이터셋을 생성합니다.

기존 2.build_dataset.py의 scratch 모드와 완벽히 동일한 조건:
  - MAX_SEQ_LENGTH = 512
  - TARGET_SEG_LEN = 250
  - VAL_SIZE = 10000
  - SEED = 42
  - SCRATCH_VOCAB_SIZE = 32000

차이점: 토크나이저 학습 전 KCI 초록을 파일로 먼저 추출 및 저장 후 학습 진행

Usage:
    python 3.build_dataset_scratch.py
"""

import os
import sys
import io
import re
import gc
import json
import random
import argparse
from tqdm import tqdm
from datasets import Dataset, Features, Value, ClassLabel, load_from_disk
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ==========================================
# [설정] - 기존 실험과 동일한 하이퍼파라미터
# ==========================================
# ─────────────────────────────────────────────────────────────────────────
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
#        단계별 산출물 위치는 README의 디렉토리 규약을 따른다.
# ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

CORPUS_FILE = os.path.join(DATA, "2.preprocessing", "cleaned", "cleaned_corpus.txt")
INTERMEDIATE_DIR = os.path.join(DATA, "2.preprocessing", "intermediate_docs")  # 기존 문서 인덱스 공유

# ARTI_CONF 는 ARTI(학술지)를 포함하는 세트라 의료 카테고리로 거르면 두 파일이 같다.
# 원 실험은 둘 다 읽었으므로 그대로 둔다(README 2.1 각주 참고).
KCI_FILES = [
    os.path.join(DATA, "1.corpus", "3.kci", "outputs", "kci_records_arti.jsonl"),
    os.path.join(DATA, "1.corpus", "3.kci", "outputs", "kci_records_arti_conf.jsonl"),
]

TARGET_CATEGORIES = [
    '가정의학', '간호학', '기타의약학', '내과학', '마취과학', '면역학',
    '물리치료학', '방사선과학', '병리학', '비뇨기과학', '산부인과학', '성형외과학',
    '소아과학', '수의학', '신경과학', '신경외과학', '약리학', '약학', '예방의학',
    '응급의학', '의공학', '의약학', '의학일반', '일반외과학', '임상병리학',
    '작업치료학', '재활의학', '정신과학', '정형외과학', '치의학', '피부과학',
    '한의학', '해부학',
]

PATTERN_KOR = re.compile(r"[\uac00-\ud7a3]")

BASELINE_MODEL = "klue/bert-base"
TOKENIZER_DIR = os.path.join(DATA, "2.preprocessing", "2.scratch", "tokenizer")
DATASET_DIR = os.path.join(DATA, "2.preprocessing", "2.scratch", "datasets")
KCI_ABSTRACT_FILE = os.path.join(DATA, "2.preprocessing", "2.scratch", "kci_abstract.txt")  # 추출된 텍스트를 저장할 파일 경로

MAX_SEQ_LENGTH = 512
TARGET_SEG_LEN = 250
SEED = 42
VAL_SIZE = 10000
NUM_PROC = 4
VOCAB_SIZE = 32000


# ==========================================
# [Phase 0] KCI 초록 텍스트 추출 및 저장
# ==========================================
def extract_and_save_kci_abstracts():
    """KCI JSONL에서 의료 카테고리 한국어 초록만 추출하여 텍스트 파일로 저장"""
    os.makedirs(os.path.dirname(KCI_ABSTRACT_FILE), exist_ok=True)

    if os.path.exists(KCI_ABSTRACT_FILE):
        print(f"  이미 추출된 KCI 초록 파일이 존재합니다: {KCI_ABSTRACT_FILE}")
        return

    print(f"  KCI 초록 텍스트를 추출하여 {KCI_ABSTRACT_FILE}에 저장합니다...")
    total_abstracts = 0
    with open(KCI_ABSTRACT_FILE, "w", encoding="utf-8") as out_f:
        for fpath in KCI_FILES:
            if not os.path.exists(fpath):
                print(f"  WARNING: {fpath} 없음, 건너뜁니다.")
                continue

            print(f"  읽는 중: {os.path.basename(fpath)}")
            with open(fpath, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue

                    category = data.get("categories", "")
                    abstract = data.get("abstract", "")

                    if category not in TARGET_CATEGORIES or not abstract:
                        continue
                    if not PATTERN_KOR.search(abstract):
                        continue

                    total_abstracts += 1
                    for text_line in abstract.split("\n"):
                        text_line = text_line.strip()
                        if text_line:
                            out_f.write(text_line + "\n")

    print(f"  -> KCI 초록 {total_abstracts:,}개 텍스트 추출 및 저장 완료")

def kci_file_iterator():
    """저장된 KCI 초록 텍스트 파일을 읽어 yield"""
    with open(KCI_ABSTRACT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


# ==========================================
# [Phase 1] 토크나이저 학습 (저장된 KCI 텍스트 파일 기반)
# ==========================================
def train_tokenizer():
    """KCI 논문 초록 텍스트 파일로 WordPiece 토크나이저 학습"""
    print("=" * 60)
    print("[Phase 1] KCI 초록 파일 기반 토크나이저 학습")
    print("=" * 60)

    if os.path.exists(TOKENIZER_DIR) and os.listdir(TOKENIZER_DIR):
        print(f"  기존 토크나이저 로드: {TOKENIZER_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        print(f"  Vocab size: {len(tokenizer):,}\n")
        return tokenizer

    # 텍스트 추출 및 파일 저장 실행
    extract_and_save_kci_abstracts()

    base_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)

    print(f"  {KCI_ABSTRACT_FILE} 파일을 읽어 토크나이저 학습 중...")
    tokenizer = base_tokenizer.train_new_from_iterator(
        kci_file_iterator(),
        vocab_size=VOCAB_SIZE,
    )

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    print(f"  -> 토크나이저 저장 완료: {TOKENIZER_DIR}")
    print(f"  Vocab size: {len(tokenizer):,}\n")
    return tokenizer


# ==========================================
# [Phase 2] 문서 인덱스 구축 (기존 공유)
# ==========================================
def split_sentences(text):
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def doc_generator():
    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        doc_lines = []
        for line in f:
            line = line.strip()
            if not line:
                if doc_lines:
                    full_text = " ".join(doc_lines)
                    sentences = split_sentences(full_text)
                    if sentences:
                        yield {"sentences": sentences}
                    doc_lines = []
            else:
                doc_lines.append(line)
        if doc_lines:
            full_text = " ".join(doc_lines)
            sentences = split_sentences(full_text)
            if sentences:
                yield {"sentences": sentences}


def build_document_index():
    if os.path.exists(INTERMEDIATE_DIR):
        print(f"[Phase 2] 기존 문서 인덱스 로드: {INTERMEDIATE_DIR}")
        doc_dataset = load_from_disk(INTERMEDIATE_DIR)
        print(f"  -> 문서: {len(doc_dataset):,}개\n")
        return doc_dataset

    print("[Phase 2] 문서 인덱스 구축 중 (최초 1회)...")
    doc_dataset = Dataset.from_generator(
        doc_generator,
        features=Features({"sentences": [Value("string")]}),
        writer_batch_size=200,
    )
    doc_dataset.save_to_disk(INTERMEDIATE_DIR)

    total_sentences = 0
    for item in tqdm(doc_dataset, desc="  문장 수 계산", ncols=80):
        total_sentences += len(item["sentences"])

    print(f"\n  -> 문서: {len(doc_dataset):,}개 | 문장: {total_sentences:,}개\n")
    return doc_dataset


# ==========================================
# [Phase 3] 세그먼트 패킹 + NSP (기존과 동일)
# ==========================================
def create_segments_from_doc(sentences, tokenizer, max_len, target_len):
    segments = []
    current_segment = []
    current_len = 0

    for sent in sentences:
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        len_sent = len(token_ids)

        if len_sent > max_len - 20:
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_len = 0
            for i in range(0, len_sent, target_len):
                sub_tokens = token_ids[i : i + target_len]
                segments.append(tokenizer.decode(sub_tokens))
            continue

        if current_len + len_sent < target_len:
            current_segment.append(sent)
            current_len += len_sent
        else:
            if current_segment:
                segments.append(" ".join(current_segment))
            current_segment = [sent]
            current_len = len_sent

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments


def generate_nsp_pairs(doc_dataset, tokenizer, max_len, target_len):
    total_docs = len(doc_dataset)
    offset = total_docs // 2

    for doc_idx, item in enumerate(
        tqdm(doc_dataset, desc="  [Phase 3] NSP 쌍 생성", mininterval=2.0, ncols=80)
    ):
        if doc_idx % 2000 == 0:
            gc.collect()

        segments = create_segments_from_doc(
            item["sentences"], tokenizer, max_len, target_len
        )
        if len(segments) < 2:
            continue

        for i in range(len(segments) - 1):
            seg_a = segments[i]

            if random.random() < 0.5:
                seg_b = segments[i + 1]
                label = 0
            else:
                target_idx = (doc_idx + offset) % total_docs
                rand_sentences = doc_dataset[target_idx]["sentences"]

                if not rand_sentences:
                    seg_b = segments[i + 1]
                    label = 0
                else:
                    start = random.randint(0, max(0, len(rand_sentences) - 5))
                    seg_b = " ".join(rand_sentences[start : start + 5])
                    label = 1

            yield {
                "sentence_a": seg_a,
                "sentence_b": seg_b,
                "next_sentence_label": label,
            }


# ==========================================
# [Phase 4] 토크나이징 + 저장
# ==========================================
def process_dataset(tokenizer, doc_dataset):
    print("=" * 60)
    print("[Phase 4] 전체 말뭉치 토크나이징 + 데이터셋 저장")
    print("=" * 60)

    random.seed(SEED)

    def nsp_gen():
        yield from generate_nsp_pairs(
            doc_dataset, tokenizer, MAX_SEQ_LENGTH, TARGET_SEG_LEN
        )

    features = Features(
        {
            "sentence_a": Value("string"),
            "sentence_b": Value("string"),
            "next_sentence_label": ClassLabel(
                num_classes=2, names=["IsNext", "NotNext"]
            ),
        }
    )

    print("\n  NSP 데이터셋 생성 중...")
    nsp_dataset = Dataset.from_generator(nsp_gen, features=features, writer_batch_size=100)
    print(f"  -> NSP 쌍: {len(nsp_dataset):,}개")

    print(f"  데이터셋 분리 (val_size={VAL_SIZE})...")
    split_datasets = nsp_dataset.train_test_split(
        test_size=VAL_SIZE, seed=SEED, shuffle=True, keep_in_memory=False
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence_a"],
            examples["sentence_b"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

    print("  토크나이징 중...")
    tokenized = split_datasets.map(
        tokenize_fn,
        batched=True,
        num_proc=NUM_PROC,
        remove_columns=["sentence_a", "sentence_b"],
    )

    os.makedirs(DATASET_DIR, exist_ok=True)
    tokenized.save_to_disk(DATASET_DIR)
    print(f"  -> 저장 완료: {DATASET_DIR}")
    print(f"     Train: {len(tokenized['train']):,}  |  Val: {len(tokenized['test']):,}\n")


# ==========================================
# [메인]
# ==========================================
def main():
    print("=" * 60)
    print("T2 실험: KCI 논문 초록 추출 및 기반 토크나이저 학습 + 전체 말뭉치 전처리")
    print("=" * 60)
    print()

    # Phase 1: KCI 초록 추출 및 토크나이저 학습
    tokenizer = train_tokenizer()

    # Phase 2: 문서 인덱스 (기존 공유)
    doc_dataset = build_document_index()

    # Phase 3-4: NSP 생성 + 토크나이징 + 저장
    process_dataset(tokenizer, doc_dataset)

    print("=" * 60)
    print("T2 전처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()