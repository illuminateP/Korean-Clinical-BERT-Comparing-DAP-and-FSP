"""
2.build_dataset.py - BERT MLM+NSP 전처리 파이프라인

cleaned_corpus.txt를 읽어 실험별(baseline/scratch) 토크나이징 후
MLM + NSP 학습에 사용할 수 있는 Dataset으로 변환·저장합니다.

Usage:
    python 2.build_dataset.py --mode baseline   # klue/bert-base 토크나이저 (기본값, DAP용)
    python 2.build_dataset.py --mode scratch    # 말뭉치 전체로 토크나이저 학습 후 처리 (초기 변형, 최종 실험에는 미사용)
    python 2.build_dataset.py --mode all        # 위 둘 다

    주의: scratch/all 은 2.scratch/tokenizer 를 먼저 채워 버리므로, 이어서 돌리는
          3.build_dataset_scratch.py 가 KCI 초록 토크나이저 학습을 건너뛰게 된다.
"""

import os
import sys
import io
import re
import gc
import random
import argparse
from tqdm import tqdm
from datasets import Dataset, Features, Value, ClassLabel, load_from_disk
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# [설정]
# ==========================================
# ─────────────────────────────────────────────────────────────────────────
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
#        단계별 산출물 위치는 README의 디렉토리 규약을 따른다.
# ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

CORPUS_FILE = os.path.join(DATA, "2.preprocessing", "cleaned", "cleaned_corpus.txt")
INTERMEDIATE_DIR = os.path.join(DATA, "2.preprocessing", "intermediate_docs")

BASELINE_MODEL = "klue/bert-base"
BASELINE_TOKENIZER_DIR = os.path.join(DATA, "2.preprocessing", "1.dap", "tokenizer")
BASELINE_OUTPUT = os.path.join(DATA, "2.preprocessing", "1.dap", "datasets")

SCRATCH_TOKENIZER_DIR = os.path.join(DATA, "2.preprocessing", "2.scratch", "tokenizer")
SCRATCH_OUTPUT = os.path.join(DATA, "2.preprocessing", "2.scratch", "datasets")

MAX_SEQ_LENGTH = 512
TARGET_SEG_LEN = 250
SEED = 42
VAL_SIZE = 10000
NUM_PROC = 4
SCRATCH_VOCAB_SIZE = 32000


# ==========================================
# [Phase 0] 말뭉치 포맷 검증
# ==========================================
def verify_corpus():
    """cleaned_corpus.txt 포맷 검증 — 5개 데이터셋 병합 상태 확인"""
    print("=" * 60)
    print("[Phase 0] 말뭉치 포맷 검증")
    print("=" * 60)

    if not os.path.exists(CORPUS_FILE):
        print(f"  ERROR: {CORPUS_FILE} 파일을 찾을 수 없습니다.")
        sys.exit(1)

    total_lines = 0
    empty_lines = 0
    doc_count = 0
    consec_empty = 0
    max_consec_empty = 0
    in_doc = False

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            if line.strip():
                consec_empty = 0
                in_doc = True
            else:
                empty_lines += 1
                consec_empty += 1
                if consec_empty > max_consec_empty:
                    max_consec_empty = consec_empty
                if in_doc:
                    doc_count += 1
                    in_doc = False
    if in_doc:
        doc_count += 1

    file_size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)
    print(f"  파일 크기      : {file_size_mb:.1f} MB")
    print(f"  총 라인 수     : {total_lines:,}")
    print(f"  유효 라인 수   : {total_lines - empty_lines:,}")
    print(f"  빈 줄 (문서경계): {empty_lines:,}")
    print(f"  문서 수        : {doc_count:,}")
    print(f"  최대 연속 빈 줄 : {max_consec_empty}")

    if doc_count < 1000:
        print("  WARNING: 문서 수가 너무 적습니다. 병합 상태를 확인하세요.")
    if max_consec_empty > 1:
        print("  WARNING: 연속 빈 줄이 2개 이상입니다. 정제 상태를 확인하세요.")

    print(f"\n  -> 검증 완료: {doc_count:,}개 문서 확인\n")
    return doc_count


# ==========================================
# [Phase 1] 문서 파싱 (공유)
# ==========================================
def split_sentences(text):
    """한국어 문장 분리 (마침표·물음표·느낌표 뒤 공백 기준)"""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def doc_generator():
    """cleaned_corpus.txt → 문서 단위 파싱 (빈 줄 = 문서 경계)"""
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
        # 마지막 문서 처리
        if doc_lines:
            full_text = " ".join(doc_lines)
            sentences = split_sentences(full_text)
            if sentences:
                yield {"sentences": sentences}


def build_document_index():
    """문서 인덱스 구축 (baseline/scratch 공유)"""
    if os.path.exists(INTERMEDIATE_DIR):
        print(f"[Phase 1] 기존 문서 인덱스 로드: {INTERMEDIATE_DIR}")
        doc_dataset = load_from_disk(INTERMEDIATE_DIR)
        print(f"  -> 문서: {len(doc_dataset):,}개\n")
        return doc_dataset

    print("[Phase 1] 문서 인덱스 구축 중 (최초 1회)...")
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
# [Phase 2] 세그먼트 패킹 + NSP 생성
# ==========================================
def create_segments_from_doc(sentences, tokenizer, max_len, target_len):
    """문서의 문장들을 토큰 길이 기반으로 세그먼트로 패킹"""
    segments = []
    current_segment = []
    current_len = 0

    for sent in sentences:
        token_ids = tokenizer.encode(sent, add_special_tokens=False)
        len_sent = len(token_ids)

        # 매우 긴 문장: 단독 세그먼트로 분할
        if len_sent > max_len - 20:
            if current_segment:
                segments.append(" ".join(current_segment))
                current_segment = []
                current_len = 0
            for i in range(0, len_sent, target_len):
                sub_tokens = token_ids[i : i + target_len]
                segments.append(tokenizer.decode(sub_tokens))
            continue

        # 현재 세그먼트에 추가 가능하면 추가
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
    """NSP 학습용 (sentence_a, sentence_b, label) 쌍 생성"""
    total_docs = len(doc_dataset)
    offset = total_docs // 2

    for doc_idx, item in enumerate(
        tqdm(doc_dataset, desc="  [Phase 2] NSP 쌍 생성", mininterval=2.0, ncols=80)
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

            # 50% 확률로 실제 다음 세그먼트(IsNext) 또는 랜덤 세그먼트(NotNext)
            if random.random() < 0.5:
                seg_b = segments[i + 1]
                label = 0  # IsNext
            else:
                target_idx = (doc_idx + offset) % total_docs
                rand_sentences = doc_dataset[target_idx]["sentences"]

                if not rand_sentences:
                    seg_b = segments[i + 1]
                    label = 0
                else:
                    start = random.randint(0, max(0, len(rand_sentences) - 5))
                    seg_b = " ".join(rand_sentences[start : start + 5])
                    label = 1  # NotNext

            yield {
                "sentence_a": seg_a,
                "sentence_b": seg_b,
                "next_sentence_label": label,
            }


# ==========================================
# [Phase 3] 모드별 전처리 (토크나이징 + 저장)
# ==========================================
def process_mode(mode, doc_dataset):
    """baseline 또는 scratch 모드에 대해 전처리 수행"""
    print("=" * 60)
    print(f"[Phase 2-3] Mode: {mode.upper()}")
    print("=" * 60)

    # --- 토크나이저 준비 ---
    if mode == "baseline":
        print(f"  토크나이저 로드: {BASELINE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
        # baseline 토크나이저도 명시적으로 저장 (pretrain 시 로컬에서 로드)
        os.makedirs(BASELINE_TOKENIZER_DIR, exist_ok=True)
        tokenizer.save_pretrained(BASELINE_TOKENIZER_DIR)
        print(f"  -> 토크나이저 저장 완료: {BASELINE_TOKENIZER_DIR}")
        output_dir = BASELINE_OUTPUT
    else:  # scratch
        if os.path.exists(SCRATCH_TOKENIZER_DIR) and os.listdir(SCRATCH_TOKENIZER_DIR):
            print(f"  기존 커스텀 토크나이저 로드: {SCRATCH_TOKENIZER_DIR}")
            tokenizer = AutoTokenizer.from_pretrained(SCRATCH_TOKENIZER_DIR)
        else:
            print("  커스텀 WordPiece 토크나이저 학습 중...")
            base_tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)

            def text_iterator():
                with open(CORPUS_FILE, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line

            tokenizer = base_tokenizer.train_new_from_iterator(
                text_iterator(),
                vocab_size=SCRATCH_VOCAB_SIZE,
            )
            os.makedirs(SCRATCH_TOKENIZER_DIR, exist_ok=True)
            tokenizer.save_pretrained(SCRATCH_TOKENIZER_DIR)
            print(f"  -> 토크나이저 저장 완료: {SCRATCH_TOKENIZER_DIR}")
        output_dir = SCRATCH_OUTPUT

    print(f"  Vocab size: {len(tokenizer):,}")

    # --- NSP 쌍 생성 ---
    random.seed(SEED)  # 재현성을 위해 모드마다 시드 재설정

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

    # --- Train / Val 분리 ---
    print(f"  데이터셋 분리 (val_size={VAL_SIZE})...")
    split_datasets = nsp_dataset.train_test_split(
        test_size=VAL_SIZE, seed=SEED, shuffle=True, keep_in_memory=False
    )

    # --- 토크나이징 ---
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

    # --- 저장 ---
    os.makedirs(output_dir, exist_ok=True)
    tokenized.save_to_disk(output_dir)
    print(f"  -> 저장 완료: {output_dir}")
    print(f"     Train: {len(tokenized['train']):,}  |  Val: {len(tokenized['test']):,}\n")


# ==========================================
# [메인 실행부]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="BERT MLM+NSP 전처리 파이프라인")
    # 기본값은 baseline. scratch/all 은 말뭉치 전체로 토크나이저를 따로 학습해
    # 2.scratch/tokenizer 에 저장하는 초기 변형이며, 그 경로에 토크나이저가 있으면
    # 3.build_dataset_scratch.py 가 KCI 초록 학습을 건너뛰고 그것을 그대로 쓴다.
    # 최종 실험(KCI 토크나이저)을 재현하려면 baseline 만 돌리고 3번 스크립트로 넘어간다.
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "scratch", "all"],
        help="전처리 모드 선택 (default: baseline)",
    )
    args = parser.parse_args()

    # Phase 0: 말뭉치 포맷 검증
    verify_corpus()

    # Phase 1: 문서 인덱스 구축 (공유)
    doc_dataset = build_document_index()

    # Phase 2-3: 모드별 전처리
    modes = ["baseline", "scratch"] if args.mode == "all" else [args.mode]
    for mode in modes:
        process_mode(mode, doc_dataset)

    print("=" * 60)
    print("전처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
