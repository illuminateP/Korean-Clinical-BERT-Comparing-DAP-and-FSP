#!/usr/bin/env python3
"""
4.inspect.py - 전처리 결과 전수조사 및 검증

모든 전처리가 올바르게 완료되었는지 검증합니다:
  1. 원본 말뭉치 (cleaned_corpus.txt) 언어 비율 분석
  2. 토크나이즈된 데이터셋 전수검사
     - [CLS], [SEP] 토큰 위치 및 개수 검증
     - token_type_ids 전환 지점 검증
     - 전체 토큰 수 통계
     - NSP 레이블 분포
     - 시퀀스 길이 분포
  3. 한국어/영어 비율 (문자 수준 + 토큰 수준)

Usage:
    python 4.inspect.py                  # 전체 검사 (기본)
    python 4.inspect.py --max_samples 1000  # 빠른 샘플 검사
"""

import os
import sys
import io
import re
import argparse
from collections import Counter
from tqdm import tqdm
from datasets import load_from_disk
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
BASELINE_MODEL = "klue/bert-base"
BASELINE_TOKENIZER = os.path.join(DATA, "2.preprocessing", "1.dap", "tokenizer")
BASELINE_DATASET = os.path.join(DATA, "2.preprocessing", "1.dap", "datasets")
SCRATCH_TOKENIZER = os.path.join(DATA, "2.preprocessing", "2.scratch", "tokenizer")
SCRATCH_DATASET = os.path.join(DATA, "2.preprocessing", "2.scratch", "datasets")


# ==========================================
# [1] 원본 말뭉치 분석
# ==========================================
def inspect_corpus():
    """cleaned_corpus.txt 한국어/영어 비율 및 기본 통계 분석"""
    print("=" * 70)
    print("[1] 원본 말뭉치 분석: cleaned_corpus.txt")
    print("=" * 70)

    if not os.path.exists(CORPUS_FILE):
        print(f"  -> {CORPUS_FILE} 없음, 건너뜁니다.\n")
        return

    kor_chars = 0
    eng_chars = 0
    num_chars = 0
    other_chars = 0
    total_lines = 0
    total_docs = 0
    in_doc = False
    line_lengths = []

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  말뭉치 분석"):
            stripped = line.strip()
            if not stripped:
                if in_doc:
                    total_docs += 1
                    in_doc = False
                continue

            in_doc = True
            total_lines += 1
            line_lengths.append(len(stripped))

            for ch in stripped:
                if "\uac00" <= ch <= "\ud7a3":
                    kor_chars += 1
                elif ch.isascii() and ch.isalpha():
                    eng_chars += 1
                elif ch.isdigit():
                    num_chars += 1
                elif not ch.isspace():
                    other_chars += 1

    if in_doc:
        total_docs += 1

    total_chars = kor_chars + eng_chars + num_chars + other_chars
    avg_line_len = sum(line_lengths) / len(line_lengths) if line_lengths else 0

    file_size_mb = os.path.getsize(CORPUS_FILE) / (1024 * 1024)

    print(f"\n  파일 크기    : {file_size_mb:.1f} MB")
    print(f"  총 문서 수   : {total_docs:,}")
    print(f"  총 라인 수   : {total_lines:,}")
    print(f"  평균 라인 길이: {avg_line_len:.1f} 문자")
    print(f"\n  [문자 수준 언어 비율] (공백 제외, 총 {total_chars:,}자)")
    print(f"    한국어 : {kor_chars:,} ({kor_chars / total_chars * 100:.1f}%)")
    print(f"    영어   : {eng_chars:,} ({eng_chars / total_chars * 100:.1f}%)")
    print(f"    숫자   : {num_chars:,} ({num_chars / total_chars * 100:.1f}%)")
    print(f"    기타   : {other_chars:,} ({other_chars / total_chars * 100:.1f}%)")
    print()


# ==========================================
# [2] 토크나이즈된 데이터셋 전수검사
# ==========================================
def inspect_dataset(name, dataset_path, tokenizer, max_samples=0):
    """전처리된 데이터셋에 대한 전수조사"""
    print("=" * 70)
    print(f"[검사] {name}")
    print(f"       Path: {dataset_path}")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"  -> 데이터셋 없음, 건너뜁니다.\n")
        return

    datasets = load_from_disk(dataset_path)

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    print(f"  [토크나이저 정보]")
    print(f"    Vocab size : {len(tokenizer):,}")
    print(f"    CLS id={cls_id}, SEP id={sep_id}, PAD id={pad_id}")
    print()

    for split_name, dataset in datasets.items():
        n_total = len(dataset)
        n_check = min(n_total, max_samples) if max_samples > 0 else n_total

        print(f"  --- Split: {split_name} ({n_total:,} samples, 검사: {n_check:,}) ---")

        # 통계 변수
        total_tokens = 0
        total_non_pad_tokens = 0
        cls_position_errors = 0
        cls_count_errors = 0
        sep_count_errors = 0
        token_type_errors = 0
        nsp_label_dist = Counter()
        seq_lengths = []

        # 토큰 수준 한국어/영어 분석 (샘플링)
        kor_tokens = 0
        eng_tokens = 0
        other_tokens = 0
        decoded_sample_count = 0
        DECODE_SAMPLE_INTERVAL = max(1, n_check // 1000)  # ~1000개 샘플만 디코딩

        error_examples = []  # 오류 예시 저장 (최대 5개)

        for idx in tqdm(range(n_check), desc=f"    전수검사 ({split_name})", ncols=80):
            example = dataset[idx]
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            token_type_ids = example["token_type_ids"]
            nsp_label = example["next_sentence_label"]

            seq_len = len(input_ids)
            non_pad = sum(1 for m in attention_mask if m == 1)

            total_tokens += seq_len
            total_non_pad_tokens += non_pad
            seq_lengths.append(non_pad)

            # --- NSP 레이블 분포 ---
            nsp_label_dist[nsp_label] += 1

            # --- [CLS] 위치 검사: 반드시 첫 번째 토큰 ---
            if input_ids[0] != cls_id:
                cls_position_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(
                        f"[CLS위치오류] idx={idx}, first_token={input_ids[0]}"
                    )

            # --- [CLS] 개수 검사: 정확히 1개 ---
            cls_count = sum(1 for x in input_ids if x == cls_id)
            if cls_count != 1:
                cls_count_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(
                        f"[CLS개수오류] idx={idx}, cls_count={cls_count}"
                    )

            # --- [SEP] 개수 검사: 정확히 2개 ---
            sep_positions = [i for i, x in enumerate(input_ids) if x == sep_id]
            if len(sep_positions) != 2:
                sep_count_errors += 1
                if len(error_examples) < 5:
                    error_examples.append(
                        f"[SEP개수오류] idx={idx}, sep_count={len(sep_positions)}, positions={sep_positions[:5]}"
                    )
            else:
                # --- token_type_ids 전환 검사 ---
                # 올바른 구조: [CLS] seg_A [SEP] seg_B [SEP] [PAD]...
                # token_type_ids: 0...0(seg_A+첫[SEP]) 1...1(seg_B+둘째[SEP]) 0...0(PAD)
                first_sep = sep_positions[0]
                second_sep = sep_positions[1]

                has_error = False

                # seg_A 영역 (0 ~ first_sep): 모두 0이어야 함
                for pos in range(first_sep + 1):
                    if token_type_ids[pos] != 0:
                        has_error = True
                        break

                # seg_B 영역 (first_sep+1 ~ second_sep): 모두 1이어야 함
                if not has_error:
                    for pos in range(first_sep + 1, second_sep + 1):
                        if token_type_ids[pos] != 1:
                            has_error = True
                            break

                # padding 영역 (second_sep+1 ~ end): 모두 0이어야 함
                if not has_error:
                    for pos in range(second_sep + 1, seq_len):
                        if token_type_ids[pos] != 0:
                            has_error = True
                            break

                if has_error:
                    token_type_errors += 1
                    if len(error_examples) < 5:
                        error_examples.append(
                            f"[TT오류] idx={idx}, sep_pos={sep_positions}"
                        )

            # --- 토큰 수준 한국어/영어 비율 (샘플링) ---
            if idx % DECODE_SAMPLE_INTERVAL == 0:
                decoded_sample_count += 1
                # 실제 토큰만 확인 (패딩 제외)
                real_tokens = [
                    input_ids[i]
                    for i in range(seq_len)
                    if attention_mask[i] == 1
                    and input_ids[i] not in (cls_id, sep_id, pad_id)
                ]
                for tid in real_tokens:
                    decoded = tokenizer.decode([tid])
                    if re.search(r"[\uac00-\ud7a3]", decoded):
                        kor_tokens += 1
                    elif re.search(r"[a-zA-Z]", decoded):
                        eng_tokens += 1
                    else:
                        other_tokens += 1

        # ---- 결과 출력 ----
        avg_len = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        min_len = min(seq_lengths) if seq_lengths else 0
        max_len = max(seq_lengths) if seq_lengths else 0

        # 길이 분포 구간
        len_bins = Counter()
        for sl in seq_lengths:
            if sl <= 64:
                len_bins["~64"] += 1
            elif sl <= 128:
                len_bins["65~128"] += 1
            elif sl <= 256:
                len_bins["129~256"] += 1
            elif sl <= 384:
                len_bins["257~384"] += 1
            else:
                len_bins["385~512"] += 1

        print(f"\n    [토큰 통계]")
        print(f"      총 토큰 수 (패딩 포함) : {total_tokens:,}")
        print(f"      유효 토큰 수 (패딩 제외): {total_non_pad_tokens:,}")
        print(f"      평균 시퀀스 길이        : {avg_len:.1f}")
        print(f"      최소 / 최대 길이        : {min_len} / {max_len}")
        print(f"      길이 분포:")
        for bin_name in ["~64", "65~128", "129~256", "257~384", "385~512"]:
            cnt = len_bins.get(bin_name, 0)
            pct = cnt / n_check * 100 if n_check > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"        {bin_name:>8s}: {cnt:>8,} ({pct:5.1f}%) {bar}")

        print(f"\n    [구분자 검증 (전수조사: {n_check:,}건)]")
        print(f"      [CLS] 위치 오류 (첫 토큰 != CLS) : {cls_position_errors:,}")
        print(f"      [CLS] 개수 오류 (count != 1)     : {cls_count_errors:,}")
        print(f"      [SEP] 개수 오류 (count != 2)     : {sep_count_errors:,}")
        print(f"      token_type_ids 전환 오류         : {token_type_errors:,}")

        total_errors = (
            cls_position_errors
            + cls_count_errors
            + sep_count_errors
            + token_type_errors
        )
        if total_errors == 0:
            print(f"      -> PASS: 구분자 검증 통과")
        else:
            print(f"      -> FAIL: 총 {total_errors:,}건 오류 발견")
            for ex in error_examples:
                print(f"         {ex}")

        print(f"\n    [NSP 레이블 분포]")
        for label in sorted(nsp_label_dist.keys()):
            count = nsp_label_dist[label]
            label_name = "IsNext" if label == 0 else "NotNext"
            ratio = count / n_check * 100 if n_check > 0 else 0
            print(f"      {label_name:8s} (label={label}): {count:>10,} ({ratio:.1f}%)")

        # 토큰 수준 한국어/영어 비율
        total_sampled_tokens = kor_tokens + eng_tokens + other_tokens
        if total_sampled_tokens > 0:
            print(
                f"\n    [토큰 수준 언어 비율] (샘플: {decoded_sample_count:,}건 디코딩)"
            )
            print(
                f"      한국어 토큰: {kor_tokens:,} ({kor_tokens / total_sampled_tokens * 100:.1f}%)"
            )
            print(
                f"      영어 토큰  : {eng_tokens:,} ({eng_tokens / total_sampled_tokens * 100:.1f}%)"
            )
            print(
                f"      기타 토큰  : {other_tokens:,} ({other_tokens / total_sampled_tokens * 100:.1f}%)"
            )

        # 샘플 출력
        print(f"\n    [디코딩 샘플 (첫 번째 예시)]")
        sample = dataset[0]
        decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=False)
        # [PAD] 이후 제거하여 가독성 확보
        decoded_clean = decoded.split("[PAD]")[0].strip()
        print(f"      {decoded_clean[:300]}")
        print(f"      NSP label: {'IsNext' if sample['next_sentence_label'] == 0 else 'NotNext'}")

        # [SEP] 위치 시각화
        sep_pos = [i for i, x in enumerate(sample["input_ids"]) if x == sep_id]
        cls_pos = [i for i, x in enumerate(sample["input_ids"]) if x == cls_id]
        print(f"      [CLS] 위치: {cls_pos}")
        print(f"      [SEP] 위치: {sep_pos}")
        print()


# ==========================================
# [3] 토크나이저 비교 (baseline vs scratch)
# ==========================================
def compare_tokenizers():
    """baseline과 scratch 토크나이저의 어휘 비교"""
    print("=" * 70)
    print("[3] 토크나이저 비교: baseline vs scratch")
    print("=" * 70)

    if not os.path.exists(SCRATCH_TOKENIZER):
        print("  -> scratch 토크나이저 없음, 건너뜁니다.\n")
        return

    base_tok = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER)
    scratch_tok = AutoTokenizer.from_pretrained(SCRATCH_TOKENIZER)

    base_vocab = set(base_tok.get_vocab().keys())
    scratch_vocab = set(scratch_tok.get_vocab().keys())

    shared = base_vocab & scratch_vocab
    only_base = base_vocab - scratch_vocab
    only_scratch = scratch_vocab - base_vocab

    print(f"  Baseline vocab  : {len(base_vocab):,}")
    print(f"  Scratch vocab   : {len(scratch_vocab):,}")
    print(f"  공통 토큰       : {len(shared):,}")
    print(f"  Baseline만 보유 : {len(only_base):,}")
    print(f"  Scratch만 보유  : {len(only_scratch):,}")

    # 의료 도메인 관련 토큰 샘플
    medical_terms = ["환자", "진단", "치료", "수술", "증상", "약물", "투여", "질환", "병원", "의사"]
    print(f"\n  [의료 용어 토큰화 비교]")
    print(f"  {'용어':>6s}  |  {'Baseline':>30s}  |  {'Scratch':>30s}")
    print(f"  {'-' * 6}  |  {'-' * 30}  |  {'-' * 30}")
    for term in medical_terms:
        base_tokens = base_tok.tokenize(term)
        scratch_tokens = scratch_tok.tokenize(term)
        print(
            f"  {term:>6s}  |  {' '.join(base_tokens):>30s}  |  {' '.join(scratch_tokens):>30s}"
        )

    # 샘플 문장 토큰화 비교
    sample_sentence = "당뇨병 환자에게 인슐린을 피하주사로 투여하였다."
    print(f"\n  [샘플 문장 토큰화 비교]")
    print(f"  문장: {sample_sentence}")
    base_toks = base_tok.tokenize(sample_sentence)
    scratch_toks = scratch_tok.tokenize(sample_sentence)
    print(f"  Baseline ({len(base_toks)}토큰): {' '.join(base_toks)}")
    print(f"  Scratch  ({len(scratch_toks)}토큰): {' '.join(scratch_toks)}")
    print()


# ==========================================
# [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="전처리 결과 전수조사 및 검증")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="검사할 최대 샘플 수 (0=전수조사, 기본값)",
    )
    args = parser.parse_args()

    # 1. 원본 말뭉치 분석
    inspect_corpus()

    # 2. Baseline 데이터셋 전수검사
    if os.path.exists(BASELINE_DATASET) and os.path.exists(BASELINE_TOKENIZER):
        tokenizer = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER)
        inspect_dataset(
            "Baseline (klue/bert-base)",
            BASELINE_DATASET,
            tokenizer,
            max_samples=args.max_samples,
        )

    # 3. Scratch 데이터셋 전수검사
    if os.path.exists(SCRATCH_DATASET) and os.path.exists(SCRATCH_TOKENIZER):
        tokenizer = AutoTokenizer.from_pretrained(SCRATCH_TOKENIZER)
        inspect_dataset(
            "Scratch (Custom Tokenizer)",
            SCRATCH_DATASET,
            tokenizer,
            max_samples=args.max_samples,
        )

    # 4. 토크나이저 비교
    compare_tokenizers()

    print("=" * 70)
    print("검증 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()
