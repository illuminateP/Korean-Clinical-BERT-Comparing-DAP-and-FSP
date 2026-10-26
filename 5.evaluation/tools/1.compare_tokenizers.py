#!/usr/bin/env python3
"""
1.compare_tokenizers.py - DAP vs Scratch vs Scratch_v2 토크나이저 비교 분석

전체 말뭉치(cleaned_corpus.txt)를 세 가지 토크나이저로 토크나이징하여
어떤 단어를 어떻게 다르게 분절하는지 비교 분석합니다.

모든 출력은 콘솔 + 5.evaluation/logs/log_compare_tokenizers.txt 에 동시 기록됩니다.

Usage:
    python 1.compare_tokenizers.py
    python 1.compare_tokenizers.py --sample_lines 500000
"""

import os
import sys
import io
import re
import argparse
from collections import Counter, defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ==========================================
# [설정]
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
PREP = os.path.join(DATA, "2.preprocessing")

CORPUS_FILE = os.path.join(PREP, "cleaned", "cleaned_corpus.txt")

# 본 저장소는 논문의 두 모델(BERT_Base=DAP, BERT_Scratch)만 포함한다.
# SCRATCH_TOKENIZER 는 초기 실험의 중간 변형(20 epoch 판본)으로, 해당 실험은
# 저장소에서 제외되어 기본 경로가 생성되지 않는다. 3자 비교를 재현하려면
# --scratch_tokenizer 로 직접 경로를 지정한다.
DAP_TOKENIZER = os.path.join(PREP, "1.dap", "tokenizer")
SCRATCH_TOKENIZER = os.path.join(PREP, "2.scratch_e20", "tokenizer")   # 저장소 미포함
SCRATCH_V2_TOKENIZER = os.path.join(PREP, "2.scratch", "tokenizer")

LOG_FILE = os.path.join(REPO_ROOT, "5.evaluation", "logs", "log_compare_tokenizers.txt")


class DualWriter:
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, text):
        sys.__stdout__.buffer.write(text.encode("utf-8", errors="replace"))
        sys.__stdout__.buffer.flush()
        self.file.write(text)

    def flush(self):
        sys.__stdout__.buffer.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def print_section(title, writer):
    border = "=" * 90
    writer.write(f"\n{border}\n{title}\n{border}\n")


# ==========================================
# [1] 어휘(Vocab) 수준 비교
# ==========================================
def analyze_vocab(dap_tok, scratch_tok, scratch_v2_tok, writer):
    print_section("[1] 어휘(Vocab) 수준 비교", writer)

    dap_vocab = set(dap_tok.get_vocab().keys())
    scratch_vocab = set(scratch_tok.get_vocab().keys())
    scratch_v2_vocab = set(scratch_v2_tok.get_vocab().keys())

    shared_all = dap_vocab & scratch_vocab & scratch_v2_vocab
    only_dap = dap_vocab - (scratch_vocab | scratch_v2_vocab)
    only_scratch = scratch_vocab - (dap_vocab | scratch_v2_vocab)
    only_scratch_v2 = scratch_v2_vocab - (dap_vocab | scratch_vocab)

    writer.write(f"  DAP vocab      : {len(dap_vocab):,}\n")
    writer.write(f"  Scratch vocab  : {len(scratch_vocab):,}\n")
    writer.write(f"  Scratch_v2 vocab: {len(scratch_v2_vocab):,}\n")
    writer.write(f"  세 토크나이저 공통 : {len(shared_all):,}\n")
    writer.write(f"  DAP만 보유      : {len(only_dap):,}\n")
    writer.write(f"  Scratch만 보유  : {len(only_scratch):,}\n")
    writer.write(f"  Scratch_v2만 보유: {len(only_scratch_v2):,}\n")

    for name, vocab in [("DAP", dap_vocab), ("Scratch", scratch_vocab), ("Scratch_v2", scratch_v2_vocab)]:
        subword = sum(1 for t in vocab if t.startswith("##"))
        whole = len(vocab) - subword
        writer.write(f"\n  [{name}] 전체 단어: {whole:,} | 서브워드(##): {subword:,}\n")

    for name, only_vocab in [("Scratch", only_scratch), ("Scratch_v2", only_scratch_v2), ("DAP", only_dap)]:
        kor_only = sorted(
            [t for t in only_vocab if re.search(r"[\uac00-\ud7a3]", t)],
            key=len,
            reverse=True,
        )
        writer.write(f"\n  [{name}에만 있는 한국어 토큰] (상위 100개, 긴 순서)\n")
        for i, t in enumerate(kor_only[:100]):
            writer.write(f"    {i + 1:3d}. {t}\n")


# ==========================================
# [2] 전체 말뭉치 토큰 효율성 비교
# ==========================================
def analyze_corpus_efficiency(dap_tok, scratch_tok, scratch_v2_tok, writer, sample_lines=0):
    print_section("[2] 전체 말뭉치 토큰 효율성 비교", writer)

    if not os.path.exists(CORPUS_FILE):
        writer.write(f"  ERROR: {CORPUS_FILE} 없음\n")
        return {}, {}, {}

    dap_total, scratch_total, scratch_v2_total = 0, 0, 0
    dap_sub, scratch_sub, scratch_v2_sub = 0, 0, 0
    line_count, doc_count = 0, 0
    in_doc = False

    word_tokenization_diff = defaultdict(lambda: {"dap": Counter(), "scratch": Counter(), "scratch_v2": Counter()})
    
    # DAP 대비 효율이 좋은 단어 수집
    scratch_wins = Counter()
    scratch_v2_wins = Counter()

    dap_len = defaultdict(list)
    scratch_len = defaultdict(list)
    scratch_v2_len = defaultdict(list)

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  말뭉치 분석 중"):
            stripped = line.strip()
            if not stripped:
                if in_doc:
                    doc_count += 1
                    in_doc = False
                continue

            in_doc = True
            if sample_lines > 0 and line_count >= sample_lines:
                break
            line_count += 1

            d_toks = dap_tok.tokenize(stripped)
            s_toks = scratch_tok.tokenize(stripped)
            s2_toks = scratch_v2_tok.tokenize(stripped)

            dap_total += len(d_toks)
            scratch_total += len(s_toks)
            scratch_v2_total += len(s2_toks)

            dap_sub += sum(1 for t in d_toks if t.startswith("##"))
            scratch_sub += sum(1 for t in s_toks if t.startswith("##"))
            scratch_v2_sub += sum(1 for t in s2_toks if t.startswith("##"))

            words = stripped.split()
            for word in words:
                if len(word) <= 1 or not re.search(r"[\uac00-\ud7a3a-zA-Z0-9]", word):
                    continue

                wd_toks = dap_tok.tokenize(word)
                ws_toks = scratch_tok.tokenize(word)
                ws2_toks = scratch_v2_tok.tokenize(word)

                word_tokenization_diff[word]["dap"][" ".join(wd_toks)] += 1
                word_tokenization_diff[word]["scratch"][" ".join(ws_toks)] += 1
                word_tokenization_diff[word]["scratch_v2"][" ".join(ws2_toks)] += 1

                char_len = len(word)
                dap_len[char_len].append(len(wd_toks))
                scratch_len[char_len].append(len(ws_toks))
                scratch_v2_len[char_len].append(len(ws2_toks))

                diff_s1 = len(wd_toks) - len(ws_toks)
                diff_s2 = len(wd_toks) - len(ws2_toks)

                if diff_s1 > 0:
                    scratch_wins[word] += diff_s1
                if diff_s2 > 0:
                    scratch_v2_wins[word] += diff_s2

    if in_doc:
        doc_count += 1

    writer.write(f"\n  분석 범위: {line_count:,}개 라인 / {doc_count:,}개 문서\n")
    writer.write(f"\n  [전체 토큰 수]\n")
    writer.write(f"    DAP       : {dap_total:,} 토큰\n")
    writer.write(f"    Scratch   : {scratch_total:,} 토큰 (DAP 대비 {scratch_total/dap_total*100:.1f}%)\n")
    writer.write(f"    Scratch_v2: {scratch_v2_total:,} 토큰 (DAP 대비 {scratch_v2_total/dap_total*100:.1f}%)\n")

    writer.write(f"\n  [서브워드(##) 토큰 비율]\n")
    if dap_total > 0:
        writer.write(f"    DAP       : {dap_sub:,} ({dap_sub / dap_total * 100:.1f}%)\n")
        writer.write(f"    Scratch   : {scratch_sub:,} ({scratch_sub / scratch_total * 100:.1f}%)\n")
        writer.write(f"    Scratch_v2: {scratch_v2_sub:,} ({scratch_v2_sub / scratch_v2_total * 100:.1f}%)\n")

    writer.write(f"\n  [단어 길이(글자수)별 평균 분절 토큰 수]\n")
    writer.write(f"  {'글자수':>5s} | {'DAP':>8s} | {'Scratch':>8s} | {'Scratch_v2':>10s}\n")
    writer.write(f"  {'-'*5} | {'-'*8} | {'-'*8} | {'-'*10}\n")
    for clen in sorted(dap_len.keys()):
        if clen > 15: continue
        d_avg = sum(dap_len[clen]) / len(dap_len[clen])
        s_avg = sum(scratch_len[clen]) / len(scratch_len[clen])
        s2_avg = sum(scratch_v2_len[clen]) / len(scratch_v2_len[clen])
        writer.write(f"  {clen:>5d} | {d_avg:>8.2f} | {s_avg:>8.2f} | {s2_avg:>10.2f}\n")

    return scratch_wins, scratch_v2_wins, word_tokenization_diff


# ==========================================
# [3] 분절 차이가 큰 단어 Top-N
# ==========================================
def show_top_differences(scratch_wins, scratch_v2_wins, word_diff, dap_tok, scratch_tok, scratch_v2_tok, writer):
    print_section("[3] 분절 차이가 드러나는 단어 비교 (DAP 대비 효율성, 빈도 가중)", writer)

    def print_top_wins(wins_counter, title, target_tok):
        writer.write(f"\n  [{title}] Top 100\n")
        writer.write(f"  {'순위':>4s}  {'단어':>15s} | {'DAP':>25s} | {'Target':>25s} | {'절약':>4s}\n")
        writer.write(f"  {'-'*4}  {'-'*15} | {'-'*25} | {'-'*25} | {'-'*4}\n")

        for rank, (word, score) in enumerate(wins_counter.most_common(100), 1):
            d_toks = dap_tok.tokenize(word)
            t_toks = target_tok.tokenize(word)
            writer.write(f"  {rank:>4d}  {word:>15s} | {' '.join(d_toks):>25s} | {' '.join(t_toks):>25s} | {len(d_toks)-len(t_toks):>+4d}\n")

    print_top_wins(scratch_wins, "Scratch가 DAP보다 적은 토큰으로 표현하는 단어", scratch_tok)
    print_top_wins(scratch_v2_wins, "Scratch_v2가 DAP보다 적은 토큰으로 표현하는 단어", scratch_v2_tok)


# ==========================================
# [4] 도메인 키워드 분절 비교
# ==========================================
def compare_domain_terms(dap_tok, scratch_tok, scratch_v2_tok, writer):
    print_section("[4] 의료 도메인 키워드 분절 비교", writer)

    medical_terms = [
        "당뇨병", "고혈압", "인슐린", "항생제", "동맥경화", "심근경색",
        "뇌졸중", "자기공명영상", "컴퓨터단층촬영", "내시경", "치매",
        "diabetes", "hypertension", "insulin", "antibiotics",
        "electrocardiogram", "pharmacokinetics"
    ]

    writer.write(f"\n  {'용어':>18s} | {'DAP':>22s} | {'Scratch':>22s} | {'Scratch_v2':>22s}\n")
    writer.write(f"  {'-'*18} | {'-'*22} | {'-'*22} | {'-'*22}\n")

    for term in medical_terms:
        d_str = f"{' '.join(dap_tok.tokenize(term))} ({len(dap_tok.tokenize(term))})"
        s_str = f"{' '.join(scratch_tok.tokenize(term))} ({len(scratch_tok.tokenize(term))})"
        s2_str = f"{' '.join(scratch_v2_tok.tokenize(term))} ({len(scratch_v2_tok.tokenize(term))})"
        writer.write(f"  {term:>18s} | {d_str:>22s} | {s_str:>22s} | {s2_str:>22s}\n")


# ==========================================
# [5] 샘플 문장 토크나이징 비교
# ==========================================
def compare_sample_sentences(dap_tok, scratch_tok, scratch_v2_tok, writer):
    print_section("[5] 샘플 문장 토크나이징 비교", writer)

    sentences = [
        "당뇨병 환자에게 인슐린을 피하주사로 투여하였다.",
        "자기공명영상(MRI) 검사에서 뇌경색 소견이 관찰되었다.",
        "항암화학요법 후 호중구감소증이 발생하여 항생제를 투여하였다.",
    ]

    for sent in sentences:
        d_toks = dap_tok.tokenize(sent)
        s_toks = scratch_tok.tokenize(sent)
        s2_toks = scratch_v2_tok.tokenize(sent)

        writer.write(f"\n  원문: {sent}\n")
        writer.write(f"  DAP        ({len(d_toks):>3d}토큰): {' '.join(d_toks)}\n")
        writer.write(f"  Scratch    ({len(s_toks):>3d}토큰): {' '.join(s_toks)}\n")
        writer.write(f"  Scratch_v2 ({len(s2_toks):>3d}토큰): {' '.join(s2_toks)}\n")


# ==========================================
# [6] 고빈도 단어 분절 패턴 분석
# ==========================================
def analyze_frequent_words(word_diff, dap_tok, scratch_tok, scratch_v2_tok, writer, top_n=100):
    print_section("[6] 고빈도 단어 분절 패턴 비교", writer)

    word_freq = {word: sum(data["dap"].values()) for word, data in word_diff.items()}
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    writer.write(f"\n  [고빈도 단어 중 분절이 다른 단어] Top {top_n}\n")
    writer.write(f"  {'순위':>4s}  {'단어':>12s} {'빈도':>8s} | {'DAP':>20s} | {'Scratch':>20s} | {'Scratch_v2':>20s}\n")
    writer.write(f"  {'-'*4}  {'-'*12} {'-'*8} | {'-'*20} | {'-'*20} | {'-'*20}\n")

    rank = 0
    for word, freq in sorted_words:
        if rank >= top_n: break

        d_toks = dap_tok.tokenize(word)
        s_toks = scratch_tok.tokenize(word)
        s2_toks = scratch_v2_tok.tokenize(word)

        if d_toks == s_toks == s2_toks:
            continue

        rank += 1
        writer.write(f"  {rank:>4d}  {word:>12s} {freq:>8,} | {' '.join(d_toks):>20s} | {' '.join(s_toks):>20s} | {' '.join(s2_toks):>20s}\n")


# ==========================================
# [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DAP vs Scratch vs Scratch_v2 토크나이저 비교 분석")
    parser.add_argument("--sample_lines", type=int, default=0, help="분석할 라인 수 제한")
    args = parser.parse_args()

    writer = DualWriter(LOG_FILE)
    writer.write(f"{'=' * 90}\n토크나이저 비교 분석 리포트\n")
    writer.write(f"DAP        : {DAP_TOKENIZER}\nScratch    : {SCRATCH_TOKENIZER}\nScratch_v2 : {SCRATCH_V2_TOKENIZER}\n")
    writer.write(f"말뭉치     : {CORPUS_FILE}\n{'=' * 90}\n")

    writer.write("\n토크나이저 로딩 중...\n")
    missing = [(n, p) for n, p in [("DAP", DAP_TOKENIZER),
                                   ("Scratch(20ep, 저장소 미포함)", SCRATCH_TOKENIZER),
                                   ("Scratch", SCRATCH_V2_TOKENIZER)] if not os.path.isdir(p)]
    if missing:
        writer.write("\n[중단] 다음 토크나이저를 찾을 수 없습니다. 2.preprocessing 단계를 먼저 실행하십시오.\n")
        for n, p in missing:
            writer.write(f"  - {n}: {p}\n")
        return

    dap_tok = AutoTokenizer.from_pretrained(DAP_TOKENIZER)
    scratch_tok = AutoTokenizer.from_pretrained(SCRATCH_TOKENIZER)
    scratch_v2_tok = AutoTokenizer.from_pretrained(SCRATCH_V2_TOKENIZER)

    analyze_vocab(dap_tok, scratch_tok, scratch_v2_tok, writer)
    s_wins, s2_wins, word_diff = analyze_corpus_efficiency(dap_tok, scratch_tok, scratch_v2_tok, writer, args.sample_lines)
    show_top_differences(s_wins, s2_wins, word_diff, dap_tok, scratch_tok, scratch_v2_tok, writer)
    compare_domain_terms(dap_tok, scratch_tok, scratch_v2_tok, writer)
    compare_sample_sentences(dap_tok, scratch_tok, scratch_v2_tok, writer)
    if word_diff: analyze_frequent_words(word_diff, dap_tok, scratch_tok, scratch_v2_tok, writer)

    print_section("분석 완료", writer)
    writer.write(f"  결과 저장: {LOG_FILE}\n")
    writer.close()
    print(f"\n  -> 전체 결과가 {LOG_FILE} 에 저장되었습니다.")

if __name__ == "__main__":
    main()