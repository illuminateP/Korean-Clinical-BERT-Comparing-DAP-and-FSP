"""
데이터셋 전처리 & 분할 스크립트
- KBMC: train.json → 9:1 train/valid 분할 (seed=42)
- KorMedQA: 4개 서브셋 (doctor/nurse/pharm/dentist) train/dev/test 변환
- 저장: datasets/saved/ (HuggingFace Dataset 포맷)

Usage:
    python 2.split.py
"""

import os
import json
import random
from collections import Counter

from datasets import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

DATASETS_DIR = os.path.join(DATA, "2.preprocessing", "finetune", "json")
SAVED_DIR = os.path.join(DATA, "2.preprocessing", "finetune", "saved")

SEED = 42

# KBMC 태그 매핑
KBMC_TAG_MAP = {
    "Disease-B": 1, "Disease-I": 1,
    "Body-B": 0, "Body-I": 0,
    "Treatment-B": 2, "Treatment-I": 2,
    "O": 3,
}
NER_ID_TO_TAG = {0: "BD", 1: "DS", 2: "ST", 3: "O"}


def split_and_save_kbmc(split_ratio=0.9):
    """KBMC: train.json → 9:1 train/valid 분할 후 HuggingFace Dataset 저장"""
    kbmc_path = os.path.join(DATASETS_DIR, "KBMC", "train.json")
    print(f"  데이터 경로: {kbmc_path}")

    samples = []
    with open(kbmc_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    total = len(samples)
    print(f"  전체 샘플 수: {total}")

    # 태그 분포
    tag_counter = Counter()
    for s in samples:
        tag_counter.update(s["Tags"].split())
    print(f"  태그 분포:")
    for tag, cnt in sorted(tag_counter.items()):
        print(f"    {tag:15s}: {cnt:>8,}")

    # 분할
    rng = random.Random(SEED)
    indices = list(range(total))
    rng.shuffle(indices)
    split_point = int(total * split_ratio)

    splits = {
        "train": [samples[i] for i in indices[:split_point]],
        "valid": [samples[i] for i in indices[split_point:]],
    }

    for split_name, split_samples in splits.items():
        # 분할별 태그 분포
        split_tag_counter = Counter()
        for s in split_samples:
            split_tag_counter.update(s["Tags"].split())
        entity_tokens = sum(v for k, v in split_tag_counter.items() if k != "O")
        total_tokens = sum(split_tag_counter.values())
        entity_ratio = entity_tokens / total_tokens * 100 if total_tokens > 0 else 0

        ds = Dataset.from_dict({
            "Sentence": [s["Sentence"] for s in split_samples],
            "Tags": [s["Tags"] for s in split_samples],
        })
        save_path = os.path.join(SAVED_DIR, "KBMC_bio", split_name)   # 원본 B/I 태그 분할본
        ds.save_to_disk(save_path)
        print(f"  [{split_name}] {len(ds)} samples (entity ratio: {entity_ratio:.1f}%) → {save_path}")


def save_kormedqa():
    """KorMedQA: 4개 서브셋 × train/dev/test → HuggingFace Dataset 저장"""
    for subset in ["doctor", "nurse", "pharm", "dentist"]:
        base_path = os.path.join(DATASETS_DIR, "KorMedQA", subset)
        if not os.path.isdir(base_path):
            print(f"  [경고] {base_path} 없음, 스킵")
            continue

        for split_name, file_name in [("train", "train.json"), ("dev", "dev.json"), ("test", "test.json")]:
            file_path = os.path.join(base_path, file_name)
            if not os.path.exists(file_path):
                print(f"  [경고] {file_path} 없음, 스킵")
                continue

            raw_samples = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw_samples.append(json.loads(line.strip()))

            ds = Dataset.from_dict({
                "question": [s["question"] for s in raw_samples],
                "A": [s.get("A") or "" for s in raw_samples],
                "B": [s.get("B") or "" for s in raw_samples],
                "C": [s.get("C") or "" for s in raw_samples],
                "D": [s.get("D") or "" for s in raw_samples],
                "E": [s.get("E") or "" for s in raw_samples],
                "answer": [s["answer"] for s in raw_samples],
                "subject": [s.get("subject", subset) for s in raw_samples],
            })
            save_path = os.path.join(SAVED_DIR, "KorMedQA", subset, split_name)
            ds.save_to_disk(save_path)
            print(f"  [{subset}/{split_name}] {len(ds)} samples → {save_path}")


def main():
    os.makedirs(SAVED_DIR, exist_ok=True)

    print("=" * 60)
    print(f"[KBMC] 9:1 분할 및 저장 (seed={SEED})")
    print("=" * 60)
    split_and_save_kbmc()

    print()
    print("=" * 60)
    print("[KorMedQA] 서브셋별 저장")
    print("=" * 60)
    save_kormedqa()

    print(f"\n[완료] 저장 경로: {SAVED_DIR}")


if __name__ == "__main__":
    main()
