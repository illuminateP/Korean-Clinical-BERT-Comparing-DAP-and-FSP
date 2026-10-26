"""
3.convert_kbmc.py - KBMC NER 데이터셋 4-Label (IO 체계) 변환 스크립트

[목표]
원본 데이터의 극단적인 띄어쓰기(노이즈)를 훼손하지 않고 그대로 유지하면서,
B/I 태그 구분을 없애고 4개의 클래스(Disease, Body, Treatment, O)로 단순화함.

[결과]
  - 띄어쓰기 및 토큰 수: 원본과 100% 동일하게 유지
  - 태그 체계: Disease-B, Disease-I -> Disease 등 접미사 제거 (4-label)
  - 실험 스크립트의 KBMC_TAG_MAP, NER_ID_TO_TAG, num_labels 수정 필요

Usage:
    python 3.convert_kbmc.py
"""

import os
import json
import random
from collections import Counter

from datasets import Dataset

# ─── 경로 ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

JSON_PATH = os.path.join(DATA, "2.preprocessing", "finetune", "json", "KBMC", "train.json")
SAVED_DIR = os.path.join(DATA, "2.preprocessing", "finetune", "saved", "KBMC")

SEED = 42


# ═══════════════════════════════════════════════════
# 1. 데이터 로드
# ═══════════════════════════════════════════════════
def load_raw_kbmc():
    samples = []
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


# ═══════════════════════════════════════════════════
# 2. IO 라벨 변환 (띄어쓰기 유지)
# ═══════════════════════════════════════════════════
def convert_tags_to_io_schema(morphemes, tags):
    """
    형태소는 전혀 건드리지 않고, 태그에서 -B, -I 접미사만 제거하여 
    Disease, Body, Treatment, O 의 4가지 클래스로 매핑합니다.
    """
    new_tags = []
    for tag in tags:
        if tag == "O":
            new_tags.append("O")
        else:
            # "Disease-B" 또는 "Disease-I" -> "Disease"로 변환
            entity_type = tag.split("-")[0]
            new_tags.append(entity_type)
            
    return morphemes, new_tags


def process_all_samples(raw_samples):
    processed = []

    for sample in raw_samples:
        tags = sample["Tags"].split()

        # 형태소 분절 없이 태그만 순회하며 4-Label로 변환
        new_tags = []
        for tag in tags:
            if tag == "O":
                new_tags.append("O")
            else:
                entity_type = tag.split("-")[0]
                new_tags.append(entity_type)

        processed.append({
            "Sentence": sample["Sentence"],  # 원본 문자열을 분해하지 않고 그대로 100% 복사
            "Tags": " ".join(new_tags),
        })

    return processed


# ═══════════════════════════════════════════════════
# 3. 검증
# ═══════════════════════════════════════════════════
def count_original_entities(tags):
    """BIO 체계에서 -B 태그를 세어 개체 수 카운트"""
    count = 0
    for t in tags:
        if t.endswith("-B"):
            count += 1
    return count

def count_io_entities(tags):
    """IO 체계에서 O가 아닌 연속된 태그 블록을 1개의 개체로 카운트"""
    count = 0
    prev_tag = "O"
    for t in tags:
        if t != "O" and t != prev_tag:
            count += 1
        prev_tag = t
    return count


def validate_conversion(raw_samples, processed):
    """IO 변환 결과 검증"""
    print("\n" + "=" * 60)
    print("[검증 1] 라벨 변환 정합성 및 원본 보존 검사")
    print("=" * 60)

    errors = []
    total_orig_entities = 0
    total_new_entities = 0
    total_words = 0

    orig_tag_counter = Counter()
    new_tag_counter = Counter()

    for idx, (orig, recon) in enumerate(zip(raw_samples, processed)):
        orig_morphemes = orig["Sentence"].split()
        orig_tags = orig["Tags"].split()
        new_words = recon["Sentence"].split()
        new_tags = recon["Tags"].split()

        # ── 형태소 및 띄어쓰기 100% 보존 확인 ──
        if orig["Sentence"] != recon["Sentence"]:
            errors.append(f"[Sample {idx}] 원본 띄어쓰기/문자열이 훼손되었습니다.")
            continue

        if len(new_words) != len(new_tags):
            errors.append(f"[Sample {idx}] 어절 수({len(new_words)})와 태그 수({len(new_tags)})가 불일치합니다.")

        # ── Entity 수 확인 (IO 병합에 따른 감소분 관찰) ──
        orig_entity_cnt = count_original_entities(orig_tags)
        new_entity_cnt = count_io_entities(new_tags)
        
        total_orig_entities += orig_entity_cnt
        total_new_entities += new_entity_cnt
        total_words += len(orig_morphemes)
        
        orig_tag_counter.update(orig_tags)
        new_tag_counter.update(new_tags)

    # ── 결과 출력 ──
    n = len(raw_samples)
    print(f"\n  전체 샘플 수: {n:,}")
    print(f"  총 토큰 수:   {total_words:,} (원본 띄어쓰기 보존됨)")
    
    # IO 체계 변환 시, "Disease-B Disease-B"처럼 연속된 동일 개체가 하나로 묶여 
    # 개체 수가 미세하게 줄어들 수 있으므로 이는 오류가 아닌 정상적인 동작으로 취급합니다.
    diff = total_orig_entities - total_new_entities
    print(f"  총 entity 수: {total_orig_entities:,} → {total_new_entities:,} (연속된 동일 개체 병합에 따른 감소: {diff}개)")

    print(f"\n  [원본 태그 분포 (7-Label)]")
    for tag, cnt in sorted(orig_tag_counter.items()):
        print(f"    {tag:15s}: {cnt:>8,}")

    print(f"\n  [변환 태그 분포 (4-Label)]")
    for tag, cnt in sorted(new_tag_counter.items()):
        print(f"    {tag:15s}: {cnt:>8,}")

    # ── 오류 보고 ──
    if errors:
        print(f"\n  [오류] {len(errors)}건 발견:")
        for e in errors[:20]:
            print(f"    {e}")
    else:
        print(f"\n  [검증 통과] 띄어쓰기 보존 및 라벨 변환 성공")

    return len(errors) == 0


def validate_max_seq_len(processed, max_seq_len=128):
    """각 실험 토크나이저별 max_seq_len 초과 검사"""
    print("\n" + "=" * 60)
    print(f"[검증 2] 토크나이저별 max_seq_len={max_seq_len} 초과 검사")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer, BertTokenizer
    except ImportError:
        print("  [SKIP] transformers 미설치")
        return

    # 본 저장소는 논문의 두 모델만 포함한다. 토크나이저는 2.preprocessing 단계가
    # data/2.preprocessing/{1.dap,2.scratch}/tokenizer 에 생성한다.
    experiments = [
        {
            "name": "1.dap",
            "type": "auto",
            "path": os.path.join(DATA, "2.preprocessing", "1.dap", "tokenizer"),
        },
        {
            "name": "2.scratch",
            "type": "auto",
            "path": os.path.join(DATA, "2.preprocessing", "2.scratch", "tokenizer"),
        },
    ]

    max_content_len = max_seq_len - 2  # [CLS] + [SEP]

    for exp in experiments:
        try:
            if exp["type"] == "auto":
                tokenizer = AutoTokenizer.from_pretrained(
                    exp["path"], local_files_only=True
                )
            else:
                tokenizer = BertTokenizer(
                    vocab_file=exp["path"], do_lower_case=False
                )
        except Exception as e:
            print(f"\n  [{exp['name']}] 토크나이저 로드 실패: {e}")
            continue

        overflow_count = 0
        token_lengths = []

        for sample_idx, sample in enumerate(processed):
            words = sample["Sentence"].split()
            # is_split_into_words=True를 사용하여 원본 단어 리스트를 그대로 토크나이징
            tokenized = tokenizer(words, is_split_into_words=True)
            total_tokens = len(tokenized["input_ids"]) - 2  # CLS, SEP 제외한 순수 토큰 길이

            token_lengths.append(total_tokens)
            if total_tokens > max_content_len:
                overflow_count += 1

        max_len = max(token_lengths) if token_lengths else 0
        avg_len = sum(token_lengths) / len(token_lengths) if token_lengths else 0

        print(f"\n  [{exp['name']}] vocab={tokenizer.vocab_size:,}")
        print(f"    토큰 길이: max={max_len}, avg={avg_len:.1f}")
        print(f"    초과 샘플: {overflow_count}/{len(processed)} ({overflow_count / len(processed) * 100:.1f}%)")


# ═══════════════════════════════════════════════════
# 4. 분할 & 저장
# ═══════════════════════════════════════════════════
def split_and_save(processed, split_ratio=0.9):
    print("\n" + "=" * 60)
    print(f"[저장] {split_ratio:.0%}/{1 - split_ratio:.0%} train/valid 분할 (seed={SEED})")
    print("=" * 60)

    rng = random.Random(SEED)
    indices = list(range(len(processed)))
    rng.shuffle(indices)
    split_point = int(len(processed) * split_ratio)

    splits = {
        "train": [processed[i] for i in indices[:split_point]],
        "valid": [processed[i] for i in indices[split_point:]],
    }

    for split_name, split_samples in splits.items():
        ds = Dataset.from_dict({
            "Sentence": [s["Sentence"] for s in split_samples],
            "Tags": [s["Tags"] for s in split_samples],
        })
        save_path = os.path.join(SAVED_DIR, split_name)
        os.makedirs(save_path, exist_ok=True)
        ds.save_to_disk(save_path)

        tag_counter = Counter()
        for s in split_samples:
            tag_counter.update(s["Tags"].split())
        entity_tokens = sum(v for k, v in tag_counter.items() if k != "O")
        total_tokens = sum(tag_counter.values())
        entity_ratio = entity_tokens / total_tokens * 100 if total_tokens > 0 else 0

        print(f"  [{split_name}] {len(ds):,} samples (entity ratio: {entity_ratio:.1f}%) → {save_path}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("[KBMC] 띄어쓰기 유지 및 4-Label 변환 처리")
    print("=" * 60)

    # 1. 로드
    raw_samples = load_raw_kbmc()
    print(f"  원본 로드: {JSON_PATH}")
    print(f"  샘플 수: {len(raw_samples):,}")

    # 2. 변환 처리
    print(f"\n  변환 중...")
    processed = process_all_samples(raw_samples)

    # 3. 검증 1: 정합성
    is_valid = validate_conversion(raw_samples, processed)
    if not is_valid:
        print("\n[중단] 검증 실패. 데이터를 저장하지 않습니다.")
        return

    # 4. 검증 2: 토크나이저별 max_seq_len
    validate_max_seq_len(processed)

    # 5. 분할 & 저장
    split_and_save(processed)

    # 6. 안내
    print("\n" + "=" * 60)
    print("[완료] 실험 스크립트 수정 사항:")
    print("=" * 60)
    print("""
  태그 체계가 4-Label (IO) 로 변경되었으므로 파인튜닝 스크립트에서 아래를 수정해야 합니다:

  KBMC_TAG_MAP = {
      "Body": 0,
      "Disease": 1,
      "Treatment": 2,
      "O": 3,
  }
  NER_ID_TO_TAG = {
      0: "Body", 1: "Disease", 2: "Treatment", 3: "O",
  }
  
  TASK_INFO = {
      ...
      "KBMC": {"type": "ner", "num_labels": 4, "dir_name": "KBMC"},
  }
""")


if __name__ == "__main__":
    main()