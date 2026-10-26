"""
파인튜닝 데이터셋 다운로더
- KorMedQA:  HuggingFace sean0042/KorMedMCQA
- KBMC:      HuggingFace SungJoo/KBMC

다운로드 위치:
    원본(Arrow): datasets/origin/{데이터셋명}/
    JSON:        datasets/json/{데이터셋명}/

Usage:
    python 1.download.py                    # 전체 다운로드
    python 1.download.py --only KorMedQA    # 특정 데이터셋만
    python 1.download.py --only KBMC
"""

import os
import shutil
import argparse
from datasets import load_dataset

# ──────────────────────────────────────────────
# 경로 설정
# ──────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

ORIGIN_ROOT = os.path.join(DATA, "2.preprocessing", "finetune", "origin")
JSON_ROOT = os.path.join(DATA, "2.preprocessing", "finetune", "json")

# ──────────────────────────────────────────────
# 데이터셋 소스 정의
# ──────────────────────────────────────────────
DATASETS = {
    "KorMedQA": {
        "repo": "sean0042/KorMedMCQA",
        "subsets": ["doctor", "nurse", "pharm", "dentist"],
        "description": "Korean Medical MCQA (국시 5지선다)",
    },
    "KBMC": {
        "repo": "SungJoo/KBMC",
        "subsets": [],
        "description": "Korean Medical NER (B-BD, B-DS, B-ST 등 BIO 태깅)",
    },
}


# ──────────────────────────────────────────────
# 다운로드 함수
# ──────────────────────────────────────────────
def download_kormedqa(origin_dir, json_dir):
    """KorMedQA: sean0042/KorMedMCQA (doctor/nurse/pharm/dentist)"""
    info = DATASETS["KorMedQA"]
    print(f"\n{'='*60}")
    print(f"[KorMedQA] {info['description']}")
    print(f"  repo:   {info['repo']}")
    print(f"  origin: {origin_dir}")
    print(f"  json:   {json_dir}")
    print(f"{'='*60}")

    for d in (origin_dir, json_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  기존 폴더 삭제 → {d}")

    for subset in info["subsets"]:
        print(f"  [{subset}] 다운로드 중...")
        ds = load_dataset(info["repo"], name=subset)

        ds.save_to_disk(os.path.join(origin_dir, subset))
        for split_name, split_ds in ds.items():
            split_ds.to_json(os.path.join(json_dir, subset, f"{split_name}.json"))
            print(f"    {split_name}: {len(split_ds):,}건")

    print(f"[KorMedQA] 완료\n")


def download_kbmc(origin_dir, json_dir):
    """KBMC: SungJoo/KBMC (Korean Medical NER)"""
    info = DATASETS["KBMC"]
    print(f"\n{'='*60}")
    print(f"[KBMC] {info['description']}")
    print(f"  repo:   {info['repo']}")
    print(f"  origin: {origin_dir}")
    print(f"  json:   {json_dir}")
    print(f"{'='*60}")

    for d in (origin_dir, json_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"  기존 폴더 삭제 → {d}")

    print(f"  다운로드 중...")

    ds = load_dataset(info["repo"])
    ds.save_to_disk(origin_dir)
    for split_name, split_ds in ds.items():
        split_ds.to_json(os.path.join(json_dir, f"{split_name}.json"))
        print(f"    {split_name}: {len(split_ds):,}건")

    print(f"[KBMC] 완료\n")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
DOWNLOADERS = {
    "KorMedQA": download_kormedqa,
    "KBMC":     download_kbmc,
}


def main():
    parser = argparse.ArgumentParser(description="파인튜닝 데이터셋 다운로더")
    parser.add_argument(
        "--only", type=str, default=None,
        choices=list(DOWNLOADERS.keys()),
        help="특정 데이터셋만 다운로드 (미지정 시 전체)",
    )
    args = parser.parse_args()

    os.makedirs(ORIGIN_ROOT, exist_ok=True)
    os.makedirs(JSON_ROOT, exist_ok=True)
    print(f"[origin] {ORIGIN_ROOT}")
    print(f"[json]   {JSON_ROOT}")

    targets = [args.only] if args.only else list(DOWNLOADERS.keys())

    for name in targets:
        origin_dir = os.path.join(ORIGIN_ROOT, name)
        json_dir = os.path.join(JSON_ROOT, name)
        DOWNLOADERS[name](origin_dir, json_dir)

    # 최종 요약
    print("=" * 60)
    print("[요약] 다운로드 결과:")
    for name in DOWNLOADERS:
        origin_path = os.path.join(ORIGIN_ROOT, name)
        json_path = os.path.join(JSON_ROOT, name)
        origin_ok = "OK" if os.path.exists(origin_path) else "미완료"
        json_ok = "OK" if os.path.exists(json_path) else "미완료"
        print(f"  {name:12s} → origin: {origin_ok}, json: {json_ok}")
    print("=" * 60)


if __name__ == "__main__":
    main()
