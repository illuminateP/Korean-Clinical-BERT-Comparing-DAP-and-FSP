import os
import numpy as np
from datasets import load_from_disk

# 설정 변수
EPOCHS = 5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

DATASET_PATHS = {
    "DAP": os.path.join(DATA, "2.preprocessing", "1.dap", "datasets"),
    "Scratch": os.path.join(DATA, "2.preprocessing", "2.scratch", "datasets"),
}

def count_tokens(batch):
    # attention_mask에서 1로 표시된 부분의 합을 구하여 패딩을 제외한 유효 토큰을 계산합니다.
    effective_tokens = np.sum(batch["attention_mask"], axis=1)
    return {"effective_tokens": effective_tokens}

def main():
    for mode, path in DATASET_PATHS.items():
        print("=" * 60)
        print(f"[{mode} 데이터셋 분석 시작]")
        print(f"경로: {path}")
        
        if not os.path.exists(path):
            print("-> 데이터셋 경로를 찾을 수 없습니다. 건너뜁니다.\n")
            continue

        # 데이터셋 로드
        dataset = load_from_disk(path)
        
        if "train" in dataset:
            train_dataset = dataset["train"]
        else:
            train_dataset = dataset
            
        print(f"훈련 데이터 문서 수: {len(train_dataset):,}")

        # datasets 라이브러리의 map 함수는 desc 파라미터를 입력하면 자동으로 tqdm 프로그레스 바를 출력합니다.
        token_counts = train_dataset.map(
            count_tokens,
            batched=True,
            batch_size=10000,
            num_proc=4,
            remove_columns=train_dataset.column_names,
            desc=f"{mode} 토큰 수 계산 중"
        )

        # 결과 집계
        total_effective_tokens = sum(token_counts["effective_tokens"])
        max_seq_len = len(train_dataset[0]["input_ids"])
        total_padded_tokens = len(train_dataset) * max_seq_len

        # 결과 출력
        print(f"\n[1 에폭 기준 토큰 수]")
        print(f"- 패딩 포함 전체 토큰 수: {total_padded_tokens:,}")
        print(f"- 패딩 제외 유효 토큰 수: {total_effective_tokens:,}")
        
        print(f"\n[{EPOCHS} 에폭 훈련 시 모델이 연산하는 총 토큰 수]")
        print(f"- 패딩 포함 총 토큰 수: {total_padded_tokens * EPOCHS:,}")
        print(f"- 패딩 제외 유효 총 토큰 수: {total_effective_tokens * EPOCHS:,}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()