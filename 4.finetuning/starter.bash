#!/bin/bash
# 두 모델 파인튜닝 순차 실행
# Usage: bash starter.bash
#
# 선행 조건: 2.preprocessing/2.finetune_data/ 의 1~3번 스크립트로
#            다운스트림 데이터셋을 먼저 준비한다.

# pipefail 필수: 아래 실행은 전부 `python ... | tee` 형태인데, 파이프라인의 종료
# 코드는 마지막 명령(tee)의 것이라 python 이 죽어도 0 이 된다. set -e 만으로는
# 첫 모델이 실패해도 다음 모델로 넘어가 '정상 완료'처럼 끝난다.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

banner() {
    echo ""
    echo "=========================================="
    echo " $1"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
}

banner "[1/2] DAP (BERT_Base)"
cd "$SCRIPT_DIR/1.dap"
python -u run_finetuning.py 2>&1 | tee -a "$LOG_DIR/log_dap.txt"

banner "[2/2] Scratch (BERT_Scratch)"
cd "$SCRIPT_DIR/2.scratch"
python -u run_finetuning.py 2>&1 | tee -a "$LOG_DIR/log_scratch.txt"

banner "완료"
