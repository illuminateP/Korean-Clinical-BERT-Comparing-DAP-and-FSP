import os
from tqdm import tqdm

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
MSD_OUT = os.path.join(DATA, "1.corpus", "4.msd", "outputs")


def calculate_corpus_tokens(file_path=None):
    if file_path is None:
        file_path = os.path.join(MSD_OUT, "corpus_sentences.txt")
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    total_word_tokens = 0
    
    # 파일 크기를 확인하여 tqdm 진행 바의 기준으로 삼습니다.
    file_size = os.path.getsize(file_path)
    
    print(f"[{os.path.basename(file_path)}] 토큰 계산 시작...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 가시성을 위해 tqdm을 사용하며, 파일 읽기 진행 상황을 바이트 단위로 업데이트합니다.
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Counting tokens") as pbar:
            for line in f:
                line_text = line.strip()
                if line_text:
                    # 공백 기준 어절 분리 및 카운트
                    tokens = line_text.split()
                    total_word_tokens += len(tokens)
                
                # 읽어들인 바이트 수만큼 진행 바 업데이트
                pbar.update(len(line.encode('utf-8')))
                
    print("\n" + "="*50)
    print(f"  총 어절(Word) 토큰 수 : {total_word_tokens:,} 개")
    print("="*50)

if __name__ == "__main__":
    calculate_corpus_tokens()