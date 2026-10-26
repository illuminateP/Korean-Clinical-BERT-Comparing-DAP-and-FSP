import os
import re
import json
import sys
import io
from tqdm import tqdm
from transformers import AutoTokenizer

# 터미널 출력과 로그 파일 저장을 동시에 수행하기 위한 클래스
class Logger(object):
    def __init__(self, filename="process_log.txt"):
        # Windows 콘솔 한글 깨짐 방지를 위해 io.TextIOWrapper 적용
        self.terminal = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') if hasattr(sys.stdout, 'buffer') else sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 로그는 5.evaluation/logs/log_corpus_stats.txt 에 남긴다 (다시 실행하면 덮어쓴다)
log_filename = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "log_corpus_stats.txt"))
sys.stdout = Logger(log_filename)

# ==========================================
# [설정]
# ==========================================
TARGET_CATEGORIES = [
    '가정의학', '간호학', '기타의약학', '내과학', '마취과학', '면역학', 
    '물리치료학', '방사선과학', '병리학', '비뇨기과학', '산부인과학', '성형외과학', 
    '소아과학', '수의학', '신경과학', '신경외과학', '약리학', '약학', '예방의학', 
    '응급의학', '의공학', '의약학', '의학일반', '일반외과학', '임상병리학', 
    '작업치료학', '재활의학', '정신과학', '정형외과학', '치의학', '피부과학', 
    '한의학', '해부학'
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
CORPUS_DATA = os.path.join(DATA, "1.corpus")
PREP = os.path.join(DATA, "2.preprocessing")

FILE_PATHS = [
    os.path.join(CORPUS_DATA, "1.wiki/outputs/merged_wiki.txt"),
    os.path.join(CORPUS_DATA, "2.aihub/outputs/merged_book.txt"),
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti.jsonl"),
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti_conf.jsonl"),
    os.path.join(CORPUS_DATA, "4.msd/outputs/corpus_sentences.txt"),
    os.path.join(CORPUS_DATA, "5.news/outputs/news_medical_corpus.txt"),
]

BASELINE_TOKENIZER_PATH = os.path.join(PREP, "1.dap", "tokenizer")
SCRATCH_V2_TOKENIZER_PATH = os.path.join(PREP, "2.scratch", "tokenizer")

PATTERN_KOR = re.compile(r'[가-힣]')
PATTERN_ENG = re.compile(r'[a-zA-Z]')
PATTERN_NUM = re.compile(r'[0-9]')

# ==========================================
# [함수]
# ==========================================
def count_lines(filepath):
    """tqdm 진행률을 위한 전체 라인 수 계산"""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count

def process_and_inspect():
    # 1. 토크나이저 로드
    print("토크나이저를 로드하는 중입니다...")
    try:
        tokenizer_baseline = AutoTokenizer.from_pretrained(BASELINE_TOKENIZER_PATH)
        tokenizer_scratch_v2 = AutoTokenizer.from_pretrained(SCRATCH_V2_TOKENIZER_PATH)
    except Exception as e:
        print(f"토크나이저 로드 실패. 경로를 확인해주세요: {e}")
        return
    print("토크나이저 로드 완료.\n")

    # 2. 개별 출처 및 전체 통계 변수 초기화
    source_stats = {}
    global_stats = {
        'total_tokens': 0, 'kor_tokens': 0, 'eng_tokens': 0,
        'num_tokens': 0, 'other_tokens': 0, 'valid_sentences': 0, 'total_docs': 0,
        'baseline_tokens': 0, 'scratch_v2_tokens': 0
    }

    valid_files = [fp for fp in FILE_PATHS if os.path.exists(fp)]
    if not valid_files:
        print("경고: 처리할 수 있는 source 파일이 없습니다.")
        return

    print(f"총 {len(valid_files)}개의 source 파일 통계를 분석합니다...\n")

    # 3. Source 파일(1~5) 읽기 및 통계 분석
    for filepath in valid_files:
        source_name = os.path.basename(filepath)
        source_stats[source_name] = {
            'total_docs': 0, 'valid_sentences': 0, 'total_tokens': 0, 
            'kor_tokens': 0, 'eng_tokens': 0, 'num_tokens': 0, 'other_tokens': 0,
            'baseline_tokens': 0, 'scratch_v2_tokens': 0
        }

        total_lines = count_lines(filepath)
        is_jsonl = filepath.endswith('.jsonl')
        doc_has_content = False
        
        with open(filepath, 'r', encoding='utf-8') as in_f:
            for line in tqdm(in_f, total=total_lines, desc=f"Source 분석 중: {source_name}"):
                lines_to_process = []
                
                if is_jsonl:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    extracted_lines = []
                    if 'sections' in data:
                        for sec in data.get('sections', []):
                            text = sec.get('text', '')
                            for t_line in text.split('\n'):
                                extracted_lines.append(t_line)
                    elif 'abstract' in data:
                        category = data.get("categories")
                        abstract = data.get("abstract", "")
                        
                        if category in TARGET_CATEGORIES and abstract:
                            if PATTERN_KOR.search(abstract):
                                for a_line in abstract.split('\n'):
                                    extracted_lines.append(a_line)

                    lines_to_process = [l.strip() for l in extracted_lines if l.strip()]

                    if lines_to_process:
                        source_stats[source_name]['total_docs'] += 1
                        global_stats['total_docs'] += 1

                else:
                    stripped_line = line.strip()
                    if not stripped_line:
                        if doc_has_content:
                            source_stats[source_name]['total_docs'] += 1
                            global_stats['total_docs'] += 1
                            doc_has_content = False
                    else:
                        lines_to_process.append(stripped_line)
                        doc_has_content = True

                # 추출된 라인들에 대한 토큰 및 서브워드 분석
                for clean_line in lines_to_process:
                    source_stats[source_name]['valid_sentences'] += 1
                    global_stats['valid_sentences'] += 1
                    
                    # 어절(Whitespace) 기반 토큰
                    tokens = clean_line.split()
                    token_count = len(tokens)
                    
                    source_stats[source_name]['total_tokens'] += token_count
                    global_stats['total_tokens'] += token_count
                    
                    for token in tokens:
                        if PATTERN_KOR.search(token):
                            source_stats[source_name]['kor_tokens'] += 1
                            global_stats['kor_tokens'] += 1
                        elif PATTERN_ENG.search(token):
                            source_stats[source_name]['eng_tokens'] += 1
                            global_stats['eng_tokens'] += 1
                        elif PATTERN_NUM.search(token):
                            source_stats[source_name]['num_tokens'] += 1
                            global_stats['num_tokens'] += 1
                        else:
                            source_stats[source_name]['other_tokens'] += 1
                            global_stats['other_tokens'] += 1

                    # 서브워드 토크나이저 기반 토큰 비교
                    baseline_len = len(tokenizer_baseline.encode(clean_line, add_special_tokens=False))
                    scratch_v2_len = len(tokenizer_scratch_v2.encode(clean_line, add_special_tokens=False))

                    source_stats[source_name]['baseline_tokens'] += baseline_len
                    source_stats[source_name]['scratch_v2_tokens'] += scratch_v2_len
                    global_stats['baseline_tokens'] += baseline_len
                    global_stats['scratch_v2_tokens'] += scratch_v2_len

        if not is_jsonl and doc_has_content:
            source_stats[source_name]['total_docs'] += 1
            global_stats['total_docs'] += 1

    # ==========================================
    # [결과 출력]
    # ==========================================
    print("\n" + "="*60)
    print("[출처별 통계 분석 결과]")
    print("="*60)
    for source, s_stats in source_stats.items():
        print(f"■ {source}")
        print(f"  - 문서 수: {s_stats['total_docs']:,}개")
        print(f"  - 문장/단락 수: {s_stats['valid_sentences']:,}개")
        print(f"  - Whitespace 토큰 수: {s_stats['total_tokens']:,}개")
        print(f"    (한글: {s_stats['kor_tokens']:,} / 영문: {s_stats['eng_tokens']:,} / 숫자: {s_stats['num_tokens']:,})")
        print(f"  - Baseline Tokenizer 토큰 수: {s_stats['baseline_tokens']:,}개")
        print(f"  - Scratch_v2 Tokenizer 토큰 수: {s_stats['scratch_v2_tokens']:,}개")
        print("-" * 60)

    print("\n" + "="*60)
    print("[전체 누적 통계 (1~5 출처 합계)]")
    print("="*60)
    print(f"- 전체 문서 수: {global_stats['total_docs']:,}개")
    print(f"- 전체 문장/단락 수: {global_stats['valid_sentences']:,}개")
    print(f"- Whitespace 토큰 수: {global_stats['total_tokens']:,}개")
    print(f"  * Baseline Tokenizer 전체 토큰 수: {global_stats['baseline_tokens']:,}개")
    print(f"  * Scratch_v2 Tokenizer 전체 토큰 수: {global_stats['scratch_v2_tokens']:,}개")
    print("="*60 + "\n")
    print(f"분석 완료. 로그가 {log_filename}에 저장되었습니다.")

if __name__ == "__main__":
    process_and_inspect()