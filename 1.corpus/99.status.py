import os
import re
import json
import sys
import io
from tqdm import tqdm

# Windows 콘솔 등에서 한글 출력 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CORPUS_DATA = os.path.join(REPO_ROOT, "data", "1.corpus")

FILE_PATHS = [
    os.path.join(CORPUS_DATA, "1.wiki/outputs/merged_wiki.txt"),
    os.path.join(CORPUS_DATA, "2.aihub/outputs/merged_book.txt"),
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti.jsonl"),
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti_conf.jsonl"),
    os.path.join(CORPUS_DATA, "4.msd/outputs/corpus_sentences.txt"),
    os.path.join(CORPUS_DATA, "5.news/outputs/news_medical_corpus.txt"),
]

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
    # 1. 개별 출처 및 전체 통계 변수 초기화
    source_stats = {}
    global_stats = {
        'total_tokens': 0, 'kor_tokens': 0, 'eng_tokens': 0,
        'num_tokens': 0, 'other_tokens': 0, 'valid_sentences': 0, 'total_docs': 0
    }

    valid_files = [fp for fp in FILE_PATHS if os.path.exists(fp)]
    if not valid_files:
        print("경고: 처리할 수 있는 파일이 하나도 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 {len(valid_files)}개의 파일 통계를 분석합니다...\n")

    # 2. 파일 읽기 및 통계 분석
    for filepath in valid_files:
        source_name = os.path.basename(filepath)
        source_stats[source_name] = {
            'total_docs': 0, 'valid_sentences': 0, 'total_tokens': 0, 
            'kor_tokens': 0, 'eng_tokens': 0, 'num_tokens': 0, 'other_tokens': 0
        }

        total_lines = count_lines(filepath)
        is_jsonl = filepath.endswith('.jsonl')
        doc_has_content = False
        
        with open(filepath, 'r', encoding='utf-8') as in_f:
            for line in tqdm(in_f, total=total_lines, desc=f"분석 중: {source_name}"):
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

                    # 빈 줄 제외
                    lines_to_process = [l.strip() for l in extracted_lines if l.strip()]

                    # JSONL은 1줄이 1문서이므로 유효한 데이터가 있으면 카운트 증가
                    if lines_to_process:
                        source_stats[source_name]['total_docs'] += 1
                        global_stats['total_docs'] += 1

                else:
                    stripped_line = line.strip()
                    if not stripped_line:
                        # 빈 줄을 만나면 이전까지 작성된 내용이 있을 때 문서 1개로 산정
                        if doc_has_content:
                            source_stats[source_name]['total_docs'] += 1
                            global_stats['total_docs'] += 1
                            doc_has_content = False
                    else:
                        lines_to_process.append(stripped_line)
                        doc_has_content = True

                # 3. 추출된 라인들에 대한 토큰 분석
                for clean_line in lines_to_process:
                    source_stats[source_name]['valid_sentences'] += 1
                    global_stats['valid_sentences'] += 1
                    
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

        # TXT 파일의 마지막 문서 경계 처리 (파일이 빈 줄 없이 끝난 경우)
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
        print(f"  - 총 토큰 수: {s_stats['total_tokens']:,}개")
        print(f"    (한글: {s_stats['kor_tokens']:,} / 영문: {s_stats['eng_tokens']:,} / 숫자: {s_stats['num_tokens']:,})")
        print("-" * 60)

    print("\n[전체 누적 통계]")
    print(f"- 전체 문서 수: {global_stats['total_docs']:,}개")
    print(f"- 전체 문장/단락 수: {global_stats['valid_sentences']:,}개")
    print(f"- 전체 토큰 수: {global_stats['total_tokens']:,}개")
    print(f"  * 한국어 포함 토큰: {global_stats['kor_tokens']:,}개")
    print(f"  * 영문 토큰(한글 미포함): {global_stats['eng_tokens']:,}개")
    print(f"  * 숫자 토큰(한/영 미포함): {global_stats['num_tokens']:,}개")
    print(f"  * 기타(특수기호 등): {global_stats['other_tokens']:,}개\n")

if __name__ == "__main__":
    process_and_inspect()