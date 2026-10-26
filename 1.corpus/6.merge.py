import os
import re
import json
import sys
import io
from collections import Counter
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

# 경로에 msd_raw.jsonl 등 추가/수정 가능
# ─────────────────────────────────────────────────────────────────────────
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
#        단계별 산출물 위치는 README의 디렉토리 규약을 따른다.
# ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA = os.path.join(REPO_ROOT, "data")

CORPUS_DATA = os.path.join(DATA, "1.corpus")
FILE_PATHS = [
    os.path.join(CORPUS_DATA, "1.wiki/outputs/merged_wiki.txt"),
    os.path.join(CORPUS_DATA, "2.aihub/outputs/merged_book.txt"),
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti.jsonl"),
    # 주의: ARTI_CONF 는 ARTI(학술지)에 학술대회를 더한 세트라, 의료 카테고리로 거르면
    #       두 파일의 내용이 같다. 원 실험은 둘 다 읽었으므로 그대로 둔다(README 2.1 각주 참고).
    #       중복 수집을 피하려면 아래 한 줄을 제외한다.
    os.path.join(CORPUS_DATA, "3.kci/outputs/kci_records_arti_conf.jsonl"),
    os.path.join(CORPUS_DATA, "4.msd/outputs/corpus_sentences.txt"),
    os.path.join(CORPUS_DATA, "5.news/outputs/news_medical_corpus.txt"),
]

OUTPUT_FILE = os.path.join(CORPUS_DATA, "merged", "final_merged_corpus.txt")

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
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 1. 카운터 및 통계 변수 초기화
    line_counter = Counter()
    heading_counter = Counter()
    
    stats = {
        'total_tokens': 0,
        'kor_tokens': 0,
        'eng_tokens': 0,
        'num_tokens': 0,
        'other_tokens': 0,
        'valid_sentences': 0,
        'total_docs': 0
    }

    valid_files = [fp for fp in FILE_PATHS if os.path.exists(fp)]
    if not valid_files:
        print("경고: 처리할 수 있는 파일이 하나도 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 {len(valid_files)}개의 파일을 병합 및 검사합니다...\n")

    # 2. 파일 읽기 및 병합
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for filepath in valid_files:
            total_lines = count_lines(filepath)
            is_jsonl = filepath.endswith('.jsonl')
            
            doc_has_content = False
            with open(filepath, 'r', encoding='utf-8') as in_f:
                for line in tqdm(in_f, total=total_lines, desc=f"처리 중: {os.path.basename(filepath)}"):
                    lines_to_process = []
                    
                    if is_jsonl:
                        try:
                            # 실시간 크롤링 시 불완전한 JSON 무시
                            data = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue
                        
                        # [MSD 포맷 처리]
                        if 'sections' in data:
                            for sec in data.get('sections', []):
                                h = sec.get('heading', '')
                                if h:
                                    heading_counter[h] += 1
                                text = sec.get('text', '')
                                for t_line in text.split('\n'):
                                    lines_to_process.append(t_line)
                                    
                        # [KCI 포맷 처리]
                        elif 'abstract' in data:
                            category = data.get("categories")
                            abstract = data.get("abstract", "")
                            
                            if category in TARGET_CATEGORIES and abstract:
                                # 한국어가 포함된 초록만 처리
                                if PATTERN_KOR.search(abstract):
                                    for a_line in abstract.split('\n'):
                                        lines_to_process.append(a_line)
                    else:
                        # [TXT 파일 처리]
                        stripped_line = line.strip()
                        if not stripped_line:
                            # 빈 줄 = 문서 경계 유지
                            if doc_has_content:
                                out_f.write('\n')
                                doc_has_content = False
                                stats['total_docs'] += 1
                        else:
                            lines_to_process.append(line)

                    # 3. 추출된 라인들에 대한 처리 (빈도 계산, 토큰 분석, 파일 저장)
                    for text_line in lines_to_process:
                        clean_line = text_line.strip()
                        if not clean_line:
                            continue
                            
                        # 요청하신 로직: 길이가 5를 초과하는 라인만 빈도 카운팅
                        if len(clean_line) > 5:
                            line_counter[clean_line] += 1
                        
                        # 토큰 검사 및 카운팅
                        tokens = clean_line.split()
                        stats['total_tokens'] += len(tokens)
                        
                        for token in tokens:
                            if PATTERN_KOR.search(token):
                                stats['kor_tokens'] += 1
                            elif PATTERN_ENG.search(token):
                                stats['eng_tokens'] += 1
                            elif PATTERN_NUM.search(token):
                                stats['num_tokens'] += 1
                            else:
                                stats['other_tokens'] += 1
                                
                        # 최종 병합 파일에 기록 (강제 필터링 없이 일단 모두 저장)
                        out_f.write(clean_line + '\n')
                        stats['valid_sentences'] += 1
                        doc_has_content = True

                    # JSONL: 각 JSON 레코드 = 하나의 문서 → 레코드 간 경계 삽입
                    if is_jsonl and doc_has_content:
                        out_f.write('\n')
                        doc_has_content = False
                        stats['total_docs'] += 1

            # 파일 끝: 마지막 문서의 경계 (빈 줄 없이 끝난 경우)
            if doc_has_content:
                out_f.write('\n')
                stats['total_docs'] += 1

    # ==========================================
    # [결과 출력]
    # ==========================================
    print("\n" + "="*50)
    print("[검사 및 병합 완료 보고서]")
    print("="*50)
    print(f"1. 처리 결과 파일: {OUTPUT_FILE}")
    print(f"2. 전체 라인(문장/단락) 수: {stats['valid_sentences']:,}개")
    print(f"3. 전체 문서 수 (빈 줄 구분): {stats['total_docs']:,}개")
    
    print("\n[토큰 통계 (띄어쓰기 기준)]")
    print(f"- 전체 토큰 수: {stats['total_tokens']:,}개")
    print(f"  * 한국어 포함 토큰: {stats['kor_tokens']:,}개")
    print(f"  * 영문 토큰(한글 미포함): {stats['eng_tokens']:,}개")
    print(f"  * 숫자 토큰(한/영 미포함): {stats['num_tokens']:,}개")
    print(f"  * 기타(특수기호 등): {stats['other_tokens']:,}개\n")

    # 요청하신 반복 텍스트 판단 파트 출력
    print('=== 반복 텍스트 라인 Top 50 (10회 이상) ===')
    for text, cnt in line_counter.most_common(50):
        if cnt >= 10:
            print(f'  [{cnt:4d}] {text[:120]}')

    print('\n=== 반복 헤딩 Top 30 (5회 이상) ===')
    for text, cnt in heading_counter.most_common(30):
        if cnt >= 5:
            print(f'  [{cnt:4d}] {text}')

if __name__ == "__main__":
    process_and_inspect()