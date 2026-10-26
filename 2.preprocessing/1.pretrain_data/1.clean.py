import os
import re
import sys
import io
from collections import Counter
from tqdm import tqdm

# Windows 콘솔 등에서 한글 출력 깨짐 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================
# [설정]
# ==========================================
# ─────────────────────────────────────────────────────────────────────────
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
#        단계별 산출물 위치는 README의 디렉토리 규약을 따른다.
# ─────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")

INPUT_FILE = os.path.join(DATA, "1.corpus", "merged", "final_merged_corpus.txt")
OUTPUT_DIR = os.path.join(DATA, "2.preprocessing", "cleaned")
OUTPUT_FILE = f"{OUTPUT_DIR}/cleaned_corpus.txt"

# [강력한 블랙리스트 (소문자 기준)]
# 이 단어들 중 하나라도 포함되면 문장 전체를 가차 없이 버립니다.
DROP_KEYWORDS = [
    # 뉴스/언론사/광고 찌꺼기
    "editor's pick", "editors' pick", "editor pick", "편집자 주", "편집자주",
    "스타일난다", "기업이 미래다", "경제기사 이렇게 읽어요", "세계지식포럼",
    "의료계 단신", "의료계 소식", "논설위원", "skylife", "무단전재", "재배포 금지", 
    "재배포금지", "기자 =", "일정>", "출판사 승인",
    
    # 위키/HTML/마크다운 찌꺼기
    "width 100%", "width=", "!wiki", "파일 external", "상위 항목", 
    "분류:", "step 1", "step 2", "step 3", "step 4",
    "icd0", "icd9", "omim", "정당 ="
]

# [정규식 제거 패턴 (문장 안에서 해당 부분만 도려냄)]
PATTERN_HTML_TAGS = re.compile(r'<[^>]+>') # <div>, <span> 등 잔류 태그
PATTERN_URL = re.compile(r'https?://\S+|www\.\S+')
PATTERN_PIXEL = re.compile(r'\d+x\d+px')
PATTERN_WIKI_BOLD = re.compile(r"''+") # '''동선''' -> 동선
PATTERN_WIKI_ATTR = re.compile(r'(width\s*=?\s*\d+%?|align\s*=\s*["\']?\w+["\']?)', re.IGNORECASE)

PATTERN_KOR = re.compile(r'[가-힣]')

# ==========================================
# [함수]
# ==========================================
def count_lines(filepath):
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count

def is_garbage_line(text):
    """이 라인이 쓰레기 데이터인지 판별 (소문자 변환 후 검사)"""
    if len(text) < 5:
        return True
        
    text_lower = text.lower()
    
    # 블랙리스트 키워드가 하나라도 포함되어 있으면 드랍
    if any(keyword in text_lower for keyword in DROP_KEYWORDS):
        return True
        
    # 특수기호 비율이 비정상적으로 높으면 드랍 (의학 데이터 보호를 위해 기준을 40%로 완화하되 표 찌꺼기 방어)
    symbol_count = sum(text.count(c) for c in ['|', '!', '{', '}', '=', '[', ']'])
    if len(text) > 0 and (symbol_count / len(text)) > 0.4:
        return True

    return False

def clean_text(text):
    """살려둘 문장 내부의 불필요한 속성과 기호 청소"""
    # URL, HTML 태그, 픽셀, 위키 강조, HTML 속성 잔해 도려내기
    text = PATTERN_URL.sub('', text)
    text = PATTERN_HTML_TAGS.sub('', text)
    text = PATTERN_PIXEL.sub('', text)
    text = PATTERN_WIKI_BOLD.sub('', text)
    text = PATTERN_WIKI_ATTR.sub('', text)
    
    # 문장 앞뒤에 쓸데없이 붙은 마크다운 특수기호 벗겨내기 
    # (예: "◆ 기업이 미래다 ◆" -> 양끝 기호 제거, "* 특징은" -> "특징은")
    text = text.strip(' *-=|!◆[]<>:;')
    
    # 다중 공백을 하나의 공백으로 압축
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def process_and_inspect():
    if not os.path.exists(INPUT_FILE):
        print(f"오류: 입력 파일을 찾을 수 없습니다. ({INPUT_FILE})")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    line_counter = Counter()
    stats = {
        'total_lines': 0, 'valid_lines': 0, 'dropped_lines': 0,
        'total_tokens': 0, 'kor_tokens': 0,
    }

    print(f"[정제 시작] {INPUT_FILE} -> {OUTPUT_FILE}\n")
    total_lines_count = count_lines(INPUT_FILE)

    prev_was_boundary = True  # 파일 시작은 경계로 취급 (중복 빈 줄 방지)
    with open(INPUT_FILE, 'r', encoding='utf-8') as in_f, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:

        for line in tqdm(in_f, total=total_lines_count, desc="정제 및 검사 중"):
            stats['total_lines'] += 1
            raw_text = line.strip()

            # 0. 빈 줄 = 문서 경계 유지 (연속 빈 줄 방지)
            if not raw_text:
                if not prev_was_boundary:
                    out_f.write('\n')
                    prev_was_boundary = True
                continue

            # 1. 가비지 필터링 (블랙리스트 기반 문장 전체 드랍)
            if is_garbage_line(raw_text):
                stats['dropped_lines'] += 1
                continue
                
            # 2. 텍스트 내부 청소 (속성/태그/불필요한 앞뒤 기호 제거)
            cleaned = clean_text(raw_text)
            
            # 청소 후 너무 짧아지면 버림
            if len(cleaned) < 5:
                stats['dropped_lines'] += 1
                continue
                
            # 3. 토큰 통계 수집
            tokens = cleaned.split()
            stats['total_tokens'] += len(tokens)
            for token in tokens:
                if PATTERN_KOR.search(token):
                    stats['kor_tokens'] += 1
            
            line_counter[cleaned] += 1
            
            # 4. 파일 저장
            out_f.write(cleaned + '\n')
            stats['valid_lines'] += 1
            prev_was_boundary = False

    # ==========================================
    # [결과 출력]
    # ==========================================
    print("\n" + "="*50)
    print("[정제 완료 보고서 v2]")
    print("="*50)
    print(f"- 원본 라인 수: {stats['total_lines']:,}개")
    print(f"- 제거된 가비지 라인: {stats['dropped_lines']:,}개")
    print(f"- 최종 유효 라인: {stats['valid_lines']:,}개")
    print(f"- 유효 토큰 수: {stats['total_tokens']:,}개 (한국어 포함: {stats['kor_tokens']:,}개)\n")

    print('=== [정제 후] 반복 텍스트 라인 Top 50 ===')
    for text, cnt in line_counter.most_common(50):
        if cnt >= 5:
            print(f'  [{cnt:4d}] {text[:120]}')

if __name__ == "__main__":
    process_and_inspect()