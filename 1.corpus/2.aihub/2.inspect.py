import os, re, json, html, random, string, gc
from collections import Counter
from pathlib import Path
from datasets import Dataset, Features, Value, ClassLabel, load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
AIHUB_DIR = os.path.join(DATA, "1.corpus", "2.aihub")


# ==========================================
# [설정]
# ==========================================
BASE_DIR = os.path.join(AIHUB_DIR, "origin")

# 국문 원천데이터 JSON 디렉토리 (08, 09)
JSON_DIRS = [
    f'{BASE_DIR}/08.전문 의학지식 데이터/Training/01.원천데이터/TS_국문_기타',
    f'{BASE_DIR}/08.전문 의학지식 데이터/Training/01.원천데이터/TS_국문_온라인 의료 정보 제공 사이트',
    f'{BASE_DIR}/08.전문 의학지식 데이터/Training/01.원천데이터/TS_국문_의학 교과서',
    f'{BASE_DIR}/08.전문 의학지식 데이터/Training/01.원천데이터/TS_국문_학술 논문 및 저널',
    f'{BASE_DIR}/08.전문 의학지식 데이터/Training/01.원천데이터/TS_국문_학회 가이드라인',
    f'{BASE_DIR}/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_국문_기타',
    f'{BASE_DIR}/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_국문_온라인 의료 정보 제공 사이트',
    f'{BASE_DIR}/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_국문_의학 교과서',
    f'{BASE_DIR}/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_국문_학술 논문 및 저널',
    f'{BASE_DIR}/09.필수의료 의학지식 데이터/3.개방데이터/1.데이터/Training/01.원천데이터/TS_국문_학회 가이드라인',
]

# 154 의료/법률 서적 TXT (Training + Validation 모두 프리트레인에 사용)
TXT_DIRS = [
    f'{BASE_DIR}/154.의료, 법률 전문 서적 말뭉치/Training/01.원천데이터/TS_01.의료',
    f'{BASE_DIR}/154.의료, 법률 전문 서적 말뭉치/Validation/01.원천데이터/VS_01.의료'
]

INTERMEDIATE_DIR = f'{BASE_DIR}/output/intermediate_docs'
OUTPUT_DIR       = f'{BASE_DIR}/output/bert_pretrain_dataset'
MODEL_NAME       = 'klue/bert-base'
MAX_SEQ_LENGTH   = 512
TARGET_SEG_LEN   = 250
VAL_SIZE         = 5000
SEED             = 42

NUM_PROC = 42

random.seed(SEED)
os.makedirs(f'{BASE_DIR}/output', exist_ok=True)
print('설정 완료')


print('=' * 60)
json_files, txt_files = [], []
for d in JSON_DIRS:
    p = Path(d)
    if p.exists():
        fs = list(p.glob('*.json'))
        json_files.extend(fs)
        print(f'  JSON [{len(fs):>6}]  {p.name}')
    else:
        print(f'  [MISSING] {d}')
for d in TXT_DIRS:
    p = Path(d)
    if p.exists():
        fs = list(p.glob('*.txt'))
        txt_files.extend(fs)
        print(f'  TXT  [{len(fs):>6}]  {p.name}')
    else:
        print(f'  [MISSING] {d}')
print('-' * 60)
print(f'  총 JSON: {len(json_files):,}  |  총 TXT: {len(txt_files):,}  |  합계: {len(json_files)+len(txt_files):,}')


print('[JSON 샘플]')
for fp in json_files[:2]:
    with open(fp, encoding='utf-8-sig') as f:
        d = json.load(f)
    content = d.get('content', '')
    print(f'  파일: {fp.name}  키: {list(d.keys())}')
    print(f'  content ({len(content)}자): {content[:120]}')
    print()
print('[TXT 샘플]')
if txt_files:
    with open(txt_files[0], encoding='utf-8-sig') as f:
        print(f'  파일: {txt_files[0].name}')
        print(f'  {f.read(200)}')


# ==========================================
# [단계 1] 텍스트 정제 (Hangeulize 없음 - 의료 영문 용어 보존)
# ==========================================
PUNCT_MAPPING = {
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u20b9': 'e', '\u00b4': "'", '\u00b0': '', '\u20ac': 'e', '\u2122': 'tm',
    '\u221a': ' sqrt ', '\u00d7': 'x', '\u00b2': '2', '\u2014': '-', '\u2013': '-',
    '\u00a3': 'e', '\u221e': 'infinity', '\u03b8': 'theta', '\u00f7': '/',
    '\u03b1': 'alpha', '\u2022': '.', '\u00e0': 'a', '\u2212': '-',
    '\u03b2': 'beta', '\u2205': '', '\u00b3': '3', '\u03c0': 'pi',
    '\u200b': ' ', '\u2026': ' ... ', '\ufeff': '', '\u00b7': '.'
}
_PUNCT     = re.escape(string.punctuation)
_WHITELIST = re.compile(rf'[^A-Za-z0-9\u3131-\u3163\uac00-\ud7a3{_PUNCT}\s]')
_HTML      = re.compile(r'<[^>]*>')
_SPACE     = re.compile(r'\s+')


def clean_str(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _HTML.sub('', text)
    for p, r in PUNCT_MAPPING.items():
        text = text.replace(p, r)
    text = _WHITELIST.sub('', text)
    return _SPACE.sub(' ', text).strip()


def clean_text_medical(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ''
    text = html.unescape(text)
    text = re.sub(r'(https?://|www\.)[A-Za-z0-9\uac00-\ud7a3./?=&%~_#-]+', '', text)
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '', text)
    text = re.sub(r' ?\(\s*\) ?', ' ', text)
    text = re.sub(r'\(\.+\)', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'외부\s*링크[:.]*', '', text)
    text = re.sub(r'Copyright.*', '', text, flags=re.IGNORECASE)
    return clean_str(text)


def is_spam(text: str) -> bool:
    length = len(text)
    if length < 20:
        return True
    boilerplate = ['저작권', '크리에이티브 커먼즈', '라이선스', '분류:', '목차',
                   '파일:', '각주', '편집', '토론', '역사']
    if length < 50 and any(k in text for k in boilerplate):
        return True
    if text.strip().startswith(('*', '>', '-')) and len(text.strip()[1:].strip()) < 5:
        return True
    if length < 300:
        return False
    ns = text.replace(' ', '')
    if not ns:
        return True
    valid = len(re.sub(r'[^\uac00-\ud7a3a-zA-Z]', '', ns))
    if valid / len(ns) < 0.20:
        return True
    words = text.split()
    if words:
        _, cnt = Counter(words).most_common(1)[0]
        if cnt / len(words) > 0.6:
            return True
    if length > 1000 and text.count(' ') / length < 0.01:
        return True
    return False


def sentence_split(text: str):
    sents = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 5]


# 빠른 테스트
sample = '환자는 HTN, DM2 진단받았으며 ACE inhibitor 투여 중. BP 140/90mmHg, HbA1c 8.2% 확인됨.'
print('정제 테스트:', clean_text_medical(sample))


def doc_generator():
    stats = Counter()

    # JSON 파일: content 필드 추출
    for d in JSON_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for fpath in p.glob('*.json'):
            try:
                with open(fpath, encoding='utf-8-sig') as f:
                    data = json.load(f)
                raw = data.get('content', '')
                if not raw:
                    stats['json_skip'] += 1; continue
                cleaned = clean_text_medical(raw)
                if not cleaned or is_spam(cleaned):
                    stats['json_skip'] += 1; continue
                sents = sentence_split(cleaned)
                if sents:
                    stats['json_ok'] += 1
                    yield {'sentences': sents}
                else:
                    stats['json_skip'] += 1
            except Exception:
                stats['json_err'] += 1

    # TXT 파일: 빈 줄 기준 문단 분리 -> 각 문단을 하나의 문서로
    for d in TXT_DIRS:
        p = Path(d)
        if not p.exists():
            continue
        for fpath in p.glob('*.txt'):
            try:
                try:
                    raw = fpath.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    raw = fpath.read_text(encoding='cp949')
                paragraphs = [pr.strip() for pr in re.split(r'\n{2,}', raw) if pr.strip()]
                for para in paragraphs:
                    cleaned = clean_text_medical(para)
                    if not cleaned or is_spam(cleaned):
                        continue
                    sents = sentence_split(cleaned)
                    if sents:
                        stats['txt_ok'] += 1
                        yield {'sentences': sents}
            except Exception:
                stats['txt_err'] += 1

    print(f'  JSON 사용:{stats["json_ok"]:,} 제외:{stats["json_skip"]:,} 에러:{stats["json_err"]:,}')
    print(f'  TXT  사용:{stats["txt_ok"]:,}  에러:{stats["txt_err"]:,}')

print('doc_generator 정의 완료')

if not os.path.exists(INTERMEDIATE_DIR):
    print('[Phase 1] 문서 인덱스 생성 중...')
    doc_dataset = Dataset.from_generator(
        doc_generator,
        features=Features({'sentences': [Value('string')]}),
        writer_batch_size=500
    )
    doc_dataset.save_to_disk(INTERMEDIATE_DIR)
    print(f'[Phase 1] 완료: {len(doc_dataset):,} 문서')
else:
    print(f'[Phase 1] 기존 인덱스 로드: {INTERMEDIATE_DIR}')
    doc_dataset = load_from_disk(INTERMEDIATE_DIR)

print(f'총 문서 수: {len(doc_dataset):,}')


import statistics as stat_mod

# ==========================================
# [추가/수정] 토큰 및 언어 분포 집계 로직
# ==========================================
total_sents, total_chars = 0, 0
sampled_lens = []

# 문서 단위 통계 변수
total_docs = 0
english_only_docs = 0
korean_included_docs = 0

# 토큰(어절) 통계 변수
english_word_tokens = 0
korean_word_tokens = 0

# 한글 포함 여부 판별 정규식
hangul_pattern = re.compile(r'[가-힣]')

for item in tqdm(doc_dataset, ncols=80, desc='언어별 토큰 집계'):
    sents = item['sentences']
    if not sents:
        continue
        
    total_docs += 1
    total_sents += len(sents)
    
    # 문서를 하나의 텍스트로 결합하여 평가
    doc_text = " ".join(sents)
    total_chars += len(doc_text)
    
    if len(sampled_lens) < 20000:
        sampled_lens.extend(len(s) for s in sents)
        
    # 문서의 총 어절 수 계산
    doc_tokens = doc_text.split()
    token_count = len(doc_tokens)
    
    # 한글 포함 여부에 따라 분류
    if not hangul_pattern.search(doc_text):
        english_only_docs += 1
        english_word_tokens += token_count
    else:
        korean_included_docs += 1
        korean_word_tokens += token_count

print('\n' + '='*60)
print(f'[전체 코퍼스 통계]')
print(f'  총 문서 수     : {total_docs:>12,}')
print(f'  총 문장 수     : {total_sents:>12,}')
print(f'  총 글자 수     : {total_chars:>12,}')
if total_sents:
    print(f'  평균 문장 길이 : {total_chars/total_sents:12.1f} 자')
if sampled_lens:
    print(f'  문장 중앙값    : {stat_mod.median(sampled_lens):12.0f} 자 (샘플 {len(sampled_lens):,}개)')

print('-'*60)
print(f'[언어별 어절(Word) 토큰 집계]')
if english_only_docs > 0:
    eng_avg = english_word_tokens / english_only_docs
    print(f'  [100% 영문 문서]')
    print(f'    - 문서 수    : {english_only_docs:>12,} 개')
    print(f'    - 어절 토큰  : {english_word_tokens:>12,} 개')
    print(f'    - 평균 어절  : {eng_avg:12.2f} 개')
else:
    print('  [100% 영문 문서]: 없음')

print()

if korean_included_docs > 0:
    kor_avg = korean_word_tokens / korean_included_docs
    print(f'  [한국어 혼용 문서]')
    print(f'    - 문서 수    : {korean_included_docs:>12,} 개')
    print(f'    - 어절 토큰  : {korean_word_tokens:>12,} 개')
    print(f'    - 평균 어절  : {kor_avg:12.2f} 개')
else:
    print('  [한국어 혼용 문서]: 없음')
print('='*60)