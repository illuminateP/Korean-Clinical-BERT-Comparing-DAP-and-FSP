"""
MSD 크롤링 데이터 전처리 → BERT 프리트레인용 말뭉치 생성

입력:  outputs/msd_raw.jsonl  (크롤러 출력)
출력:
  - outputs/corpus_sentences.txt   (문장 단위, 문서 경계 빈 줄)
  - outputs/corpus_stats.json      (통계 정보)
"""

import json
import os
import re
import unicodedata

from tqdm import tqdm

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
MSD_OUT = os.path.join(DATA, "1.corpus", "4.msd", "outputs")


# ─── 설정 ──────────────────────────────────────────────
INPUT_FILE = os.path.join(MSD_OUT, "msd_raw.jsonl")
OUTPUT_CORPUS = os.path.join(MSD_OUT, "corpus_sentences.txt")
OUTPUT_STATS = os.path.join(MSD_OUT, "corpus_stats.json")

# 최소 문장 길이 (글자 수)
MIN_SENTENCE_LEN = 10
# 최대 문장 길이
MAX_SENTENCE_LEN = 1000

# 스킵할 섹션 헤딩 (정확 일치)
SKIP_SECTION_HEADINGS = {
    "추가 정보",
    "알고 계십니까?",
}

# 마지막 섹션이 이 헤딩이면 통째로 스킵 (보통 저작권/링크만 들어있음)
SKIP_LAST_SECTION_HEADINGS = {
    "추가 정보",
}


# ─── 텍스트 정제 함수들 ────────────────────────────────
def normalize_unicode(text: str) -> str:
    """유니코드 정규화 (NFC)."""
    return unicodedata.normalize("NFC", text)


def remove_html_artifacts(text: str) -> str:
    """HTML 잔여물 제거."""
    text = re.sub(r"<[^>]+>", "", text)  # 태그
    text = re.sub(r"&[a-zA-Z]+;", " ", text)  # HTML 엔티티 (&nbsp; 등)
    text = re.sub(r"&#\d+;", " ", text)  # 숫자 엔티티
    return text


def remove_consecutive_duplicate_lines(text: str) -> str:
    """연속 중복 라인 제거 (크롤러 중첩 태그 파싱으로 인한 21% 노이즈)."""
    lines = text.split("\n")
    deduped = []
    prev = None
    for line in lines:
        stripped = line.strip()
        if stripped == prev and len(stripped) > 5:
            continue
        deduped.append(line)
        prev = stripped
    return "\n".join(deduped)


def clean_whitespace(text: str) -> str:
    """공백 정리."""
    text = text.replace("\t", " ").replace("\r", "")
    text = re.sub(r" +", " ", text)  # 연속 공백 → 단일 공백
    text = re.sub(r"\n{3,}", "\n\n", text)  # 과도한 줄바꿈
    return text.strip()


# 줄 단위로 제거할 패턴 (컴파일)
_LINE_REMOVE_PATTERNS = [
    # Copyright (여러 변형)
    re.compile(r"Copyright\s*©.*", re.IGNORECASE),
    # 주의 경고문
    re.compile(r"^주의 경고문$"),
    # 이미지/사진/자료 크레딧
    re.compile(r"^(위\s+|아래\s+)?(이미지|사진|자료)\s*제공\s*[:：]"),
    re.compile(r"SCIENCE PHOTO LIBRARY", re.IGNORECASE),
    re.compile(r"^DR\s+[A-Z].*?/SCIENCE PHOTO", re.IGNORECASE),
    re.compile(r"^[A-Z]+/SCIENCE PHOTO", re.IGNORECASE),
    re.compile(r"^©\s*Springer", re.IGNORECASE),
    # Test your Knowledge / Take a Quiz
    re.compile(r"^Test your Knowledge$", re.IGNORECASE),
    re.compile(r"^Take a Quiz!?$", re.IGNORECASE),
    # 탭 네비게이션 (원인 | 증상 | 진단 | 치료 등)
    re.compile(r"^[가-힣\s]+(\|[가-힣\s]*){2,}$"),
    # 파이프로 끝나는 네비게이션 텍스트
    re.compile(r"^.+\|$"),
    # 작성자/검토자 정보
    re.compile(r"^작성자\s*[:：\n]"),
    re.compile(r"^검토자\s*[:：\n]"),
    re.compile(r"^검토/개정일"),
    # 버전 ID
    re.compile(r"^v\d{5,}_ko$"),
    # 실험실 검사 / 3D 모델 위젯 텍스트 (단독 줄)
    re.compile(r"^실험실 검사$"),
    re.compile(r"^3D 모델$"),
    # "요약 정보" 단독 줄
    re.compile(r"^요약 정보$"),
]


def remove_boilerplate(text: str) -> str:
    """MSD 사이트 공통 상용구 제거."""
    # 멀티라인 패턴 (텍스트 블록 단위)
    block_patterns = [
        r"MSD 매뉴얼.*?에서 확인하십시오\.?",
        r"의학적 의견이 필요한 경우.*?연락하십시오\.?",
        r"이 자료는.*?제공됩니다\.?",
        r"맨 위",
        r"주제 리소스",
        r"더 읽기",
        r"이 주제에서",
        r"다음도 참고하십시오\.?",
        r"여기를 클릭하십시오\.?",
        r"비디오를 보십시오\.?",
        r"표를 참조하십시오\.?",
        r"다음의 영어로 된 자료가 유용할 수 있습니다.*",
        r"본 매뉴얼은 이 참고 자료의 내용에 대한 책임을.*",
    ]
    for pat in block_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 줄 단위 필터링
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            filtered.append(line)
            continue
        skip = False
        for pat in _LINE_REMOVE_PATTERNS:
            if pat.search(stripped):
                skip = True
                break
        if not skip:
            filtered.append(line)
    return "\n".join(filtered)


def remove_references_and_urls(text: str) -> str:
    """URL, 참고문헌 번호 제거."""
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\d+\]", "", text)  # [1], [2] 등
    return text


def remove_cross_references(text: str) -> str:
    """(또한XXX참조) 같은 내부 링크 텍스트 정리."""
    # "(또한간염에 대한 개요,만성 B형 간염참조.)" → 제거
    text = re.sub(r"\(또한[^)]*참조\.?\)", "", text)
    return text


def split_sentences(text: str) -> list[str]:
    """한국어 문장 분리. 마침표/물음표/느낌표 기준."""
    # 약어 보호: 예) 1.5, 3.0 등
    text = re.sub(r"(\d)\.\s*(\d)", r"\1<DOT>\2", text)

    # 문장 분리
    sentences = re.split(r"(?<=[.?!。])\s+", text)

    # 약어 복원
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    result = []
    for sent in sentences:
        sent = sent.strip()
        if MIN_SENTENCE_LEN <= len(sent) <= MAX_SENTENCE_LEN:
            result.append(sent)
    return result


def should_skip_section(heading: str, text: str, is_last: bool) -> bool:
    """의미 없는 섹션인지 판단."""
    h = heading.strip()

    # 스킵 대상 헤딩
    if h in SKIP_SECTION_HEADINGS:
        return True

    # 마지막 섹션 중 스킵 대상
    if is_last and h in SKIP_LAST_SECTION_HEADINGS:
        return True

    # Copyright만 있는 섹션
    stripped = text.strip()
    if not stripped:
        return True
    if re.match(r"^Copyright.*All rights reserved\.?$", stripped, re.IGNORECASE):
        return True

    return False


def clean_document(doc: dict) -> list[str]:
    """단일 문서를 정제하여 문장 리스트로 반환."""
    # 섹션별 텍스트 우선 사용, 없으면 full_text
    texts = []
    sections = doc.get("sections", [])
    if sections:
        for i, sec in enumerate(sections):
            heading = sec.get("heading", "")
            body = sec.get("text", "")
            is_last = i == len(sections) - 1

            if should_skip_section(heading, body, is_last):
                continue

            if heading:
                texts.append(heading)
            if body:
                texts.append(body)
    else:
        full_text = doc.get("full_text", "")
        if full_text:
            texts.append(full_text)

    raw_text = "\n".join(texts)

    # 순차적 정제 파이프라인
    text = normalize_unicode(raw_text)
    text = remove_html_artifacts(text)
    text = remove_consecutive_duplicate_lines(text)
    text = remove_boilerplate(text)
    text = remove_cross_references(text)
    text = remove_references_and_urls(text)
    text = clean_whitespace(text)

    # 문장 분리
    sentences = split_sentences(text)
    return sentences


# ─── 메인 처리 ─────────────────────────────────────────
def preprocess():
    if not os.path.exists(INPUT_FILE):
        print(f"입력 파일이 없습니다: {INPUT_FILE}")
        print("먼저 1.crawl.py 를 실행하세요.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CORPUS), exist_ok=True)

    # 문서 수 세기
    total_docs = sum(1 for _ in open(INPUT_FILE, encoding="utf-8"))
    print(f"[전처리] 입력 문서: {total_docs:,}개")

    stats = {
        "total_documents": total_docs,
        "valid_documents": 0,
        "total_sentences": 0,
        "total_characters": 0,
        "categories": {},
    }

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as fin,
        open(OUTPUT_CORPUS, "w", encoding="utf-8") as fout,
    ):
        for line in tqdm(fin, total=total_docs, desc="전처리"):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            sentences = clean_document(doc)
            if not sentences:
                continue

            # 문서 경계: 빈 줄로 구분 (BERT NSP/문서 경계 학습에 활용)
            for sent in sentences:
                fout.write(sent + "\n")
            fout.write("\n")  # 문서 경계

            # 통계 갱신
            cat = doc.get("category", "기타")
            stats["valid_documents"] += 1
            stats["total_sentences"] += len(sentences)
            stats["total_characters"] += sum(len(s) for s in sentences)
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1

    # 통계 저장
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 전처리 결과:")
    print(f"  유효 문서: {stats['valid_documents']:,}개")
    print(f"  총 문장:   {stats['total_sentences']:,}개")
    print(f"  총 글자:   {stats['total_characters']:,}자")
    print(f"  카테고리:  {len(stats['categories'])}개")
    print(f"  말뭉치:    {OUTPUT_CORPUS}")
    print(f"  통계:      {OUTPUT_STATS}")


if __name__ == "__main__":
    preprocess()
