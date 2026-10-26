"""
국립국어원 신문 말뭉치에서 의학/건강/생명과학 관련 기사를 추출하여
BERT 프리트레인용 말뭉치로 가공합니다.

입력:  ./N*RW*.json  (국립국어원 신문 말뭉치 원시 JSON)
출력:
  - outputs/news_medical_corpus.txt   (문장 단위, 문서 경계 빈 줄)
  - outputs/news_medical_stats.json   (통계 정보)

필터링 전략:
  - 미용/건강 토픽: 의학/건강 키워드가 포함된 기사만 추출
  - IT/과학 토픽: 생명과학/의학 키워드가 포함된 기사만 추출
"""

import glob
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
NEWS_DIR = os.path.join(DATA, "1.corpus", "5.news")


# ─── 설정 ──────────────────────────────────────────────
INPUT_DIR = os.path.join(NEWS_DIR, "origin")
OUTPUT_DIR = os.path.join(NEWS_DIR, "outputs")
OUTPUT_CORPUS = os.path.join(OUTPUT_DIR, "news_medical_corpus.txt")
OUTPUT_STATS = os.path.join(OUTPUT_DIR, "news_medical_stats.json")

MIN_SENTENCE_LEN = 10

# ─── 키워드 ────────────────────────────────────────────
# 의학/건강/생명과학 관련 키워드
MEDICAL_KEYWORDS = {
    # 의료 일반
    "병원", "환자", "의사", "의료", "의학", "진료", "처방", "투약",
    # 질환/증상
    "질환", "질병", "증상", "증후군", "합병증", "감염", "염증", "통증",
    # 치료/시술
    "치료", "진단", "수술", "시술", "검사", "내시경", "방사선", "항암",
    # 약물
    "약물", "항생제", "항바이러스", "진통제", "스테로이드", "백신", "신약",
    # 주요 질병
    "암", "당뇨", "고혈압", "심장", "뇌졸중", "치매", "간염", "폐렴",
    "결핵", "에이즈", "코로나", "메르스", "사스", "독감", "인플루엔자",
    # 생명과학
    "세포", "유전자", "DNA", "RNA", "단백질", "줄기세포", "면역",
    "항체", "게놈", "유전체", "바이오", "생명공학", "생명과학",
    # 장기/신체
    "간", "신장", "폐", "뇌", "혈관", "혈액", "골수", "장기이식",
    # 기타 의학
    "임상", "제약", "뇌과학", "역학", "전염병", "팬데믹",
    "건강보험", "응급", "중환자", "마취", "재활",
}

# 미용 전용 키워드 (미용/건강에서 의학과 무관한 기사 배제용)
BEAUTY_ONLY_KEYWORDS = {
    "화장품", "메이크업", "스킨케어", "에스테틱", "미백", "네일",
    "헤어스타일", "퍼머", "염색", "뷰티팁",
}

# 대상 토픽
TARGET_TOPICS = {"미용/건강", "IT/과학"}


# ─── 텍스트 정제 ───────────────────────────────────────
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_whitespace(text: str) -> str:
    text = text.replace("\t", " ").replace("\r", "")
    text = re.sub(r" +", " ", text)
    return text.strip()


# 신문 기사 상용구 제거 패턴
_BOILERPLATE_PATTERNS = [
    re.compile(r"^[\[【\(].{0,30}(기자|특파원|통신원)[\]】\)]"),  # [홍길동 기자]
    re.compile(r"^[\[【\(].{0,20}(뉴스|신문|일보|타임즈|저널)[\]】\)]"),  # [OO뉴스]
    re.compile(r"[가-힣]{2,4}\s*(기자|특파원)\s*[=@]"),  # 홍길동 기자 =
    re.compile(r"(무단\s*(전재|복제|배포)|저작권)", re.IGNORECASE),
    re.compile(r"^사진\s*[=:]"),
    re.compile(r"^▶|^◇|^■|^●|^★|^☞|^※"),  # 기호로 시작하는 안내문구
    re.compile(r"(구독|뉴스레터|앱\s*다운로드).*((누르|클릭|터치)|(하세요|합니다))"),
    re.compile(r"^(관련\s*기사|관련\s*뉴스)"),
    # 방송사 채널 안내 노이즈
    re.compile(r"(중심언론|CBS\s*뉴스FM|음악FM|TV\s*CH\s*\d+)"),
]


def remove_boilerplate_line(line: str) -> bool:
    """해당 줄이 상용구인지 판단. True이면 제거 대상."""
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(line):
            return True
    return False


def split_sentences(text: str) -> list[str]:
    """한국어 문장 분리."""
    text = re.sub(r"(\d)\.\s*(\d)", r"\1<DOT>\2", text)
    sentences = re.split(r"(?<=[.?!。])\s+", text)
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    result = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) >= MIN_SENTENCE_LEN:
            result.append(sent)
    return result


# ─── 기사 필터링 ──────────────────────────────────────
def has_medical_keywords(text: str) -> bool:
    """의학/건강/생명과학 키워드가 포함되어 있는지 확인."""
    return any(kw in text for kw in MEDICAL_KEYWORDS)


def is_beauty_only(text: str) -> bool:
    """미용 전용 기사인지 확인 (의학 키워드 없이 미용 키워드만 있는 경우)."""
    has_beauty = any(kw in text for kw in BEAUTY_ONLY_KEYWORDS)
    has_medical = has_medical_keywords(text)
    return has_beauty and not has_medical


def should_include(doc: dict) -> bool:
    """기사를 포함할지 결정."""
    m = doc.get("metadata", {})
    topic = m.get("topic", "")

    if topic not in TARGET_TOPICS:
        return False

    # 본문 텍스트 결합
    text = " ".join(p.get("form", "") for p in doc.get("paragraph", []))

    if topic == "미용/건강":
        # 의학 키워드 포함 & 미용 전용이 아닌 기사
        return has_medical_keywords(text) and not is_beauty_only(text)

    if topic == "IT/과학":
        return has_medical_keywords(text)

    return False


# ─── 기사 텍스트 추출 및 정제 ─────────────────────────
def extract_clean_sentences(doc: dict) -> list[str]:
    """기사에서 정제된 문장 리스트 추출."""
    paragraphs = doc.get("paragraph", [])
    lines = []
    for p in paragraphs:
        form = p.get("form", "").strip()
        if not form:
            continue
        form = normalize_unicode(form)
        form = clean_whitespace(form)
        if not form or len(form) < 5:
            continue
        if remove_boilerplate_line(form):
            continue
        lines.append(form)

    text = "\n".join(lines)
    return split_sentences(text)


# ─── 메인 처리 ─────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 입력 파일 목록
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.json")))
    print(f"[추출] 입력 파일: {len(files)}개")

    # 입력이 하나도 없으면 빈 산출물을 만들지 않는다. 빈 파일이 생기면 다음 단계의
    # 6.merge.py 가 os.path.exists 만 보고 통과시켜, 신문 말뭉치(전체의 약 81%)가
    # 빠진 채로 병합이 '성공'해 버린다.
    if not files:
        print(f"[중단] 입력 디렉토리에 JSON 파일이 없습니다: {INPUT_DIR}")
        print("       국립국어원 모두의 말뭉치(신문)를 내려받아 위 경로에 배치하십시오.")
        return

    stats = {
        "input_files": len(files),
        "total_documents_scanned": 0,
        "extracted_documents": 0,
        "extracted_sentences": 0,
        "extracted_characters": 0,
        "by_topic": {},
        "by_publisher": {},
    }

    with open(OUTPUT_CORPUS, "w", encoding="utf-8") as fout:
        for fname in tqdm(files, desc="파일 처리"):
            try:
                with open(fname, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"  오류 스킵: {fname}: {e}")
                continue

            for doc in data.get("document", []):
                stats["total_documents_scanned"] += 1

                if not should_include(doc):
                    continue

                sentences = extract_clean_sentences(doc)
                if not sentences:
                    continue

                # 문서 경계: 빈 줄로 구분
                for sent in sentences:
                    fout.write(sent + "\n")
                fout.write("\n")

                # 통계
                m = doc.get("metadata", {})
                topic = m.get("topic", "기타")
                publisher = m.get("publisher", "기타")

                stats["extracted_documents"] += 1
                stats["extracted_sentences"] += len(sentences)
                stats["extracted_characters"] += sum(len(s) for s in sentences)
                stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1
                stats["by_publisher"][publisher] = (
                    stats["by_publisher"].get(publisher, 0) + 1
                )

    # 통계 저장 (publisher는 상위 30개만)
    top_publishers = dict(
        sorted(stats["by_publisher"].items(), key=lambda x: -x[1])[:30]
    )
    stats["by_publisher"] = top_publishers

    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n[완료] 추출 결과:")
    print(f"  스캔 문서:  {stats['total_documents_scanned']:,}개")
    print(f"  추출 문서:  {stats['extracted_documents']:,}개")
    print(f"  총 문장:    {stats['extracted_sentences']:,}개")
    print(f"  총 글자:    {stats['extracted_characters']:,}자")
    print(f"  토픽별: {stats['by_topic']}")
    print(f"  말뭉치:     {OUTPUT_CORPUS}")
    print(f"  통계:       {OUTPUT_STATS}")


if __name__ == "__main__":
    main()
