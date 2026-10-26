"""
MSD 매뉴얼 한국어판 크롤러
https://www.msdmanuals.com/ko/home 의 의학 문서를 수집합니다.

수집 흐름:
  1) sitemap(home-topic.xml.gz)에서 전체 토픽 URL 목록 추출
  2) 각 URL을 방문하여 본문 텍스트 추출
  3) JSONL 형태로 저장
"""

import gzip
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote

import requests

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
MSD_OUT = os.path.join(DATA, "1.corpus", "4.msd", "outputs")

from bs4 import BeautifulSoup
from tqdm import tqdm

# ─── 설정 ──────────────────────────────────────────────
BASE_URL = "https://www.msdmanuals.com"
SITEMAP_INDEX_URL = f"{BASE_URL}/ko/sitemaps/home-topic.xml.gz"
OUTPUT_DIR = MSD_OUT
URL_LIST_FILE = os.path.join(OUTPUT_DIR, "msd_topic_urls.json")
RAW_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "msd_raw.jsonl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
REQUEST_DELAY = 1.0   # 워커당 딜레이 (초)
MAX_WORKERS = 8       # 동시 요청 수
MAX_RETRIES = 3
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
_rate_lock = threading.Lock()
_last_request_time = 0.0


# ─── 1단계: sitemap에서 URL 목록 추출 ──────────────────
def fetch_topic_urls_from_sitemap() -> list[str]:
    """sitemap(gzip)에서 /ko/home/ 토픽 URL 목록을 추출합니다."""
    print("[1/3] sitemap에서 토픽 URL 목록 추출 중...")
    resp = SESSION.get(SITEMAP_INDEX_URL, timeout=60)
    resp.raise_for_status()

    # gzip 해제 (서버가 이미 디코딩해서 보내는 경우 대비)
    raw = resp.content
    try:
        xml_text = gzip.decompress(raw).decode("utf-8-sig")
    except gzip.BadGzipFile:
        xml_text = raw.decode("utf-8-sig")

    root = ET.fromstring(xml_text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    urls = []
    for loc in root.findall(".//sm:loc", ns):
        url = loc.text.strip()
        # /ko/home/ 하위 토픽만 수집 (monograph 등 제외)
        if "/ko/home/" in url and "/monograph/" not in url:
            urls.append(url)

    print(f"  → 총 {len(urls):,}개 토픽 URL 발견")
    return urls


# ─── 2단계: 개별 페이지 본문 추출 ──────────────────────
def extract_text_from_page(html: str, url: str) -> dict | None:
    """HTML에서 의학 문서 본문을 추출합니다."""
    soup = BeautifulSoup(html, "html.parser")

    # 제목 추출
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else None

    # 본문 영역 탐색 (여러 선택자 시도)
    content_area = (
        soup.find("div", class_="Topic")
        or soup.find("div", class_="TopicPage")
        or soup.find("article")
        or soup.find("main")
    )

    if not content_area:
        return None

    # 불필요 요소 제거
    for tag in content_area.find_all(
        ["script", "style", "nav", "footer", "header", "aside", "iframe",
         "button", "input", "form", "svg", "figure", "img"]
    ):
        tag.decompose()

    # 클래스 기반 불필요 요소 제거
    for cls in ["quiz", "QuizWidget", "MarketingBlock", "RelatedTopics",
                 "DrugInfo", "CalculatorWidget", "CopyrightNotice"]:
        for tag in content_area.find_all(class_=re.compile(cls, re.I)):
            tag.decompose()

    # 섹션별 추출
    sections = []
    current_heading = None
    current_paragraphs = []

    for elem in content_area.descendants:
        if elem.name in ("h1", "h2", "h3", "h4"):
            # 이전 섹션 저장
            if current_paragraphs:
                text = "\n".join(current_paragraphs)
                if text.strip():
                    sections.append({
                        "heading": current_heading,
                        "text": text.strip()
                    })
            current_heading = elem.get_text(strip=True)
            current_paragraphs = []
        elif elem.name in ("p", "li"):
            text = elem.get_text(strip=True)
            if text and len(text) > 5:
                current_paragraphs.append(text)

    # 마지막 섹션
    if current_paragraphs:
        text = "\n".join(current_paragraphs)
        if text.strip():
            sections.append({
                "heading": current_heading,
                "text": text.strip()
            })

    # 전체 본문 (fallback)
    full_text = content_area.get_text(separator="\n", strip=True)
    # 너무 짧은 페이지는 건너뜀
    if len(full_text) < 100:
        return None

    # URL에서 카테고리 추출
    path_parts = unquote(url).replace(BASE_URL + "/ko/home/", "").split("/")
    category = path_parts[0] if path_parts else None

    return {
        "url": url,
        "title": title,
        "category": category,
        "sections": sections,
        "full_text": full_text,
    }


def _rate_limit():
    """전역 rate limiter — 워커들이 최소 REQUEST_DELAY 간격을 유지."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = REQUEST_DELAY - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def crawl_page(url: str) -> dict | None:
    """단일 페이지를 크롤링합니다. 재시도 로직 포함."""
    for attempt in range(MAX_RETRIES):
        try:
            _rate_limit()
            resp = SESSION.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return extract_text_from_page(resp.text, url)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * 5
                print(f"  ⚠ 재시도 {attempt+1}/{MAX_RETRIES} ({url}): {e}")
                time.sleep(wait)
            else:
                print(f"  ✗ 실패 ({url}): {e}")
                return None
    return None


# ─── 3단계: 메인 크롤링 루프 ──────────────────────────
def crawl_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # URL 목록 캐싱 (재실행 시 sitemap 재요청 방지)
    if os.path.exists(URL_LIST_FILE):
        print(f"[캐시] {URL_LIST_FILE} 에서 URL 목록 로드")
        with open(URL_LIST_FILE, "r", encoding="utf-8") as f:
            urls = json.load(f)
    else:
        urls = fetch_topic_urls_from_sitemap()
        with open(URL_LIST_FILE, "w", encoding="utf-8") as f:
            json.dump(urls, f, ensure_ascii=False, indent=2)
        print(f"  → URL 목록 저장: {URL_LIST_FILE}")

    # 이미 수집한 URL 추적 (이어하기 지원)
    crawled_urls = set()
    if os.path.exists(RAW_OUTPUT_FILE):
        with open(RAW_OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    crawled_urls.add(data.get("url"))
                except json.JSONDecodeError:
                    continue
        print(f"[이어하기] 이미 수집된 문서: {len(crawled_urls):,}개")

    remaining = [u for u in urls if u not in crawled_urls]
    print(f"[2/3] 총 {len(remaining):,}개 페이지 크롤링 시작 (워커 {MAX_WORKERS}개)...")

    success = 0
    write_lock = threading.Lock()
    with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(crawl_page, url): url for url in remaining}
            for future in tqdm(as_completed(futures), total=len(futures), desc="크롤링"):
                result = future.result()
                if result:
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        success += 1

    total = len(crawled_urls) + success
    print(f"\n[완료] 총 {total:,}개 문서 수집 (이번 실행: {success:,}개)")
    print(f"  → 저장 경로: {RAW_OUTPUT_FILE}")


if __name__ == "__main__":
    crawl_all()
