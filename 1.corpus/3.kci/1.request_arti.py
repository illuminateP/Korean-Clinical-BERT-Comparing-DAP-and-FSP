import os
import requests
import xml.etree.ElementTree as ET
import json
import time
import random
import gc

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
KCI_OUT = os.path.join(DATA, "1.corpus", "3.kci", "outputs")


def download_kci_to_jsonl_arti_resume(
    output_file=os.path.join(KCI_OUT, "kci_records_arti.jsonl"),
    checkpoint_file=os.path.join(KCI_OUT, "checkpoint_arti.txt")
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    base_url = "https://open.kci.go.kr/oai/request"
    namespaces = {
        'oai': 'http://www.openarchives.org/OAI/2.0/',
        'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/',
        'dc': 'http://purl.org/dc/elements/1.1/'
    }

    # 수집 대상 세트를 ARTI(학술지 논문)로 설정
    target_set = "ARTI"
    
    resume_token = None
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as cp:
            resume_token = cp.read().strip()
            print(f"체크포인트 발견: {resume_token} 지점에서 ARTI 수집을 재개합니다.")

    if resume_token:
        current_url = f"{base_url}?verb=ListRecords&resumptionToken={resume_token}"
    else:
        # 체크포인트가 없는데 산출물이 남아 있으면 이전 수집이 이미 완주한 것이다.
        # 아래 open(..., 'a') 가 기존 파일 뒤에 그대로 덧붙이므로 그냥 두면 전체가
        # 두 배로 중복된다. 조용히 망가뜨리는 대신 중단하고 사용자에게 알린다.
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"[중단] 이전 수집 결과가 이미 존재합니다: {output_file}")
            print("       체크포인트가 없으므로 이번 실행은 처음부터 다시 수집합니다.")
            print("       그대로 진행하면 기존 내용 뒤에 중복 누적되므로 중단합니다.")
            print(f"       다시 수집하려면 먼저 해당 파일을 삭제하십시오:")
            print(f"         rm {output_file}")
            return
        current_url = f"{base_url}?verb=ListRecords&metadataPrefix=oai_dc&set={target_set}"
    
    print(f"========== [{target_set}] 세트 수집 시작/재개 ==========")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        page_count = 1
        while True:
            print(f"[{target_set} - {page_count}페이지] 요청 중: {current_url}")
            
            max_retries = 5
            retry_count = 0
            request_success = False
            
            while retry_count < max_retries:
                try:
                    response = requests.get(current_url, timeout=30)
                    response.raise_for_status()
                    request_success = True
                    break
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    wait_time = 5 * retry_count
                    print(f"네트워크 오류. {wait_time}초 후 재시도 ({retry_count}/{max_retries}): {e}")
                    time.sleep(wait_time)
            
            if not request_success:
                print("최대 재시도 횟수 초과.")
                break

            root = ET.fromstring(response.content)
            records = root.findall('.//oai:record', namespaces)
            
            valid_record_count = 0
            for record in records:
                record_data = {"article_id": None, "title": None, "categories": None, "abstract": None, "language": None}
                dc_node = record.find('.//oai_dc:dc', namespaces)
                
                if dc_node is not None:
                    id_elem = dc_node.find('dc:identifier[@type="artiId"]', namespaces)
                    if id_elem is not None: record_data["article_id"] = id_elem.text
                    cat_elem = dc_node.find('dc:subject', namespaces)
                    if cat_elem is not None: record_data["categories"] = cat_elem.text
                    lang_elem = dc_node.find('dc:language', namespaces)
                    if lang_elem is not None: record_data["language"] = lang_elem.text
                    title_elem = dc_node.find('dc:title[@lang="original"]', namespaces)
                    if title_elem is not None: record_data["title"] = title_elem.text
                    abs_elem = dc_node.find('dc:description[@lang="original"]', namespaces)
                    if abs_elem is not None: record_data["abstract"] = abs_elem.text

                    if record_data["article_id"] and record_data["abstract"]:
                        f.write(json.dumps(record_data, ensure_ascii=False) + '\n')
                        valid_record_count += 1

            print(f" -> {valid_record_count}개 저장 완료")
            f.flush()

            token_element = root.find('.//oai:resumptionToken', namespaces)
            if token_element is not None and token_element.text:
                new_token = token_element.text
                with open(checkpoint_file, 'w', encoding='utf-8') as cp:
                    cp.write(new_token)
                
                current_url = f"{base_url}?verb=ListRecords&resumptionToken={new_token}"
                page_count += 1
                
                root.clear()
                del root
                del response
                gc.collect()
                time.sleep(random.uniform(0.1, 1.0))
            else:
                print(f"[{target_set}] 모든 레코드 다운로드 완료.")
                # 재개 북마크를 지운다. 남겨두면 다음 실행이 '소진된 토큰'으로
                # 재개를 시도해 마지막 페이지를 중복 수집하거나 빈 응답만 받는다.
                # (수집한 JSONL 은 건드리지 않는다)
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print(f"  체크포인트 삭제: {checkpoint_file}")
                break

if __name__ == "__main__":
    download_kci_to_jsonl_arti_resume()