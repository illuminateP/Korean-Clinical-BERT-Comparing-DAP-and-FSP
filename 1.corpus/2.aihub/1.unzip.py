import os
import zipfile
from pathlib import Path

# ==========================================
# [경로] 모든 단계는 저장소 루트의 data/ 아래를 공유한다.
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
AIHUB_DIR = os.path.join(DATA, "1.corpus", "2.aihub")


def count_total_files(base_path):
    count = 0
    for root, dirs, files in os.walk(base_path):
        count += len(files)
    return count

def extract_and_delete_zips(base_path):
    base_dir = Path(base_path)
    # 하위 디렉토리를 포함하여 모든 .zip 파일 검색
    zip_files = list(base_dir.rglob("*.zip"))
    
    for zip_path in zip_files:
        # 파일명에서 확장자(.zip)를 제외한 이름을 가져와 새 폴더 경로로 설정
        extract_dir = zip_path.parent / zip_path.stem
        # 해당 경로에 폴더가 없다면 생성 (이미 있으면 무시)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for zip_info in zip_ref.infolist():
                    # Windows 환경에서 압축된 한글 파일명이 Linux에서 깨지는 현상 방지
                    try:
                        zip_info.filename = zip_info.filename.encode('cp437').decode('cp949')
                    except (UnicodeEncodeError, UnicodeDecodeError):
                        pass
                    
                    # 생성한 새 폴더 경로(extract_dir) 안에 파일 압축 해제
                    zip_ref.extract(zip_info, path=extract_dir)
            
            # 모든 파일 해제 성공 시 원본 zip 파일 삭제
            os.remove(zip_path)
            print(f"해제 및 삭제 완료: {zip_path.name} -> {extract_dir.name}/ 폴더로 이동됨")
        
        except zipfile.BadZipFile:
            print(f"손상된 압축 파일 에러: {zip_path.name}")

if __name__ == "__main__":
    target_directory = os.path.join(AIHUB_DIR, "origin")
    
    print("작업 시작 전 파일 개수를 계산합니다...")
    before_count = count_total_files(target_directory)
    print(f"작업 전 총 파일 개수: {before_count}개\n")
    
    print("압축 해제 및 원본 삭제 작업을 시작합니다...")
    extract_and_delete_zips(target_directory)
    
    print("\n작업 완료 후 파일 개수를 계산합니다...")
    after_count = count_total_files(target_directory)
    print(f"작업 후 총 파일 개수: {after_count}개")
    
    diff = after_count - before_count
    print(f"결과: {diff}개의 파일이 변동되었습니다.")