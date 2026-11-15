"""DVC 데이터 창고(구글 드라이브 폴더) 경로를 내 컴퓨터에 설정한다. 처음 한 번만.

  python scripts/set_storage.py                         # 자동으로 찾기
  python scripts/set_storage.py "G:\\내 드라이브\\road-risk-dvc"   # 직접 지정

구글 드라이브 데스크톱 앱이 켜져 있어야 하고, 공유받은 road-risk-dvc 폴더가
'내 드라이브'에 바로가기로 추가되어 있어야 한다 (가이드 참고).
설정은 .dvc/config.local 에 저장되고 깃에는 안 올라간다 (사람마다 경로가 달라서).
※ 'dvc remote modify --local' 명령은 경로에 한글이 있으면 오류가 나서 이 스크립트를 쓴다.
"""
import glob
import locale
import os
import string
import sys
from pathlib import Path

FOLDER = "road-risk-dvc"
REMOTE = "gdrive"
ROOT = Path(__file__).resolve().parents[1]


def candidates():
    pats = []
    if os.name == "nt":
        for d in string.ascii_uppercase:
            pats += [f"{d}:/내 드라이브/{FOLDER}", f"{d}:/My Drive/{FOLDER}"]
    home = str(Path.home())
    pats += [f"{home}/Library/CloudStorage/GoogleDrive-*/*/{FOLDER}",  # macOS
             f"{home}/Google Drive/*/{FOLDER}", f"{home}/Google Drive/{FOLDER}"]
    found = []
    for p in pats:
        found += [g for g in glob.glob(p) if os.path.isdir(g)]
    return found


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        found = candidates()
        if not found:
            sys.exit(f"[오류] '{FOLDER}' 폴더를 못 찾았습니다.\n"
                     "  1) 구글 드라이브 데스크톱 앱이 켜져 있는지\n"
                     "  2) 공유받은 폴더를 '내 드라이브에 바로가기 추가' 했는지 확인한 뒤,\n"
                     "  탐색기에서 그 폴더 경로를 복사해 이렇게 실행하세요:\n"
                     f'  python scripts/set_storage.py "G:\\내 드라이브\\{FOLDER}"')
        path = found[0]
        if len(found) > 1:
            print("여러 개를 찾아서 첫 번째를 씁니다:", *found, sep="\n  ")
    path = os.path.abspath(path).replace("\\", "/")
    if not os.path.isdir(path):
        sys.exit(f"[오류] 폴더가 없습니다: {path}")
    cfg = ROOT / ".dvc" / "config.local"
    # DVC는 설정 파일을 시스템 기본 인코딩으로 읽는다 (한국어 Windows = cp949, PYTHONUTF8=1 이면 utf-8).
    # 그래서 같은 인코딩으로 써야 한글 경로("내 드라이브")가 안 깨진다.
    enc = locale.getpreferredencoding(False)
    cfg.write_text(f"['remote \"{REMOTE}\"']\n    url = \"{path}\"\n", encoding=enc)
    print(f"설정 완료: 데이터 창고 = {path}")
    print("이제 'dvc pull' (받기) / 'dvc push' (올리기) 를 쓸 수 있습니다.")


if __name__ == "__main__":
    main()
