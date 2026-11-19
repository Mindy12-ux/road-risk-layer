"""무인 교통단속카메라 → data/features/camera.csv  (키: grid_id)

  python scripts/features/camera.py 경로/전국무인교통단속카메라표준데이터.csv

500m 칸마다 속도를 줄이게 만드는 고정식 단속카메라가 몇 곳 있는지 센다. 키는 grid_id 하나.
  - 과속단속카메라_수 : 단속구분 1(속도) + 구간단속 카메라(단속구분 99 중 단속구간위치구분이 적힌 것, 시점·종점 각각)
  - 신호단속카메라_수 : 단속구분 2(신호). 교차로 신호위반 카메라로, 원자료에 제한속도가 함께 적혀 있다
불법주정차 카메라(단속구분 4, 구청 관리)는 속도와 관계없어 세지 않는다.

원자료: 공공데이터포털 전국무인교통단속카메라표준데이터 (https://www.data.go.kr/data/15028200/standard.do)
  - 2026-06-19 수정본, 전국 44,650줄. 이동식 카메라는 원자료에 없음
  - 단속구분 코드는 "4"와 "04"처럼 섞여 있어 앞의 0을 떼고 본다

같은 위치(위도·경도)에 같은 종류가 여러 줄이면 한 곳으로 센다. (차로마다 따로 등록된 카메라 등)

※ 지금 설치된 카메라 기준이다. 설치연도를 반영하지 않으므로, 2020~2025년에 새로 생긴 카메라도
   그 전 날짜의 줄에 똑같이 붙는다 (서울 과속·신호 카메라의 대부분이 2020년 이후 설치).

칸 번호와 서울 범위를 고르는 방식은 point_geo.py 참고. 기준 데이터셋(data/base/keys.csv)의 모든 칸을 넣고 없는 칸은 0.
필요한 패키지: pip install pyproj
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT_DIR, write_csv  # noqa: E402
from point_geo import add_grid_id, base_cells, count_per_cell, fill_cells, read_portal_csv  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = read_portal_csv(sys.argv[1])
    cam, dropped = add_grid_id(raw)
    print(f"  원자료 {len(raw):,}줄 (좌표가 없거나 틀려서 뺀 줄 {dropped:,})")

    # 단속구분: "1", "02", "1+02" 처럼 적혀 있다 → 앞의 0을 뗀 코드 집합으로
    codes = cam["단속구분"].fillna("").str.split(r"[^0-9]+").apply(lambda xs: {x.lstrip("0") for x in xs if x})
    section = codes.apply(lambda s: "99" in s) & cam["단속구간위치구분"].fillna("").str.strip().ne("")
    speed = codes.apply(lambda s: "1" in s) | section
    signal = codes.apply(lambda s: "2" in s)

    cells = base_cells()
    counts = pd.concat([
        count_per_cell(cam[speed], "과속단속카메라_수", cells),
        count_per_cell(cam[signal], "신호단속카메라_수", cells),
    ], axis=1)
    d, cols = fill_cells(counts, cells)

    write_csv(d, FEAT_DIR / "camera.csv")
    print(f"data/features/camera.csv  {len(d):,}줄 (칸), 변수 {cols}")
    print("  카메라가 1곳 이상 있는 칸:", {c: int((d[c] > 0).sum()) for c in cols})


if __name__ == "__main__":
    main()
