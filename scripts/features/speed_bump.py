"""과속방지턱 → data/features/speed_bump.csv  (키: grid_id)

  python scripts/features/speed_bump.py 경로/전국과속방지턱표준데이터.csv

500m 칸마다 과속방지턱이 몇 곳 있는지 센다. 키는 grid_id 하나.
  - 과속방지턱_수 : 칸 안의 과속방지턱 수

원자료: 공공데이터포털 전국과속방지턱표준데이터 (https://www.data.go.kr/data/15028195/standard.do), 전국 19,411줄

※ 원자료가 불완전하다. 붙이기 전에 꼭 알아둘 것:
  - 서울 자료의 데이터기준일자가 모두 2019~2020년이다
  - 구마다 등록된 줄 수 차이가 아주 크다 (강동·관악·구로·금천·송파·용산·은평·종로는 500줄 안팎,
    강남·중구·동대문 등은 20줄도 안 됨). 실제 개수 차이가 아니라 구청이 올린 자료 양의 차이로 보인다
  - 노원구는 1,382줄 중 1,321줄의 좌표가 기준 칸 밖(서울 바깥)을 가리켜 빠진다
  따라서 0은 "과속방지턱이 없다"가 아니라 "자료에 없다"일 수 있다.

같은 위치(위도·경도)에 여러 줄이면 한 곳으로 센다.
칸 번호와 서울 범위를 고르는 방식은 point_geo.py 참고. 기준 데이터셋(data/base/keys.csv)의 모든 칸을 넣고 없는 칸은 0.
필요한 패키지: pip install pyproj
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEAT_DIR, write_csv  # noqa: E402
from point_geo import add_grid_id, base_cells, count_per_cell, fill_cells, read_portal_csv  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = read_portal_csv(sys.argv[1])
    bump, dropped = add_grid_id(raw)
    print(f"  원자료 {len(raw):,}줄 (좌표가 없거나 틀려서 뺀 줄 {dropped:,})")

    cells = base_cells()
    d, cols = fill_cells(count_per_cell(bump, "과속방지턱_수", cells).to_frame(), cells)

    write_csv(d, FEAT_DIR / "speed_bump.csv")
    print(f"data/features/speed_bump.csv  {len(d):,}줄 (칸), 변수 {cols}")
    print(f"  과속방지턱이 1곳 이상 있는 칸: {int((d[cols[0]] > 0).sum()):,}개")


if __name__ == "__main__":
    main()
