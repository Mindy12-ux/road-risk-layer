"""보호구역 → data/features/zone.csv  (키: grid_id)

  python scripts/features/zone.py 경로/어린이보호구역_서울_통합.csv 경로/노인장애인보호구역_서울_통합.csv

500m 칸마다 보호구역이 몇 개 있는지 센다. 보호구역은 시간에 따라 안 바뀌므로 키는 grid_id 하나.
  - 어린이보호구역_수 : 어린이보호구역 (초등학교·유치원·어린이집 등 앞)
  - 노인보호구역_수   : 노인·장애인보호구역 중 장소유형코드 1
  - 장애인보호구역_수 : 노인·장애인보호구역 중 장소유형코드 2
  - 보호구역CCTV_합계 : 그 칸 보호구역들의 CCTV 설치대수 합 (빈 값은 0으로 봄)

원자료 (공공데이터포털, 2026년 기준. 도봉구 어린이보호구역은 2019년 자료, 도봉구 노인장애인보호구역은 없음)
  - 전국어린이보호구역표준데이터 → 서울 25개 구 파일을 합친 것 (1,598줄)
  - 전국노인장애인보호구역표준데이터 → 서울 24개 구 파일을 합친 것 (214줄)

같은 위치(위도·경도)에 같은 종류가 여러 줄이면 한 곳으로 센다.
  예) "언남초등학교"와 "언남초교병설유치원"은 같은 좌표 = 보호구역 하나

칸 번호(grid_id = cx_cy)는 기준 데이터셋과 같은 방식:
  위도·경도 → EPSG:5179(미터 좌표) → 500으로 나눈 몫
기준 데이터셋(data/base/keys.csv)에 나오는 모든 칸을 넣고, 보호구역이 없는 칸은 0.

필요한 패키지: pip install pyproj
"""
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_DIR, FEAT_DIR, normalize_keys, read_table, write_csv  # noqa: E402

CELL = 500
TO_METER = Transformer.from_crs(4326, 5179, always_xy=True)


def to_grid_id(lat, lon):
    x, y = TO_METER.transform(lon.astype(float).values, lat.astype(float).values)
    return [f"{int(a // CELL)}_{int(b // CELL)}" for a, b in zip(x, y)]


def count_per_cell(df, name):
    """같은 좌표는 한 곳으로 합친 뒤, 칸마다 개수를 센다."""
    one = df.drop_duplicates(["위도", "경도"]).copy()
    one["grid_id"] = to_grid_id(one["위도"], one["경도"])
    print(f"  {name}: {len(df):,}줄 → 같은 위치 합친 뒤 {len(one):,}곳")
    return one.groupby("grid_id").size().rename(name)


def cctv_per_cell(df):
    one = df.drop_duplicates(["위도", "경도"]).copy()
    one["grid_id"] = to_grid_id(one["위도"], one["경도"])
    one["cctv"] = pd.to_numeric(one["CCTV설치대수"], errors="coerce").fillna(0)
    return one.groupby("grid_id")["cctv"].sum()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    child = read_table(sys.argv[1])
    senior = read_table(sys.argv[2])
    senior["장소유형코드"] = senior["장소유형코드"].astype(str).str.lstrip("0")

    counts = pd.concat([
        count_per_cell(child, "어린이보호구역_수"),
        count_per_cell(senior[senior["장소유형코드"] == "1"], "노인보호구역_수"),
        count_per_cell(senior[senior["장소유형코드"] == "2"], "장애인보호구역_수"),
    ], axis=1)
    cctv = pd.concat([cctv_per_cell(child), cctv_per_cell(senior)]).groupby(level=0).sum()
    counts["보호구역CCTV_합계"] = cctv

    # 기준 데이터셋에 나오는 모든 칸을 넣고, 보호구역이 없는 칸은 0
    cells = normalize_keys(read_table(BASE_DIR / "keys.csv"))["grid_id"].drop_duplicates()
    d = pd.DataFrame({"grid_id": cells}).merge(
        counts.rename_axis("grid_id").reset_index(), on="grid_id", how="left")
    cols = ["어린이보호구역_수", "노인보호구역_수", "장애인보호구역_수", "보호구역CCTV_합계"]
    d[cols] = d[cols].fillna(0).astype(int)

    outside = sorted(set(counts.index) - set(cells))
    write_csv(d.sort_values("grid_id"), FEAT_DIR / "zone.csv")
    print(f"data/features/zone.csv  {len(d):,}줄 (칸), 변수 {cols}")
    print("  보호구역이 1개 이상 있는 칸:",
          {c: int((d[c] > 0).sum()) for c in cols[:3]})
    if outside:
        print(f"  ※ 기준 데이터셋에 없는 칸에 있는 보호구역 칸 {len(outside)}개는 뺐습니다 (사고·무사고 줄이 없는 칸)")


if __name__ == "__main__":
    main()
