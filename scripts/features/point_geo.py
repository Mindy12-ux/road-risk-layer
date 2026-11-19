"""camera.py, speed_bump.py 가 같이 쓰는 함수: 위도·경도가 있는 시설 목록 → 칸마다 개수.

칸 번호(grid_id = cx_cy)는 기준 데이터셋과 같은 방식: 위도·경도 → EPSG:5179(미터 좌표) → 500으로 나눈 몫.
서울인지 아닌지는 시도명 컬럼이 아니라 좌표로 가린다 (기준 데이터셋에 나오는 칸 안에 있으면 센다).
  - 공공데이터포털 표준데이터에는 시도명 칸이 밀리거나 비어 있는 줄이 있어서 시도명으로 고르면 빠지는 줄이 생긴다.
  - 서울 경계에 걸친 칸은 경기도 쪽 시설도 같이 센다 (그 칸 안에 실제로 있으므로).
"""
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_DIR, normalize_keys, read_table  # noqa: E402

CELL = 500
LL_TO_5179 = Transformer.from_crs(4326, 5179, always_xy=True)


def read_portal_csv(path):
    """공공데이터포털 CSV 읽기. 포털에서 바로 받으면 EUC-KR(cp949), 다시 저장한 파일은 utf-8-sig."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str)
        except UnicodeDecodeError:
            pass
    sys.exit(f"[오류] {path}: utf-8, cp949 어느 쪽으로도 읽을 수 없습니다.")


def base_cells():
    """기준 데이터셋에 나오는 모든 칸."""
    return normalize_keys(read_table(BASE_DIR / "keys.csv"))["grid_id"].drop_duplicates()


def add_grid_id(df):
    """위도·경도로 grid_id 를 붙인다. 좌표가 비었거나 우리나라 범위를 벗어난 줄은 뺀다."""
    lat = pd.to_numeric(df["위도"], errors="coerce")
    lon = pd.to_numeric(df["경도"], errors="coerce")
    ok = lat.between(33, 39) & lon.between(124, 132)
    d = df[ok].copy()
    x, y = LL_TO_5179.transform(lon[ok].values, lat[ok].values)
    d["grid_id"] = [f"{int(a // CELL)}_{int(b // CELL)}" for a, b in zip(x, y)]
    return d, int((~ok).sum())


def count_per_cell(df, name, cells):
    """기준 칸 안에 있는 줄만 남기고, 같은 좌표는 한 곳으로 합친 뒤, 칸마다 개수를 센다."""
    inside = df[df["grid_id"].isin(set(cells))]
    one = inside.drop_duplicates(["위도", "경도"])
    print(f"  {name}: 기준 칸 안 {len(inside):,}줄 → 같은 위치 합친 뒤 {len(one):,}곳")
    return one.groupby("grid_id").size().rename(name)


def fill_cells(counts, cells):
    """기준 데이터셋의 모든 칸을 넣고, 시설이 없는 칸은 0."""
    d = pd.DataFrame({"grid_id": cells}).merge(
        counts.rename_axis("grid_id").reset_index(), on="grid_id", how="left")
    cols = [c for c in d.columns if c != "grid_id"]
    d[cols] = d[cols].fillna(0).astype(int)
    return d.sort_values("grid_id"), cols
