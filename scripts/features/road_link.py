"""칸 안의 도로 → data/features/road_link.csv  (키: grid_id)

  python scripts/features/road_link.py 경로/MOCT_LINK.shp

500m 칸 안을 지나는 도로 구간(표준노드링크)을 길이로 가중해 요약한다. 도로는 시간에 따라 안 바뀌므로 키는 grid_id 하나.
  - 칸_도로연장_m          : 칸 안 도로 길이 합 (표준노드링크는 방향마다 따로 있어 양방향 도로는 두 번 셈)
  - 칸_평균제한속도_kmh    : 제한속도의 길이 가중 평균 (제한속도 정보 있는 구간만)
  - 칸_최대차로수          : 칸 안 구간 중 가장 많은 차로 수
  - 칸_최고도로등급        : 칸 안 가장 높은 도로 등급 0 고속국도, 1 도시고속도로, 2 일반국도, 3 특별·광역시도,
                              4 국가지원지방도, 5 지방도, 6 시·군도, 7 기타  (숫자가 작을수록 큰 도로)
  - 칸_간선도로비율        : 길이 중 고속국도·도시고속도로·일반국도 비율
  - 칸_터널비율            : 길이 중 터널 비율
  - 칸_교량고가비율        : 길이 중 교량·고가도로 비율
도로가 하나도 없는 칸은 도로연장 0, 나머지 빈 값.

원자료: 국토교통부 국가교통정보센터 전국표준노드링크 (MOCT_LINK.shp, https://www.its.go.kr/nodelink/nodelinkRef)
칸 번호는 기준 데이터셋과 같은 방식 (EPSG:5179 → 500으로 나눈 몫). 기준 데이터셋(keys.csv)에 나오는 칸만 저장.
필요한 패키지: pip install pyproj pyshp
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, FEAT_DIR, normalize_keys, read_table, write_csv  # noqa: E402
from road_geo import read_links  # noqa: E402

RANK_CODE = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4, 106: 5, 107: 6, 108: 7}


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cells = set(normalize_keys(read_table(BASE_DIR / "keys.csv"))["grid_id"])
    links = read_links(sys.argv[1], ["MAX_SPD", "LANES", "ROAD_RANK", "ROAD_TYPE"])
    print(f"  도로 구간 {len(links):,}개 (서울 주변)")

    rows = []
    for rec, ps in links:
        spd = float(rec["MAX_SPD"] or 0)
        lanes = float(rec["LANES"] or 0)
        rank = RANK_CODE.get(int(rec["ROAD_RANK"]), 7)
        typ = int(rec["ROAD_TYPE"])
        for g, L in ps:
            if g in cells:
                rows.append((g, L, spd, lanes, rank, typ))
    p = pd.DataFrame(rows, columns=["grid_id", "L", "spd", "lanes", "rank", "typ"])
    p["spdL"] = np.where(p.spd > 0, p.spd * p.L, 0)
    p["spdW"] = np.where(p.spd > 0, p.L, 0)
    p["artL"] = np.where(p["rank"] <= 2, p.L, 0)
    p["tunL"] = np.where(p.typ == 4, p.L, 0)
    p["brgL"] = np.where(p.typ.isin([1, 3]), p.L, 0)
    p["lanes"] = p.lanes.where(p.lanes > 0)

    g = p.groupby("grid_id").agg(L=("L", "sum"), spdL=("spdL", "sum"), spdW=("spdW", "sum"), lanes=("lanes", "max"),
                                 rank=("rank", "min"), artL=("artL", "sum"), tunL=("tunL", "sum"), brgL=("brgL", "sum"))
    out = pd.DataFrame({
        "칸_도로연장_m": g.L.round(0),
        "칸_평균제한속도_kmh": (g.spdL / g.spdW.replace(0, np.nan)).round(1),
        "칸_최대차로수": g.lanes,
        "칸_최고도로등급": g["rank"],
        "칸_간선도로비율": (g.artL / g.L).round(3),
        "칸_터널비율": (g.tunL / g.L).round(3),
        "칸_교량고가비율": (g.brgL / g.L).round(3),
    })
    d = pd.DataFrame({"grid_id": sorted(cells)}).merge(out.rename_axis("grid_id").reset_index(), on="grid_id", how="left")
    d["칸_도로연장_m"] = d["칸_도로연장_m"].fillna(0)
    for c in ["칸_최대차로수", "칸_최고도로등급"]:
        d[c] = d[c].astype("Int64")

    write_csv(d, FEAT_DIR / "road_link.csv")
    cols = [c for c in d.columns if c != "grid_id"]
    print(f"data/features/road_link.csv  {len(d):,}줄 (칸), 변수 {cols}")
    print(f"  도로가 하나도 없는 칸: {(d['칸_도로연장_m'] == 0).sum():,}개")


if __name__ == "__main__":
    main()
