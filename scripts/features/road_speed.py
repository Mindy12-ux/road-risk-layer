"""칸의 도로 통행속도 → data/features/road_speed.csv  (키: date, 발생시, grid_id)

  python scripts/features/road_speed.py 경로/topis_hourly 경로/TOPIS_구간위치.csv 경로/MOCT_LINK.shp

서울시 TOPIS 도로별 시간대별 통행속도(차들의 평균 속도, 사고 차량 속도 아님)를 500m 칸에 붙인다.
  - 칸_직전1시간_통행속도_kmh : 그 칸 도로의 1시간 전 평균 통행속도 (발생시 0시면 전날 23시)
                                사고가 나면 그 시간 도로가 막히므로, 사고 영향이 안 섞인 1시간 전 값을 쓴다
  - 칸_평소속도_kmh           : 같은 칸 도로의 같은 달·같은 시각 평균 (그 달 날짜들의 평균)
  - 칸_직전1시간_속도비율     : 직전1시간 속도 ÷ 같은 시각 평소 속도 (1보다 작으면 평소보다 막힘)
  - 칸_속도출처               : 0 없음, 1 구간(칸을 지나는 TOPIS 구간), 2 도로평균(칸을 지나는 도로의 도로명 평균)

어떤 속도를 쓰나
  1) 구간: TOPIS 구간 중 위치를 찾은 것(TOPIS_구간위치.csv)이 칸을 지나면 그 구간들의 평균
  2) 도로평균: 없으면 칸을 지나는 도로(표준노드링크)의 도로명과 같은 TOPIS 도로의 그 날짜·그 시각 평균
  3) 둘 다 없으면 빈 값

원자료
  - topis_hourly/YYYYMM.csv : TOPIS 도로별 일자별 통행속도 2020~2025 (xlsx → csv, 'HH시' 칸은 (HH-1)~HH시 평균)
  - TOPIS_구간위치.csv       : TOPIS 링크아이디별 경로(위도 경도;...)  (TOPIS 번호는 표준노드링크와 달라 따로 찾음)
  - MOCT_LINK.shp           : 국토교통부 전국표준노드링크
칸 번호는 기준 데이터셋과 같은 방식 (EPSG:5179 → 500으로 나눈 몫). keys.csv 에 나오는 조합만 저장.
필요한 패키지: pip install pyproj pyshp
"""
import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BASE_DIR, FEAT_DIR, normalize_keys, read_table, write_csv  # noqa: E402
from road_geo import LL_TO_5179, norm_name, pieces, read_links  # noqa: E402


def fnum(x):
    try:
        x = float(x)
    except ValueError:
        return np.nan
    return x if x > 0 else np.nan


def month_mean(tbl):
    """(키, 날짜) → 시간별 [합, 개수]  ⇒  (키, 연월) → 시간별 평균의 [합, 개수]"""
    m = defaultdict(lambda: [np.zeros(24), np.zeros(24)])
    for (k, d), (sm, c) in tbl.items():
        ok = c > 0
        mm = m[(k, d[:6])]
        mm[0][ok] += sm[ok] / c[ok]
        mm[1] += ok
    return m


def value(tbl, k, d, h):
    a = tbl.get((k, d))
    return a[0][h] / a[1][h] if a is not None and a[1][h] > 0 else None


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    topis_dir, geo_csv, shp = sys.argv[1:]
    keys = normalize_keys(read_table(BASE_DIR / "keys.csv"))
    cells = set(keys["grid_id"])

    # 칸 → 지나는 TOPIS 구간
    cell_seg = defaultdict(set)
    with open(geo_csv, encoding="utf-8-sig") as f:
        for g in csv.DictReader(f):
            path = g["경로(위도 경도;...)"]
            if not path:
                continue
            ll = [tuple(map(float, q.split())) for q in path.split(";")]
            xs, ys = LL_TO_5179.transform([b for _, b in ll], [a for a, _ in ll])
            for c, _ in pieces(xs, ys):
                if c in cells:
                    cell_seg[c].add(g["링크아이디"])
    segs = set().union(*cell_seg.values())
    print(f"  TOPIS 구간이 지나는 칸 {len(cell_seg):,}개 (구간 {len(segs):,}개)")

    # 칸 → 지나는 도로명
    cell_road = defaultdict(set)
    for rec, ps in read_links(shp, ["ROAD_NAME"]):
        name = norm_name(rec["ROAD_NAME"])
        if name:
            for c, _ in ps:
                if c in cells:
                    cell_road[c].add(name)

    # TOPIS 시간별 속도: 구간별 / 도로명별 (날짜마다)
    seg = defaultdict(lambda: [np.zeros(24), np.zeros(24)])
    road = defaultdict(lambda: [np.zeros(24), np.zeros(24)])
    files = sorted(Path(topis_dir).glob("??????.csv"))
    for p in files:
        with open(p, encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd)
            for row in rd:
                v = np.array([fnum(x) for x in row[6:30]])
                ok = ~np.isnan(v)
                v0 = np.where(ok, v, 0)
                a = road[(norm_name(row[2]), row[1])]
                a[0] += v0
                a[1] += ok
                if row[0] in segs:
                    a = seg[(row[0], row[1])]
                    a[0] += v0
                    a[1] += ok
    print(f"  TOPIS 파일 {len(files)}개 읽음")
    seg_m, road_m = month_mean(seg), month_mean(road)
    roads_known = {k for k, _ in road}
    cell_road = {c: s & roads_known for c, s in cell_road.items()}

    def speed(c, d, h):
        """(통행속도, 평소속도, 출처) — 그 칸·날짜(YYYYMMDD)·시각"""
        for src, tbl, tblm, ks in ((1, seg, seg_m, cell_seg.get(c, ())), (2, road, road_m, cell_road.get(c, ()))):
            vs = [(value(tbl, k, d, h), value(tblm, k, d[:6], h)) for k in ks]
            vs = [(v, u) for v, u in vs if v is not None and u is not None]
            if vs:
                return np.mean([v for v, _ in vs]), np.mean([u for _, u in vs]), src
        return None, None, 0

    out = []
    for date, h, c in keys[["date", "발생시", "grid_id"]].drop_duplicates().itertuples(index=False):
        d = date.replace("-", "")
        _, usual, _ = speed(c, d, h)
        if h > 0:
            pd_, ph = d, h - 1
        else:
            pd_, ph = (dt.date(int(d[:4]), int(d[4:6]), int(d[6:])) - dt.timedelta(days=1)).strftime("%Y%m%d"), 23
        pv, pu, src = speed(c, pd_, ph)
        out.append((date, h, c,
                    round(pv, 1) if pv is not None else None,
                    round(usual, 1) if usual is not None else None,
                    round(pv / pu, 3) if pv is not None else None,
                    src))
    cols = ["칸_직전1시간_통행속도_kmh", "칸_평소속도_kmh", "칸_직전1시간_속도비율", "칸_속도출처"]
    d = pd.DataFrame(out, columns=["date", "발생시", "grid_id"] + cols)
    write_csv(d, FEAT_DIR / "road_speed.csv")
    print(f"data/features/road_speed.csv  {len(d):,}줄, 변수 {cols}")
    print("  속도출처:", d["칸_속도출처"].map({0: "없음", 1: "구간", 2: "도로평균"}).value_counts().to_dict())


if __name__ == "__main__":
    main()
