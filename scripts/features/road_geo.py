"""road_link.py, road_speed.py 가 같이 쓰는 도형 함수.

칸 번호(grid_id = cx_cy)는 기준 데이터셋과 같은 방식: EPSG:5179(미터 좌표) → 500으로 나눈 몫.
표준노드링크(MOCT_LINK.shp)는 EPSG:5186(중부원점 TM) 좌표라 5179로 바꿔서 쓴다.
"""
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from pyproj import Transformer

CELL = 500
STEP = 10  # 선을 10m 간격으로 잘라 조각마다 어느 칸인지 본다
TM_TO_5179 = Transformer.from_crs(5186, 5179, always_xy=True)
LL_TO_5179 = Transformer.from_crs(4326, 5179, always_xy=True)


def pieces(xs, ys):
    """선(5179 좌표 꼭짓점들)을 STEP 간격 조각으로 나눠 (grid_id, 조각 길이 m) 를 돌려준다."""
    out = []
    for i in range(len(xs) - 1):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        n = max(1, int(math.ceil(seg / STEP)))
        for t in (np.arange(n) + 0.5) / n:  # 조각 가운데 점
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            out.append((f"{int(x // CELL)}_{int(y // CELL)}", seg / n))
    return out


def norm_name(x):
    """도로명 비교용 (공백·괄호·점 제거)."""
    return re.sub(r"[\s\(\)（）·.]", "", str(x or ""))


def read_links(shp_path, fields):
    """MOCT_LINK.shp → [(속성 dict, [(grid_id, 길이), ...]), ...]  (서울 주변만)"""
    import shapefile  # pip install pyshp

    p = Path(shp_path)
    r = shapefile.Reader(str(p.with_suffix("")), encoding="cp949")
    names = [f[0] for f in r.fields[1:]]
    out = []
    for sr in r.iterShapeRecords():
        pts = sr.shape.points
        if not pts:
            continue
        mx, my = pts[len(pts) // 2]
        if not (170000 <= mx <= 230000 and 520000 <= my <= 570000):  # 서울 + 경계 근처 (중부원점 TM)
            continue
        rec = dict(zip(names, sr.record))
        xs, ys = TM_TO_5179.transform([a for a, _ in pts], [b for _, b in pts])
        out.append(({k: rec[k] for k in fields}, pieces(xs, ys)))
    return out


def cell_lengths(items):
    """[(key, [(grid_id, 길이), ...]), ...] → {grid_id: {key: 길이 합}}"""
    cells = defaultdict(lambda: defaultdict(float))
    for key, ps in items:
        for g, L in ps:
            cells[g][key] += L
    return cells
