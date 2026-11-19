"""표준노드링크(MOCT_LINK.shp) → web/data/road_links_seoul.csv  (길안내 제한속도 표지판용)

  python web/make_road_limits.py data/service/moct/MOCT_LINK.shp

도로 구간(링크)마다 제한속도(MAX_SPD)와 모양을 저장한다. 서울 주변, 제한속도가 있는 구간만.
좌표는 EPSG:5179 (미터) 로 바꿔 "x y;x y;..." 로 저장.
필요한 패키지: pip install pyshp pyproj
"""
import sys
from pathlib import Path

import pandas as pd
from pyproj import Transformer

TM_TO_5179 = Transformer.from_crs(5186, 5179, always_xy=True)
OUT = Path(__file__).resolve().parent / "data" / "road_links_seoul.csv"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    import shapefile

    r = shapefile.Reader(str(Path(sys.argv[1]).with_suffix("")), encoding="cp949", encodingErrors="replace")
    names = [f[0] for f in r.fields[1:]]
    rows = []
    for sr in r.iterShapeRecords():
        pts = sr.shape.points
        if not pts:
            continue
        mx, my = pts[len(pts) // 2]
        if not (170000 <= mx <= 230000 and 520000 <= my <= 570000):  # 서울 + 경계 근처 (중부원점 TM)
            continue
        rec = dict(zip(names, sr.record))
        spd = int(rec["MAX_SPD"] or 0)
        if spd <= 0:
            continue
        xs, ys = TM_TO_5179.transform([a for a, _ in pts], [b for _, b in pts])
        rows.append((rec["LINK_ID"], str(rec["ROAD_NAME"] or "").strip(), spd,
                     ";".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys))))
    df = pd.DataFrame(rows, columns=["link_id", "도로명", "제한속도", "모양"])
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"  {len(df):,}개 구간 → {OUT}")


if __name__ == "__main__":
    main()
