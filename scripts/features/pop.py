"""생활인구 → data/features/pop.csv  (키: date, 발생시, grid_id)

  python scripts/features/pop.py 경로/pop_500m
pop_500m/YYYYMM.csv.gz 는 population/run_monthly.py 결과 (전체 칸 x 날짜 x 시간이라 매우 큼).
그래서 기준 데이터셋에 실제로 나오는 조합(data/base/keys.csv)만 남겨 작게 만든다.
  - 그 달 파일은 있는데 칸이 없으면 = 사람이 없는 칸(한강, 산) → 0
  - 그 달 파일이 없거나 그 컬럼이 통째로 비어 있으면(단기외국인 불량 달) → 빈 값
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BASE_DIR, FEAT_DIR, normalize_keys, read_table, write_csv  # noqa: E402

REN = {"pop_local": "생활인구_내국인", "pop_long": "생활인구_장기외국인",
       "pop_temp": "생활인구_단기외국인", "pop_total": "생활인구_합계"}
K = ["date", "발생시", "grid_id"]

src = Path(sys.argv[1])
keys = normalize_keys(read_table(BASE_DIR / "keys.csv"))
keys["_ym"] = keys["date"].str[:7].str.replace("-", "")
got, missing = [], []
for ym, need in keys.groupby("_ym"):
    f = src / f"{ym}.csv.gz"
    if not f.exists():
        missing.append(ym)
        continue
    p = normalize_keys(pd.read_csv(f, dtype={"grid_id": str}).rename(columns={"hour": "발생시", **REN}))
    m = need[K].merge(p[K + [c for c in REN.values() if c in p]], on=K, how="left")
    for c in REN.values():
        if c in p and p[c].notna().any():
            m[c] = m[c].fillna(0)
    got.append(m)
    print(f"  {ym}: {len(m):,}줄")
d = pd.concat(got, ignore_index=True)
write_csv(d, FEAT_DIR / "pop.csv")
print(f"data/features/pop.csv  {len(d):,}줄")
if missing:
    print(f"※ pop_500m 에 없는 달 {len(missing)}개 ({missing[0]}~{missing[-1]} 중) → 그 달 행은 붙였을 때 빈 값")
