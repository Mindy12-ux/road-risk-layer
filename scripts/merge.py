"""기준 데이터셋(data/base) + 고른 변수 파일(data/features) → data/merged/

  python scripts/merge.py                      # 쓸 수 있는 변수 목록 보기
  python scripts/merge.py sun pop              # 기준 데이터셋에 sun, pop 붙이기 → data/merged/
  python scripts/merge.py sun --out data/merged_sun   # 결과 폴더 바꾸기
  python scripts/merge.py sun --keep-keys      # date, grid_id 컬럼도 남기기 (기본: 빼서 원본과 같은 모양)
  python scripts/merge.py --check speed        # 내 변수 파일 형식만 검사 (붙이지 않음)

변수 파일 규칙 (data/features/<이름>.csv 또는 .parquet)
  - 키 컬럼(date, 발생시, grid_id) 중 1개 이상 + 내 변수 컬럼
      날짜·시간마다 다른 값      → date, 발생시        (예: 일출·일몰)
      칸마다 다른 값(시간 무관)  → grid_id             (예: 도로 밀도)
      칸·날짜·시간마다 다른 값   → date, 발생시, grid_id (예: 생활인구)
  - 같은 키 조합이 두 줄 이상 있으면 안 됨
  - 변수 컬럼 이름이 기준 데이터셋이나 다른 변수와 겹치면 안 됨 (예: "속도" 말고 "평균속도_kmh")
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from common import BASE_DIR, FEAT_DIR, KEYS, MERGED_DIR, normalize_keys, read_table, write_csv

EXTS = [".csv", ".csv.gz", ".parquet"]


def available():
    out = {}
    for p in sorted(FEAT_DIR.iterdir()) if FEAT_DIR.exists() else []:
        for e in EXTS:
            if p.name.endswith(e):
                out[p.name[: -len(e)]] = p
    return out


def load_feature(name, feats, base_cols):
    if name not in feats:
        sys.exit(f"[오류] data/features/ 에 '{name}' 파일이 없습니다. "
                 f"있는 것: {', '.join(feats) or '(없음)'}  →  'dvc pull' 을 했는지 확인하세요.")
    f = read_table(feats[name])
    keys = [k for k in KEYS if k in f.columns]
    if not keys:
        sys.exit(f"[오류] {name}: 키 컬럼(date, 발생시, grid_id)이 하나도 없습니다. 컬럼: {list(f.columns)}")
    f = normalize_keys(f)
    dup = f.duplicated(keys).sum()
    if dup:
        sys.exit(f"[오류] {name}: 같은 {keys} 조합이 {dup}줄 중복됩니다. 한 조합에 한 줄만 있어야 합니다.")
    vals = [c for c in f.columns if c not in keys]
    clash = [c for c in vals if c in base_cols]
    if clash:
        sys.exit(f"[오류] {name}: 컬럼 이름 {clash} 이(가) 이미 있습니다. 다른 이름으로 바꿔 주세요.")
    return f, keys, vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="*", help="붙일 변수 이름 (data/features/ 의 파일 이름)")
    ap.add_argument("--out", default=str(MERGED_DIR))
    ap.add_argument("--keep-keys", action="store_true")
    ap.add_argument("--check", metavar="이름")
    a = ap.parse_args()

    feats = available()
    splits = sorted(p for p in BASE_DIR.glob("*.csv") if p.stem != "keys") if BASE_DIR.exists() else []
    if not splits:
        sys.exit("[오류] data/base/ 가 비어 있습니다. 먼저 'dvc pull' 을 하세요.")

    if not a.features and not a.check:
        print("쓸 수 있는 변수:", ", ".join(feats) or "(없음)")
        print("예) python scripts/merge.py " + " ".join(list(feats)[:2]))
        return

    base_cols = set(pd.read_csv(splits[0], nrows=0, encoding="utf-8-sig").columns)

    if a.check:
        f, keys, vals = load_feature(a.check, feats, base_cols)
        keyset = normalize_keys(read_table(BASE_DIR / "keys.csv"))[keys].drop_duplicates()
        hit = keyset.merge(f[keys], on=keys, how="left", indicator=True)["_merge"].eq("both").mean()
        print(f"[통과] {a.check}: {len(f):,}줄, 키 {keys}, 변수 {len(vals)}개 {vals}")
        print(f"       기준 데이터셋 키 조합 중 {hit:.1%} 가 이 파일에 있음 (나머지는 붙였을 때 빈 값)")
        return

    loaded, used = [], set(base_cols)
    for name in a.features:
        f, keys, vals = load_feature(name, feats, used)
        used |= set(vals)
        loaded.append((name, f, keys, vals))

    out = Path(a.out)
    for sp in splits:
        d = read_table(sp)
        n = len(d)
        d = normalize_keys(d)
        for name, f, keys, vals in loaded:
            d = d.merge(f, on=keys, how="left")
        assert len(d) == n, "행 수가 바뀌었습니다 (버그)"
        if not a.keep_keys:
            d = d.drop(columns=["date", "grid_id"])
        write_csv(d, out / sp.name)
        miss = {v: f"{d[v].isna().mean():.1%}" for _, _, _, vals in loaded for v in vals if d[v].isna().any()}
        print(f"{out.name}/{sp.name}: {n:,}행, 컬럼 {d.shape[1]}개" + (f", 빈 값 비율 {miss}" if miss else ""))


if __name__ == "__main__":
    main()
