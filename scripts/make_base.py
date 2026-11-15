"""원본 전처리본(train/valid/test) → 기준 데이터셋 (data/base/)

원본과 같은 행·같은 컬럼·같은 순서에 키 컬럼 2개만 맨 뒤에 붙인다.
  date    : 원본에는 날짜가 없어서 복원한다
            - 사고 행  : taas_accidents.csv 사고번호 앞 8자리 (연도·월·요일·시·위경도로 연결)
            - 무사고 행: no_accident.csv 기준날짜 (같은 키로 연결)
  grid_id : cx, cy → "cx_cy"
  (발생시는 원본에 이미 있음)
그리고 data/base/keys.csv (전체 행에 나오는 date·발생시·grid_id 조합)를 만든다.
변수가 아주 큰 경우(생활인구 등) 이 조합에 해당하는 줄만 남기면 파일이 작아진다.

사용 (저장소 맨 위 폴더에서, 데이터 담당자가 한 번만):
  python scripts/make_base.py --split train=원본/train.csv --split valid=원본/valid.csv --split test=원본/test.csv \
      --taas data/raw/taas_accidents.csv --noacc data/raw/no_accident.csv
"""
import argparse

import numpy as np
import pandas as pd

from common import BASE_DIR, KEYS, read_table, write_csv

KEY6 = ["연도", "월", "요일", "발생시", "위도", "경도"]
WD = {"월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, "금요일": 4, "토요일": 5, "일요일": 6}


def add_dates(al, taas, noacc):
    a = read_table(taas)
    a["date"] = pd.to_datetime(a["사고번호"].astype(str).str[:8])
    a["월"], a["요일"] = a["date"].dt.month, a["요일"].map(WD)
    a = a.drop_duplicates(KEY6)[KEY6 + ["date"]]

    n = read_table(noacc)
    n["date"] = pd.to_datetime(n["기준날짜"])
    n["월"], n["요일"] = n["date"].dt.month, n["요일"].map(WD)
    n = n.drop_duplicates(KEY6)[KEY6 + ["date"]]

    pos = (al["사고여부"] > 0).values
    al["date"] = pd.NaT
    al.loc[pos, "date"] = al.loc[pos, KEY6].merge(a, on=KEY6, how="left")["date"].values
    al.loc[~pos, "date"] = al.loc[~pos, KEY6].merge(n, on=KEY6, how="left")["date"].values
    miss = al["date"].isna().sum()
    assert miss == 0, f"날짜를 못 찾은 행이 {miss}개 있습니다 (taas/no_accident 파일이 원본과 맞는지 확인)"
    al["date"] = pd.to_datetime(al["date"]).dt.strftime("%Y-%m-%d")
    return al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", required=True, help="이름=경로 (여러 번)")
    ap.add_argument("--taas", required=True)
    ap.add_argument("--noacc", required=True)
    a = ap.parse_args()

    parts, cols = [], {}
    for s in a.split:
        name, path = s.split("=", 1)
        d = read_table(path)
        cols[name] = list(d.columns)
        d["_split"], d["_row"] = name, np.arange(len(d))
        parts.append(d)
    al = pd.concat(parts, ignore_index=True)

    print("날짜 복원 중…")
    al = add_dates(al, a.taas, a.noacc)
    al["grid_id"] = al["cx"].astype(int).astype(str) + "_" + al["cy"].astype(int).astype(str)

    for name in cols:
        d = al[al["_split"] == name].sort_values("_row")[cols[name] + ["date", "grid_id"]]
        write_csv(d, BASE_DIR / f"{name}.csv")
        print(f"  data/base/{name}.csv  {len(d):,}행  (원본 컬럼 {len(cols[name])}개 + date, grid_id)")
    keys = al[KEYS].drop_duplicates().sort_values(KEYS)
    write_csv(keys, BASE_DIR / "keys.csv")
    print(f"  data/base/keys.csv  {len(keys):,}개 조합")


if __name__ == "__main__":
    main()
