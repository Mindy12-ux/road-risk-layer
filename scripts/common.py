"""공통 설정: 키 컬럼, 경로, 읽기/쓰기 함수."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE_DIR = DATA / "base"          # 기준 데이터셋 (train/valid/test + 키)
FEAT_DIR = DATA / "features"      # 팀원별 변수 파일
MERGED_DIR = DATA / "merged"      # merge.py 결과 (깃·DVC에 안 올림, 누구나 다시 만들 수 있음)

# 모든 변수 파일이 쓸 수 있는 키. 변수 파일은 이 중 1개 이상을 가져야 한다.
#   date    : 날짜 "2021-07-18"
#   발생시  : 시간 0~23 (정수)
#   grid_id : 500m 칸 "cx_cy" (예: "1910_3905"), grid/grid.parquet 의 grid_id 와 같음
KEYS = ["date", "발생시", "grid_id"]


def read_table(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def write_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig: 엑셀에서 한글 안 깨짐


def normalize_keys(df):
    """키 컬럼 형식 통일 (날짜 문자열, 시간 정수, 칸 문자열)."""
    df = df.copy()
    if "date" in df:
        s = df["date"].astype(str).str.strip()
        fmt = "%Y%m%d" if s.str.fullmatch(r"\d{8}").all() else None
        df["date"] = pd.to_datetime(s, format=fmt).dt.strftime("%Y-%m-%d")
    if "발생시" in df:
        df["발생시"] = df["발생시"].astype(int)
    if "grid_id" in df:
        df["grid_id"] = df["grid_id"].astype(str).str.strip()
    return df
