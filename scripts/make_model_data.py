"""merge.py 결과(data/merged) → 모델 학습용 데이터(data/model)  — dataset_v2 와 같은 규칙

  python scripts/merge.py sun zone road_link road_speed pop --keep-keys    # 먼저 (--keep-keys 꼭!)
  python scripts/make_model_data.py                                         # → data/model/train·valid·test.csv

하는 일 (dataset_v2 만들 때와 같은 규칙)
  1) 사고 행의 날씨·노면 = 무사고 파일의 같은 날짜·시간 값으로 통일 (출처가 달라서 생기는 정답 누출 방지)
     → data/raw/no_accident.csv 필요 (dvc pull 하면 생김)
  2) 위도·경도·연도·date·grid_id 제거 (좌표 암기 방지. 위치는 cx, cy + 칸 이력 피처 + 붙인 칸 변수로)
  3) 연도별 분할: 2021~2023 train / 2024 valid / 2025 test (2020은 '작년까지' 이력이 없어 제외)
  4) 라벨 이름 사고여부 → 사고심각도 (0 무사고, 1 경상, 2 중상, 3 사망)
  붙인 변수(sun, pop …)는 그대로 따라감. 빈 값은 그대로 둠 (LightGBM은 빈 값을 알아서 처리).

옵션
  --merged data/merged_xxx   merge.py 결과 폴더가 다를 때
  --out data/model_xxx       결과 폴더 바꾸기 (실험별로 나눌 때)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from common import DATA, MERGED_DIR, read_table, write_csv

W = {"맑음": 0, "비": 1, "눈": 2}
S = {"건조": 0, "젖음/습기": 1, "서리/결빙": 2, "적설": 3}
DROP = ["위도", "경도", "연도", "date", "grid_id"]
SPLIT = {"train": (2021, 2023), "valid": (2024, 2024), "test": (2025, 2025)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default=str(MERGED_DIR))
    ap.add_argument("--noacc", default=str(DATA / "raw" / "no_accident.csv"))
    ap.add_argument("--out", default=str(DATA / "model"))
    a = ap.parse_args()

    files = sorted(p for p in Path(a.merged).glob("*.csv"))
    if not files:
        sys.exit(f"[오류] {a.merged} 에 CSV가 없습니다. 먼저 'python scripts/merge.py 변수들 --keep-keys' 를 하세요.")
    al = pd.concat([read_table(p) for p in files], ignore_index=True)
    if "date" not in al:
        sys.exit("[오류] date 컬럼이 없습니다. merge.py 를 '--keep-keys' 를 붙여서 다시 실행하세요.\n"
                 "   예) python scripts/merge.py sun pop --keep-keys")
    if not Path(a.noacc).exists():
        sys.exit(f"[오류] {a.noacc} 가 없습니다. 'dvc pull' 을 했는지 확인하세요.")
    base_cols = ["월", "요일", "발생시", "주말", "밤", "출퇴근", "기상상태", "노면상태", "도로_큰분류", "도로_작은분류",
                 "칸_작년까지사고", "칸_작년까지큰사고", "칸_작년까지같은시간대사고", "주변_작년까지사고", "cx", "cy"]
    added = [c for c in al.columns if c not in base_cols + DROP + ["사고여부"]]
    print(f"{a.merged}: {len(al):,}행, 붙인 변수 {len(added)}개 {added}")

    # 1) 무사고 파일 → 시간별 날씨표 (같은 날짜·시간이면 서울 전체 같은 값)
    n = read_table(a.noacc)
    n["dt"] = pd.to_datetime(n["기준날짜"]) + pd.to_timedelta(n["발생시"], unit="h")
    hw = n.groupby("dt").agg(w=("기상상태", "first"), s=("노면상태", "first"))
    hw["w"], hw["s"] = hw["w"].map(W), hw["s"].map(S)

    dt = pd.to_datetime(al["date"]) + pd.to_timedelta(al["발생시"].astype(int), unit="h")
    pos = al["사고여부"] > 0
    al.loc[pos, "기상상태"] = dt[pos].map(hw["w"])
    al.loc[pos, "노면상태"] = dt[pos].map(hw["s"])
    miss = al[["기상상태", "노면상태"]].isna().any(axis=1)
    al = al[~miss].copy()
    al[["기상상태", "노면상태"]] = al[["기상상태", "노면상태"]].astype(int)
    print(f"사고 행 날씨·노면 통일 (날씨표에 없는 시간 {miss.sum()}행 제외)")

    # 2)~4)
    al = al.rename(columns={"사고여부": "사고심각도"})
    cols = base_cols + added + ["사고심각도"]
    out = Path(a.out)
    for k, (y0, y1) in SPLIT.items():
        d = al[al["연도"].between(y0, y1)].sample(frac=1, random_state=42)[cols]
        write_csv(d, out / f"{k}.csv")
        print(f"  {out.name}/{k}.csv  {len(d):,}행 ({y0}~{y1})  심각도별 {d['사고심각도'].value_counts().sort_index().to_dict()}")
    na = al.loc[al["연도"] >= 2021, added].isna().mean()
    na = na[na > 0]
    if len(na):
        print("붙인 변수 빈 값 비율 (모델은 그대로 학습함):", {k: f"{v:.1%}" for k, v in na.items()})


if __name__ == "__main__":
    main()
