"""
전처리 (새 버전) : 사고가 날까? (0 = 사고없음, 1 = 사고) 를 맞히는 모델용 데이터 만들기

사용법 (ML project 폴더에서):
    python scripts/preprocess_v2.py

입력:
data/processed/합본_사고난+사고안난.csv           사고 난 + 안 난 데이터
data/raw/taas/taas_accidents_사망중상경상.csv     과거 사고 기록, 사고 날짜 복원
data/raw/osm/seoul_gu_boundaries.geojson          서울 밖 줄 거르기
data/raw/weather/seoul_hourly_openmeteo.json      기상상태·노면상태 (모든 줄을 Open-Meteo로 통일)
road-risk-layer/data/features/*.csv               팀원 변수 (도로구조, 생활인구, 통행속도, 일출일몰)

출력:
data/model_input/train.csv, valid.csv, test.csv, info.json

변수 31개 = 기본 14 + 1순위 7 + 2순위 10
- 1순위(칸별 과거 사고 성격)는 '그 해 직전 3년' 기록만 씁니다. 2020년은 과거 기록이 없어 빈칸.
- 경상/중상/사망은 '심각도_참고' 칸으로만 남기고 모델에는 넣지 않습니다.
"""

import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_PATH = os.path.join(ROOT, "data", "processed", "합본_사고난+사고안난.csv")
RAW_ACC_PATH = os.path.join(ROOT, "data", "raw", "taas", "taas_accidents_사망중상경상.csv")
GU_PATH = os.path.join(ROOT, "data", "raw", "osm", "seoul_gu_boundaries.geojson")
WEATHER_PATH = os.path.join(ROOT, "data", "raw", "weather", "seoul_hourly_openmeteo.json")
FEAT_DIR = os.path.join(ROOT, "road-risk-layer", "data", "features")
OUT_DIR = os.path.join(ROOT, "data", "model_input")

CELL = 500          # 칸 크기 (미터)
HIST_YEARS = 3      # 과거 기록: 직전 3년
SMOOTH = 5          # 비율 스무딩 강도 (과거 사고가 적은 칸은 서울 평균 쪽으로 당김)
SPLITS = {"train": 0.6, "valid": 0.2, "test": 0.2}
SEED = 42

TARGET = "사고여부"                                         # 0 = 사고없음, 1 = 사고
SEVERITY = {"사고없음": 0, "경상사고": 1, "중상사고": 2, "사망사고": 3}   # 참고용 (모델에 안 넣음)
WEEKDAYS = {"월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, "금요일": 4, "토요일": 5, "일요일": 6}
CATEGORIES = {
    "기상상태": ["맑음", "비", "눈"],
    "노면상태": ["건조", "젖음/습기", "서리/결빙", "적설"],
    "도로_큰분류": ["단일로", "교차로", "기타", "미분류"],
    "도로_작은분류": ["기타", "교차로안", "교차로부근", "교차로횡단보도내", "교량", "터널", "고가도로위", "지하차도(도로)내", "미분류"],
}

BASE = ["월", "요일", "발생시", "주말", "밤", "출퇴근",
        "기상상태", "노면상태", "도로_큰분류", "도로_작은분류",
        "칸_작년까지사고", "칸_작년까지큰사고", "칸_작년까지같은시간대사고", "주변_작년까지사고"]
P1 = ["칸_과거이륜차비율", "칸_과거보행자비율", "칸_과거노인피해비율", "칸_과거과속신호위반비율",
    "칸_과거자전거PM비율", "칸_과거차대차비율", "칸_과거추돌비율"]
P2 = {   # 팀원 변수 파일 → 쓸 칸
    "road_link": ["칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율", "칸_교량고가비율"],
    "pop": ["생활인구_합계"],
    "road_speed": ["칸_평소속도_kmh", "칸_직전1시간_속도비율"],
    "sun": ["태양고도", "박명비율"],
}
FEATURES = BASE + P1 + [c for cols in P2.values() for c in cols]

TO_METER = Transformer.from_crs(4326, 5179, always_xy=True)


def to_cell(lon, lat):
    x, y = TO_METER.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return (x // CELL).astype(int), (y // CELL).astype(int)


def load_raw():
    """TAAS 원본: 과거 기록 계산 + 사고 날짜 복원에 사용"""
    raw = pd.read_csv(RAW_ACC_PATH, encoding="utf-8-sig", dtype=str)
    raw["연도"] = raw["연도"].astype(int)
    raw["발생시"] = raw["발생시"].astype(int)
    raw["시간대"] = raw["발생시"] // 3
    raw["cx"] = (raw["X_UTMK"].astype(float) // CELL).astype(int)
    raw["cy"] = (raw["Y_UTMK"].astype(float) // CELL).astype(int)
    raw["date"] = pd.to_datetime(raw["사고번호"].str[:8]).dt.strftime("%Y-%m-%d")   # 사고번호 앞 8자리 = 날짜

    # 1순위 변수용 표시 (그 사고가 어떤 사고였나)
    two = ["이륜", "원동기"]
    bike = ["자전거", "개인형이동수단(PM)"]
    raw["이륜차"] = raw["가해차종"].isin(two) | raw["피해차종"].isin(two)
    raw["보행자"] = raw["사고유형"].str.startswith("차대사람")
    raw["노인피해"] = raw["피해자연령"].eq("65세 이상")
    raw["과속신호위반"] = raw["법규위반"].isin(["과속", "신호위반"])
    raw["자전거PM"] = raw["가해차종"].isin(bike) | raw["피해차종"].isin(bike)
    raw["차대차"] = raw["사고유형"].str.startswith("차대차")
    raw["추돌"] = raw["사고유형"].eq("차대차 - 추돌")
    return raw


FLAG_OF = {"칸_과거이륜차비율": "이륜차", "칸_과거보행자비율": "보행자", "칸_과거노인피해비율": "노인피해",
        "칸_과거과속신호위반비율": "과속신호위반", "칸_과거자전거PM비율": "자전거PM",
        "칸_과거차대차비율": "차대차", "칸_과거추돌비율": "추돌"}


def add_history(df, raw):
    """그 해 직전 3년 기록으로 기본 과거 변수 4개 + 1순위 변수 7개를 만듭니다. 미래는 절대 안 봅니다."""
    parts = []
    for year, g in df.groupby("연도"):
        g = g.copy()
        if year == 2020:   # 과거 기록이 없음 → 빈칸 (0으로 채우면 '안전한 곳'이라는 거짓 정보가 됨)
            for c in BASE[10:] + P1:
                g[c] = np.nan
            parts.append(g)
            continue
        past = raw[(raw["연도"] < year) & (raw["연도"] >= year - HIST_YEARS)]
        n_years = past["연도"].nunique()
        key = pd.MultiIndex.from_arrays([g["cx"], g["cy"]])
        by_cell = past.groupby(["cx", "cy"])
        count = by_cell.size()

        # 기본 과거 변수 (1년 평균 건수)
        g["칸_작년까지사고"] = (count / n_years).reindex(key).fillna(0).values
        big = past[past["사고내용"].isin(["중상사고", "사망사고"])].groupby(["cx", "cy"]).size() / n_years
        g["칸_작년까지큰사고"] = big.reindex(key).fillna(0).values
        band = past.groupby(["cx", "cy", "시간대"]).size() / n_years
        g["칸_작년까지같은시간대사고"] = band.reindex(
            pd.MultiIndex.from_arrays([g["cx"], g["cy"], g["시간대"]])).fillna(0).values
        neigh = np.zeros(len(g))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neigh += (count / n_years).reindex(
                    pd.MultiIndex.from_arrays([g["cx"] + dx, g["cy"] + dy])).fillna(0).values
        g["주변_작년까지사고"] = neigh

        # 1순위 변수 (스무딩 비율): (해당 사고 수 + 서울평균 × 5) ÷ (전체 사고 수 + 5)
        for col, flag in FLAG_OF.items():
            seoul = past[flag].mean()
            ratio = (by_cell[flag].sum() + seoul * SMOOTH) / (count + SMOOTH)
            g[col] = ratio.reindex(key).fillna(seoul).values   # 과거 사고 0건 칸 = 서울 평균
        parts.append(g)
    return pd.concat(parts, ignore_index=True)   # 예전 전처리와 같은 순서 (연도순) → 같은 줄이 같은 파일로 감


def openmeteo_weather():
    """Open-Meteo 시간별 날씨 → 기상상태·노면상태 (scripts/make_negatives.py와 같은 규칙)
    사고 난 줄(경찰 기록)과 사고 안 난 줄(Open-Meteo)의 출처가 달라서 모델이 '출처'를 외우는 문제를
    막기 위해, 모든 줄을 이 값으로 통일합니다."""
    with open(WEATHER_PATH, encoding="utf-8") as f:
        w = pd.DataFrame(json.load(f)["hourly"])
    sky = np.full(len(w), "맑음", dtype=object)
    sky[w["precipitation"] >= 1.0] = "비"                                   # 1시간에 1mm 이상
    sky[(w["snowfall"] > 0) & (w["precipitation"] >= 0.5)] = "눈"
    recent_rain = w["precipitation"].rolling(3, min_periods=1).sum()        # 최근 3시간
    recent_snow = w["snowfall"].rolling(3, min_periods=1).sum()
    recent_12h = w["precipitation"].rolling(12, min_periods=1).sum()        # 최근 12시간
    surface = np.full(len(w), "건조", dtype=object)
    surface[recent_rain >= 1.0] = "젖음/습기"
    surface[(recent_12h >= 1.0) & (w["temperature_2m"] <= 0)] = "서리/결빙"   # 비·눈 뒤 영하
    surface[recent_snow >= 0.5] = "적설"
    t = pd.to_datetime(w["time"])
    return pd.DataFrame({"date": t.dt.strftime("%Y-%m-%d"), "발생시": t.dt.hour,
                         "기상상태": sky, "노면상태": surface})


def add_dates(df, raw):
    """팀원 변수를 붙이려면 날짜가 필요합니다. 사고 안 난 줄은 기준날짜, 사고 난 줄은 TAAS 사고번호에서 복원."""
    k6 = ["연도", "월", "요일", "발생시", "위도", "경도"]
    r = raw.copy()
    r["월"] = pd.to_datetime(r["date"]).dt.month
    r["요일"] = r["요일"].map(WEEKDAYS)
    r["위도"], r["경도"] = r["위도"].astype(float), r["경도"].astype(float)
    r = r.drop_duplicates(k6)[k6 + ["date"]]
    pos = df[TARGET] == 1
    df["date"] = df["기준날짜"]
    df.loc[pos, "date"] = df.loc[pos, k6].merge(r, on=k6, how="left")["date"].values
    miss = df["date"].isna().sum()
    assert miss == 0, f"날짜를 못 찾은 줄이 {miss}개 있어요"
    df["grid_id"] = df["cx"].astype(str) + "_" + df["cy"].astype(str)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig", dtype=str)
    print(f"0) 불러옴: {len(df):,}줄")

    # ------------------------------------------------ 1단계: 정답 0/1 + 심각도는 참고용으로만
    df["심각도_참고"] = df["사고내용"].map(SEVERITY)
    assert df["심각도_참고"].notna().all(), "사고내용에 모르는 이름이 있어요"
    df[TARGET] = (df["심각도_참고"] > 0).astype(int)
    print("1) 정답 0/1로 바꿈:", df[TARGET].value_counts().sort_index().to_dict())

    # ------------------------------------------------ 2단계: 서울 밖 줄 빼기
    gu = gpd.read_file(GU_PATH)
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(df["경도"].astype(float), df["위도"].astype(float)), crs=4326)
    inside = gpd.sjoin(pts, gu, how="left", predicate="within").groupby(level=0)["name"].first().notna()
    removed = df.loc[~inside.values, TARGET].value_counts().to_dict()
    df = df[inside.values].reset_index(drop=True)
    print(f"2) 서울 밖 줄 뺌: {removed} → 남은 줄 {len(df):,}")

    # ------------------------------------------------ 3단계: 글자 → 숫자, 기본 새 칸
    df["연도"] = df["연도"].astype(int)
    df["발생시"] = df["발생시"].astype(int)
    df["위도"] = df["위도"].astype(float)
    df["경도"] = df["경도"].astype(float)
    df["월"] = df["발생년월"].str.extract(r"(\d+)월")[0].astype(int)
    df["요일"] = df["요일"].map(WEEKDAYS)
    df["주말"] = (df["요일"] >= 5).astype(int)
    df["밤"] = ((df["발생시"] >= 20) | (df["발생시"] <= 5)).astype(int)
    df["출퇴근"] = df["발생시"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    road = df["도로형태"].str.split(" - ", n=1, expand=True)
    df["도로_큰분류"], df["도로_작은분류"] = road[0], road[1]
    df["cx"], df["cy"] = to_cell(df["경도"], df["위도"])
    df["시간대"] = df["발생시"] // 3
    print("3) 월, 요일, 주말, 밤, 출퇴근, 도로 분류, 500m 칸 만듦")

    # ------------------------------------------------ 4단계: 과거 기록 (기본 4개 + 1순위 7개)
    raw = load_raw()
    df = add_history(df, raw)
    print(f"4) 과거 기록 변수 11개 만듦 (2020년 {(df['연도'] == 2020).sum():,}줄은 빈칸)")

    # ------------------------------------------------ 5단계: 팀원 변수 (2순위 10개) 붙이기
    df = add_dates(df, raw)
    for name, cols in P2.items():
        f = pd.read_csv(os.path.join(FEAT_DIR, f"{name}.csv"), encoding="utf-8-sig")
        keys = [k for k in ["date", "발생시", "grid_id"] if k in f.columns]
        before = len(df)
        df = df.merge(f[keys + cols], on=keys, how="left")
        assert len(df) == before, f"{name}: 줄 수가 바뀌었어요 (키 중복)"
        print(f"5) {name:<10} 붙임: {cols}  빈칸 비율 " +
            ", ".join(f"{c} {df[c].isna().mean():.1%}" for c in cols))

    # ------------------------------------------------ 5-2단계: 기상상태·노면상태를 Open-Meteo로 통일
    om = openmeteo_weather()
    pos = df[TARGET] == 1
    before = (df.loc[pos, "기상상태"].values.copy(), df.loc[pos, "노면상태"].values.copy())
    df = df.drop(columns=["기상상태", "노면상태"]).merge(om, on=["date", "발생시"], how="left")
    assert df["기상상태"].notna().all(), "Open-Meteo에 없는 시간이 있어요"
    print(f"5-2) 기상상태·노면상태를 Open-Meteo로 통일 (사고 난 줄 중 바뀐 비율: "
          f"기상 {(before[0] != df.loc[pos, '기상상태'].values).mean():.1%}, "
          f"노면 {(before[1] != df.loc[pos, '노면상태'].values).mean():.1%})")

    # ------------------------------------------------ 6단계: 이름 → 번호
    for col, names in CATEGORIES.items():
        unknown = set(df[col].unique()) - set(names)
        assert not unknown, f"{col}에 번호표에 없는 이름: {unknown}"
        df[col] = df[col].map({n: i for i, n in enumerate(names)})
    print("6) 기상상태·노면상태·도로 분류에 번호 붙임")

    # ------------------------------------------------ 7단계: 연도별 60/20/20 나누기
    # 연도 × 심각도(0~3) 묶음마다 나눔 → 예전과 같은 줄이 같은 파일로 감 (팀 data/base와 일치)
    rng = np.random.default_rng(SEED)
    df["split"] = ""
    for _, idx in df.groupby(["연도", "심각도_참고"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        a, b = int(len(idx) * SPLITS["train"]), int(len(idx) * (SPLITS["train"] + SPLITS["valid"]))
        df.loc[idx[:a], "split"] = "train"
        df.loc[idx[a:b], "split"] = "valid"
        df.loc[idx[b:], "split"] = "test"

    keep = FEATURES + [TARGET, "심각도_참고", "연도", "date", "grid_id", "위도", "경도"]   # 뒤 7칸은 모델이 안 봄
    print("7) 각 연도에서 60% / 20% / 20%로 나눔:")
    for split in SPLITS:
        part = df[df["split"] == split][keep]
        part.to_csv(os.path.join(OUT_DIR, f"{split}.csv"), index=False, encoding="utf-8-sig")
        c = part[TARGET].value_counts().sort_index()
        print(f"   {split:<5} {len(part):>7,}줄   사고없음 {c.get(0, 0):,}   사고 {c.get(1, 0):,}")

    info = {"target": TARGET, "labels": {"사고없음": 0, "사고": 1}, "severity_reference": SEVERITY,
            "features": FEATURES, "feature_groups": {"기본": BASE, "1순위": P1, "2순위": P2},
            "categorical": list(CATEGORIES), "category_codes": CATEGORIES, "weekdays": WEEKDAYS,
            "not_features": ["심각도_참고", "연도", "date", "grid_id", "위도", "경도"],
            "cell_size_m": CELL, "history_years": HIST_YEARS, "smooth": SMOOTH,
            "split_rule": "각 연도 × 심각도 묶음 안에서 60/20/20 무작위", "splits": SPLITS, "seed": SEED}
    with open(os.path.join(OUT_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"완료. 변수 {len(FEATURES)}개. 저장 위치: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()