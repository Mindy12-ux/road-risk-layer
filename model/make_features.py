"""
2단계: 모델에 넣을 변수 19개 만들기 (Flask에서 불러다 쓰는 파일)

사용법 (다른 파일에서):
    from make_features import FeatureMaker
    fm = FeatureMaker()                                   # data/service_tables/ 표 불러오기 (한 번만)
    X = fm.make()                                         # 지금 시각, 서울 전체 칸, Open-Meteo 날씨
    X = fm.make(when="2026-11-18 08:00", weather="비", road_state="조금 막힘")   # 미래 시각 + 직접 고른 날씨
    X = fm.make(cells=["1910_3905", "1911_3905"], speed_ratio={"1910_3905": 0.8})  # 몇 칸만 + 칸별 속도비율
    → grid_id + 변수 19개 표 (RiskModel(MODEL_DIR).predict(X)에 바로 넣으면 됨)

확인용 실행 (road-risk-layer 폴더에서):
    python model/make_features.py
    → test 데이터로 학습 때와 같은 값이 나오는지 검증 + 지금 시각 서울 전체 예측 시간 측정

변수 19개를 어디서 가져오나
월, 요일, 발생시, 밤        : 고른 시각으로 계산 (preprocess.py와 같은 규칙)
태양고도                   : 서울(37.5665, 126.978)에서 그 시각 30분의 태양 높이 (팀 sun.csv와 같은 기준)
기상상태, 노면상태          : Open-Meteo 시간별 날씨 → preprocess.py와 같은 규칙 / 또는 사용자가 고른 날씨
도로·과거 사고 변수          : data/service_tables/칸_정보.csv
생활인구_합계               : data/service_tables/칸_생활인구.csv (칸 × 주말여부 × 시각 평균)
칸_평소속도_kmh             : data/service_tables/칸_평소속도.csv (칸 × 월 × 시각)
칸_직전1시간_속도비율        : speed_ratio(실시간, 3단계) > road_state(도로상황.json) > 기본 1.0
                            평소속도가 없는 칸은 학습 때처럼 빈 값

필요한 패키지: pandas numpy requests
"""

import datetime as dt
import json
import os
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SERVICE_DIR = os.path.join(ROOT, "data", "service_tables")
MODEL_DIR = os.path.join(ROOT, "model", "results", "models")
MODEL_INFO = os.path.join(MODEL_DIR, "모델_정보.json")
WEATHER_FILE = os.path.join(ROOT, "data", "raw", "seoul_hourly_openmeteo.json")   # 검증용 (없으면 건너뜀)

SEOUL = (37.5665, 126.978)                # 태양고도 기준 위치 (서울시청)
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_POINT = (37.5747, 126.96)          # 학습 때 받은 Open-Meteo 날씨와 같은 지점
WEATHER_CACHE_SEC = 30 * 60                # 날씨는 30분마다 새로 받음
WEATHER_RETRY_SEC = 5 * 60                 # 받기에 실패하면 5분 동안은 다시 안 부르고 기본값 사용
SKY = ["맑음", "비", "눈"]                  # 모델_정보.json 번호 순서
SURFACE = ["건조", "젖음/습기", "서리/결빙", "적설"]
USER_WEATHER = {"맑음": ("맑음", "건조"), "비": ("비", "젖음/습기"), "눈": ("눈", "적설")}


# ---------------------------------------------------------------- 시각 → 변수
def sun_elevation(times, lat=SEOUL[0], lon=SEOUL[1]):
    """태양 높이(도). NOAA 계산식. times = 한국 시각 (그 시각 30분을 넣으면 팀 sun.csv 기준과 같음)"""
    t = pd.DatetimeIndex(times) - pd.Timedelta(hours=9)                  # 한국 시각 → 세계 표준시
    T = (np.asarray(t.to_julian_date()) - 2451545) / 36525
    L0 = (280.46646 + T * (36000.76983 + 0.0003032 * T)) % 360
    M = np.radians(357.52911 + T * (35999.05029 - 0.0001537 * T))
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    C = (np.sin(M) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(2 * M) * (0.019993 - 0.000101 * T)
         + np.sin(3 * M) * 0.000289)
    om = np.radians(125.04 - 1934.136 * T)
    lam = np.radians(L0 + C - 0.00569 - 0.00478 * np.sin(om))
    eps = np.radians(23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60 + 0.00256 * np.cos(om))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    y, L0r = np.tan(eps / 2) ** 2, np.radians(L0)
    eqt = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(M) + 4 * e * y * np.sin(M) * np.cos(2 * L0r)
                         - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * M))
    minutes = np.asarray(t.hour * 60 + t.minute + t.second / 60)
    ha = np.radians(((minutes + eqt + 4 * lon) % 1440) / 4 - 180)
    la = np.radians(lat)
    cos_z = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return np.round(90 - np.degrees(np.arccos(np.clip(cos_z, -1, 1))), 2)


def time_features(when):
    """when(한국 시각 Series) → 월, 요일, 발생시, 밤, 주말, 태양고도"""
    when = pd.to_datetime(pd.Series(when)).dt.floor("h")
    out = pd.DataFrame({"월": when.dt.month, "요일": when.dt.day_of_week, "발생시": when.dt.hour})
    out["밤"] = ((out["발생시"] >= 20) | (out["발생시"] <= 5)).astype(int)
    out["주말"] = (out["요일"] >= 5).astype(int)
    out["태양고도"] = sun_elevation(when + pd.Timedelta(minutes=30))
    return out


# ---------------------------------------------------------------- 날씨
def weather_rules(hourly):
    """Open-Meteo 시간별 값 → 기상상태·노면상태 이름 (preprocess.py openmeteo_weather와 같은 규칙)"""
    w = pd.DataFrame(hourly)
    sky = np.full(len(w), "맑음", dtype=object)
    sky[w["precipitation"] >= 1.0] = "비"                                  # 1시간에 1mm 이상
    sky[(w["snowfall"] > 0) & (w["precipitation"] >= 0.5)] = "눈"
    recent_rain = w["precipitation"].rolling(3, min_periods=1).sum()       # 최근 3시간
    recent_snow = w["snowfall"].rolling(3, min_periods=1).sum()
    recent_12h = w["precipitation"].rolling(12, min_periods=1).sum()       # 최근 12시간
    surface = np.full(len(w), "건조", dtype=object)
    surface[recent_rain >= 1.0] = "젖음/습기"
    surface[(recent_12h >= 1.0) & (w["temperature_2m"] <= 0)] = "서리/결빙"
    surface[recent_snow >= 0.5] = "적설"
    return pd.DataFrame({"time": pd.to_datetime(w["time"]), "기상상태": sky, "노면상태": surface}).set_index("time")


class FeatureMaker:
    def __init__(self, service_dir=SERVICE_DIR):
        def read(name, **kw):
            return pd.read_csv(os.path.join(service_dir, name), encoding="utf-8-sig", **kw)

        with open(MODEL_INFO, encoding="utf-8") as f:
            self.features = json.load(f)["features"]
        self.info = read("칸_정보.csv").set_index("grid_id")
        self.cells = self.info.index.tolist()
        self.pop = read("칸_생활인구.csv").set_index(["grid_id", "주말", "발생시"])["생활인구_합계"]
        self.usual = read("칸_평소속도.csv").set_index(["grid_id", "월", "발생시"])["칸_평소속도_kmh"]
        with open(os.path.join(service_dir, "도로상황.json"), encoding="utf-8") as f:
            self.road_states = json.load(f)["값"]
        self._weather = None                     # (받은 시각, 날씨 표)
        self._fail_at = None                     # 마지막으로 날씨 받기에 실패한 시각
        self.weather_source = None

    # ── 날씨 ────────────────────────────────────────────
    def fetch_weather(self):
        """Open-Meteo 어제~3일 뒤 시간별 날씨 (30분 동안은 저장해 둔 값 다시 씀)"""
        if self._weather and time.time() - self._weather[0] < WEATHER_CACHE_SEC:
            return self._weather[1]
        if self._fail_at and time.time() - self._fail_at < WEATHER_RETRY_SEC:   # 방금 실패했으면 잠시 쉬기
            raise ConnectionError("조금 전에 날씨 받기 실패")
        params = {"latitude": WEATHER_POINT[0], "longitude": WEATHER_POINT[1],
                "hourly": "temperature_2m,precipitation,snowfall", "timezone": "Asia/Seoul",
                "past_days": 1, "forecast_days": 4}
        try:
            r = requests.get(WEATHER_URL, params=params, timeout=5)
            r.raise_for_status()
        except Exception:
            self._fail_at = time.time()
            raise
        table = weather_rules(r.json()["hourly"])
        self._weather = (time.time(), table)
        return table

    def weather_at(self, when, weather=None):
        """(기상상태 이름, 노면상태 이름, 출처). weather를 고르면 그 값, 아니면 Open-Meteo, 실패하면 맑음/건조"""
        if weather is not None:
            sky, surface = USER_WEATHER[weather]
            return sky, surface, "사용자 선택"
        try:
            row = self.fetch_weather().loc[pd.Timestamp(when).floor("h")]
            return row["기상상태"], row["노면상태"], "Open-Meteo"
        except Exception as e:                    # 인터넷 문제, 3일 넘는 시각 등
            return "맑음", "건조", f"기본값 (날씨를 못 받음: {type(e).__name__})"

    # ── 변수 만들기 ──────────────────────────────────────
    def build(self, rows):
        """여러 칸·여러 시각을 한꺼번에. rows 칸: grid_id, when, 기상상태, 노면상태(이름), 속도비율(없으면 1.0)"""
        rows = rows.reset_index(drop=True)
        X = time_features(rows["when"])
        X["grid_id"] = rows["grid_id"].values
        X["기상상태"] = rows["기상상태"].map({n: i for i, n in enumerate(SKY)}).values
        X["노면상태"] = rows["노면상태"].map({n: i for i, n in enumerate(SURFACE)}).values

        info = self.info.reindex(X["grid_id"])
        for c in ["도로_작은분류", "칸_작년까지사고", "칸_과거이륜차비율", "칸_과거보행자비율",
                "칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율"]:
            X[c] = info[c].values
        band = (X["발생시"] // 3).values                                 # 시간대 0~7
        band_cols = info[[f"같은시간대사고_{b}" for b in range(8)]].values
        X["칸_작년까지같은시간대사고"] = band_cols[np.arange(len(X)), band]

        X["생활인구_합계"] = self.pop.reindex(pd.MultiIndex.from_arrays(
            [X["grid_id"], X["주말"], X["발생시"]])).values
        X["칸_평소속도_kmh"] = self.usual.reindex(pd.MultiIndex.from_arrays(
            [X["grid_id"], X["월"], X["발생시"]])).values
        ratio = rows["속도비율"].astype(float).fillna(1.0).values if "속도비율" in rows else np.ones(len(X))
        X["칸_직전1시간_속도비율"] = np.where(X["칸_평소속도_kmh"].notna(), ratio, np.nan)   # 속도 없는 칸 = 빈 값
        return X[["grid_id"] + self.features]

    def make(self, when=None, cells=None, weather=None, road_state=None, speed_ratio=None):
        """
        when        : 시각 ("2026-11-18 08:00" 또는 datetime). 없으면 지금
        cells       : 칸 목록. 없으면 서울 전체 2,118칸
        weather     : None = Open-Meteo / "맑음", "비", "눈" = 사용자가 고른 날씨
        road_state  : "원활", "평소", "조금 막힘", "많이 막힘" (미래 시각용)
        speed_ratio : 실시간 속도비율. 숫자 하나(모든 칸) 또는 {grid_id: 비율} (3단계 결과). road_state보다 먼저
        """
        when = pd.Timestamp(when) if when is not None else pd.Timestamp(dt.datetime.now())
        cells = list(cells) if cells is not None else self.cells
        unknown = [c for c in cells if c not in self.info.index]
        assert not unknown, f"서울 칸 목록에 없는 칸: {unknown[:5]}"
        sky, surface, src = self.weather_at(when, weather)
        self.weather_source = src

        ratio = pd.Series(self.road_states[road_state] if road_state else 1.0, index=cells, dtype=float)
        if isinstance(speed_ratio, dict):
            ratio.update(pd.Series(speed_ratio, dtype=float))
        elif speed_ratio is not None:
            ratio[:] = float(speed_ratio)
        rows = pd.DataFrame({"grid_id": cells, "when": when.floor("h"), "기상상태": sky, "노면상태": surface,
                            "속도비율": ratio.values})
        return self.build(rows)


# ---------------------------------------------------------------- 확인용 (python model/make_features.py)
def check():
    import sys
    from sklearn.metrics import average_precision_score, roc_auc_score
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from predict import RiskModel

    fm = FeatureMaker()
    rm = RiskModel(MODEL_DIR)
    test = pd.read_csv(os.path.join(ROOT, "model", "results", "model_input", "test.csv"), encoding="utf-8-sig")
    print(f"1) test {len(test):,}줄로 검증")

    # (1) 시각 변수
    tf = time_features(pd.to_datetime(test["date"]) + pd.to_timedelta(test["발생시"], unit="h"))
    same = {c: (tf[c].values == test[c].values).mean() for c in ["월", "요일", "발생시", "밤"]}
    sun_gap = np.abs(tf["태양고도"].values - test["태양고도"].values)
    print("   시각 변수가 test와 같은 비율: " + ", ".join(f"{c} {v:.1%}" for c, v in same.items()) +
        f" / 태양고도 차이 평균 {sun_gap.mean():.2f}도 (최대 {sun_gap.max():.2f}도)")

    # (2) 날씨 규칙: 학습 때 받은 Open-Meteo 파일에 같은 규칙 → test 값과 비교 (파일이 있을 때만)
    t_when = pd.to_datetime(test["date"]) + pd.to_timedelta(test["발생시"], unit="h")
    if os.path.exists(WEATHER_FILE):
        with open(WEATHER_FILE, encoding="utf-8") as f:
            wx = weather_rules(json.load(f)["hourly"])
        got = wx.reindex(t_when)
        sky_ok = (got["기상상태"].map({n: i for i, n in enumerate(SKY)}).values == test["기상상태"].values).mean()
        surf_ok = (got["노면상태"].map({n: i for i, n in enumerate(SURFACE)}).values == test["노면상태"].values).mean()
        print(f"   날씨 규칙이 test와 같은 비율: 기상상태 {sky_ok:.1%}, 노면상태 {surf_ok:.1%}")
    else:                                        # 날씨 원본이 없으면 test에 적힌 날씨를 그대로 씀
        print("   (날씨 원본 파일이 없어서 날씨 규칙 비교는 건너뜀)")
        got = pd.DataFrame({"기상상태": test["기상상태"].map(dict(enumerate(SKY))).values,
                            "노면상태": test["노면상태"].map(dict(enumerate(SURFACE))).values})

    # (3) 서비스 방식으로 만든 변수로 점수 비교
    #     칸_정보의 과거 사고는 2026년 기준(2023~2025)이라 test 연도와 달라서, 과거 사고 4개만 test 값을 그대로 씀
    rows = pd.DataFrame({"grid_id": test["grid_id"], "when": t_when,
                        "기상상태": got["기상상태"].values, "노면상태": got["노면상태"].values,
                        "속도비율": test["칸_직전1시간_속도비율"].values})
    X = fm.build(rows)
    for c in ["칸_작년까지사고", "칸_작년까지같은시간대사고", "칸_과거이륜차비율", "칸_과거보행자비율"]:
        X[c] = test[c].values
    y = test["사고여부"].values
    p0 = rm.booster.predict(test[fm.features].astype(float))
    p1 = rm.booster.predict(X[fm.features].astype(float))
    print(f"   PR-AUC  test 원래 변수 {average_precision_score(y, p0):.4f} → 서비스 방식 변수 {average_precision_score(y, p1):.4f}")
    print(f"   ROC-AUC test 원래 변수 {roc_auc_score(y, p0):.4f} → 서비스 방식 변수 {roc_auc_score(y, p1):.4f}")
    print("   (차이 = 생활인구를 평균값으로, 도로_작은분류를 칸에서 가장 흔한 값으로 바꾼 영향)")

    # (4) 지금 시각 서울 전체
    Xn = fm.make()                                   # 첫 번째: 날씨 받기 포함
    t0 = time.time()
    Xn = fm.make()                                   # 두 번째: 저장해 둔 날씨 사용 (웹에서 보통 이 경우)
    t1 = time.time()
    res = rm.predict(Xn)
    t2 = time.time()
    grade = pd.Series([r["등급"] for r in res]).value_counts(normalize=True)
    now = pd.Timestamp(dt.datetime.now()).floor("h")
    print(f"2) 지금 시각 {now:%Y-%m-%d %H시}, 날씨 출처: {fm.weather_source}")
    print(f"   서울 {len(Xn):,}칸 변수 만들기 {t1 - t0:.2f}초 + 등급·이유 {t2 - t1:.2f}초")
    print("   등급 비율: " + ", ".join(f"{g} {grade.get(g, 0):.1%}" for g in ["높음", "보통", "낮음"]))
    top = int(np.argmax([r["위험점수"] for r in res]))
    print(f"   가장 위험한 칸 {Xn['grid_id'][top]}: {res[top]['등급']}, 이유 {res[top]['이유'] + res[top]['오늘']}")
    rain = fm.make(when=now + pd.Timedelta(days=1), weather="비", road_state="많이 막힘")
    res_r = rm.predict(rain)
    up = np.mean([r["등급"] != "낮음" for r in res_r]) - np.mean([r["등급"] != "낮음" for r in res])
    print(f"3) 내일 같은 시각 + 비 + 많이 막힘으로 바꾸면: 보통 이상 칸이 {up:+.1%}p 바뀜, 날씨 출처: {fm.weather_source}")


if __name__ == "__main__":
    check()

# 2단계 결과: 시각·날씨 변수가 test와 100% 같고, 
# 서비스 방식 변수로도 PR-AUC 0.516(원래 0.518)이 나옴
# Open-Meteo 실시간 날씨도 잘 받아졌고, 서울 2,118칸 예측이 약 1.2초라 웹에서 바로 쓸 수 있음