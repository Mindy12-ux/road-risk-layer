"""
3단계 - ⑤-① 노면·기상 '흔적' 점검
의심: 사고난 줄과 사고안난 줄의 날씨·노면 출처가 다름
    사고난 줄   → 경찰이 사고 현장에서 적은 기록 (TAAS)
    사고안난 줄 → Open-Meteo 시간별 날씨로 규칙을 정해 만든 값 (scripts/make_negatives.py)
    → 모델이 '진짜 위험'이 아니라 '어느 출처에서 온 줄인지'를 외웠을 수 있음

사용법 (ML project 폴더에서):
    python 상관분석/feat/weather_check.py

입력: data/model_input/train.csv, valid.csv, info.json, data/raw/weather/seoul_hourly_openmeteo.json
출력: 화면 출력만 (파일 저장 없음)

점검 순서:
1) 사고없음 줄: 저장된 날씨·노면 = Open-Meteo 규칙으로 다시 만든 값인지 (출처 확인)
2) 사고 줄: 경찰 기록 vs 같은 날짜·시간의 Open-Meteo 값이 얼마나 일치하나
3) 날씨+노면 조합별 사고 비율: 원래 값 vs 모든 줄을 Open-Meteo로 통일한 값
4) 모델 비교 (LightGBM, 씨앗 3개, valid PR-AUC)
    A 원래 그대로 (출처가 섞인 날씨·노면)
    B 날씨·노면 빼기
    C 모든 줄의 날씨·노면을 Open-Meteo로 통일 (같은 출처)
    → C가 B와 비슷하면: 원래 점수 차이는 '출처 흔적'이었음
    → C가 A와 비슷하면: 날씨·노면이 진짜 위험 신호임
※ 약 1분 걸림
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
WEATHER_PATH = f"{ROOT}/data/raw/weather/seoul_hourly_openmeteo.json"

DROP = ["주말", "칸_과거차대차비율", "도로_큰분류"]     # 2단계(다중공선성)에서 정리한 변수
SEEDS = [42, 7, 2024]
pd.set_option("display.width", 200)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
FEATURES = [c for c in info["features"] if c not in DROP]
SKY_NAMES = info["category_codes"]["기상상태"]       # 번호 → 이름
SURF_NAMES = info["category_codes"]["노면상태"]


#  2. Open-Meteo 규칙 (scripts/make_negatives.py의 load_weather와 똑같이) ──
def load_weather():
    with open(WEATHER_PATH, encoding="utf-8") as f:
        w = pd.DataFrame(json.load(f)["hourly"])
    w["time"] = pd.to_datetime(w["time"])
    sky = np.full(len(w), "맑음", dtype=object)
    sky[w["precipitation"] >= 1.0] = "비"
    sky[(w["snowfall"] > 0) & (w["precipitation"] >= 0.5)] = "눈"
    recent_rain = w["precipitation"].rolling(3, min_periods=1).sum()
    recent_snow = w["snowfall"].rolling(3, min_periods=1).sum()
    recent_12h = w["precipitation"].rolling(12, min_periods=1).sum()
    surface = np.full(len(w), "건조", dtype=object)
    surface[recent_rain >= 1.0] = "젖음/습기"
    surface[(recent_12h >= 1.0) & (w["temperature_2m"] <= 0)] = "서리/결빙"
    surface[recent_snow >= 0.5] = "적설"
    return pd.DataFrame({"om_기상": sky, "om_노면": surface}, index=w["time"])


weather = load_weather()


def add_openmeteo(df):
    """줄마다 (날짜 + 발생시)의 Open-Meteo 날씨·노면을 붙임"""
    when = pd.to_datetime(df["date"]) + pd.to_timedelta(df["발생시"], unit="h")
    om = weather.reindex(when)
    df = df.copy()
    df["om_기상"] = om["om_기상"].values
    df["om_노면"] = om["om_노면"].values
    df["원래_기상"] = df["기상상태"].map(dict(enumerate(SKY_NAMES)))
    df["원래_노면"] = df["노면상태"].map(dict(enumerate(SURF_NAMES)))
    return df


train = add_openmeteo(train)
valid = add_openmeteo(valid)
missing = train["om_기상"].isna().sum() + valid["om_기상"].isna().sum()
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / Open-Meteo 시간이 없는 줄 {missing:,}개")

neg = train[train[TARGET] == 0]
pos = train[train[TARGET] == 1]

#  3. 점검 1: 사고없음 줄은 정말 Open-Meteo 규칙으로 만든 값인가 
same_sky = (neg["원래_기상"] == neg["om_기상"]).mean()
same_surf = (neg["원래_노면"] == neg["om_노면"]).mean()
print("\n[점검 1] 사고없음 줄: 저장된 값 = Open-Meteo 규칙 값?")
print(f"   기상 일치 {same_sky:.1%} / 노면 일치 {same_surf:.1%}  (≈100%면 출처가 Open-Meteo 맞음)")

#  4. 점검 2: 사고 줄은 경찰 기록 vs Open-Meteo가 얼마나 같은가 
print("\n[점검 2] 사고 줄: 경찰 기록 vs 같은 시간 Open-Meteo")
print(f"   기상 일치 {(pos['원래_기상'] == pos['om_기상']).mean():.1%} / "
    f"노면 일치 {(pos['원래_노면'] == pos['om_노면']).mean():.1%}")
print("\n   노면 (행 = 경찰 기록, 열 = Open-Meteo, 줄 수)")
print(pd.crosstab(pos["원래_노면"], pos["om_노면"], margins=True, margins_name="합계").to_string())
print("\n   기상 (행 = 경찰 기록, 열 = Open-Meteo, 줄 수)")
print(pd.crosstab(pos["원래_기상"], pos["om_기상"], margins=True, margins_name="합계").to_string())

#  5. 점검 3: 날씨+노면 조합별 사고 비율 (원래 vs Open-Meteo 통일) 
def combo_table(sky, surf):
    t = pd.DataFrame({"조합": sky + "+" + surf, "사고": train[TARGET]})
    t = t.groupby("조합")["사고"].agg(줄="size", 사고비율="mean")
    return t


orig = combo_table(train["원래_기상"], train["원래_노면"])
unified = combo_table(train["om_기상"], train["om_노면"])
table = orig.add_suffix("_원래").join(unified.add_suffix("_통일"), how="outer").fillna(0)
table = table.sort_values("줄_원래", ascending=False)
table[["줄_원래", "줄_통일"]] = table[["줄_원래", "줄_통일"]].astype(int)
print(f"\n[점검 3] 날씨+노면 조합별 사고 비율 (train 전체 사고 비율 {train[TARGET].mean():.3f})")
print("   원래 = 사고 줄은 경찰 기록, 사고없음 줄은 Open-Meteo (출처 섞임)")
print("   통일 = 모든 줄을 Open-Meteo로 (같은 출처)")
print(table.round(3).to_string())

#  6. 점검 4: 모델 비교 
for df in (train, valid):
    df["om_기상_code"] = df["om_기상"].map({n: i for i, n in enumerate(SKY_NAMES)})
    df["om_노면_code"] = df["om_노면"].map({n: i for i, n in enumerate(SURF_NAMES)})


def score(tr, va, features):
    cats = [c for c in info["categorical"] if c in features]
    out = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                            min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            random_state=seed, verbose=-1)
        m.fit(tr[features], tr[TARGET], categorical_feature=cats)
        out.append(average_precision_score(va[TARGET], m.predict_proba(va[features])[:, 1]))
    return np.array(out)


def unify(df):
    df = df.copy()
    df["기상상태"] = df["om_기상_code"]
    df["노면상태"] = df["om_노면_code"]
    return df


no_ws = [c for c in FEATURES if c not in ("기상상태", "노면상태")]
A = score(train, valid, FEATURES)
B = score(train, valid, no_ws)
C = score(unify(train), unify(valid), FEATURES)
print("\n[점검 4] 모델 비교 (valid PR-AUC, 씨앗 3개 평균)")
for name, s in [("A 원래 그대로 (출처 섞임)", A), ("B 날씨·노면 빼기", B), ("C Open-Meteo로 통일", C)]:
    print(f"   {name:<24} {s.mean():.4f}  (A 대비 {s.mean() - A.mean():+.4f})")
gap = A.mean() - B.mean()
kept = (C.mean() - B.mean()) / gap if gap > 0 else float("nan")
print(f"\n   → 날씨·노면이 주던 점수 {gap:.4f} 중, 같은 출처로 통일해도 남는 몫: {kept:.0%}")
print("     (0%에 가까우면 출처 흔적, 100%에 가까우면 진짜 위험 신호)")

# 노면·기상 점검: 사고안난 줄은 Open-Meteo 규칙 값(100% 일치), 사고난 줄은 경찰 기록이라 출처가 달랐고, 
# 모든 줄을 Open-Meteo로 통일하면 날씨·노면이 주던 점수(PR-AUC +0.020)의 6%만 남음 → 94%는 진짜 위험이 아니라 '출처 흔적'이었음.
# 따라서 날씨·노면을 내비게이션 이유로 쓰려면 전처리에서 모든 줄의 날씨·노면을 같은 시간의 Open-Meteo 값으로 통일해야 함.
# ※ 전처리(scripts/preprocess.py 5-2단계)에서 모든 줄의 기상·노면을 Open-Meteo로 통일함 (2026-11-17).
#   이 파일의 결과는 통일 '전' 데이터 기준이며, 통일 후 데이터로 다시 돌리면 점검 1·2가 100% 일치로 나와서 의미 없음.

# Open-Meteo로 통일하여 전처리 전체 수정 
# 수정 함으로써 날씨 관련 상관분석 전체 수정