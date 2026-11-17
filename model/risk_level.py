"""
5단계 - ② 위험 등급 기준선 정하기 (높음 / 보통 / 낮음, valid로 정함)
목적: 모델의 위험 점수를 사용자에게 보여줄 3단계 등급으로 나누는 경계값 정하기

사용법 (ML project 폴더에서, model/tuning.py를 먼저 실행해서 최적_설정.json이 있어야 함):
    python model/risk_level.py

입력: data/model_input/train.csv (학습), valid.csv (기준 정하기), info.json
      eports/tuning/최적_설정.json (5-① 튜닝 결과: 설정 + 나무 개수 + 변수 목록)
출력: reports/risk_level/
        등급_후보비교.csv              후보 A/B/C별 등급마다 줄 비율, 사고 비율, 포착률, 배수
        위험등급_기준.json             고른 후보의 경계 확률값 → 5-③ test, Flask에서 그대로 사용

왜 확률값이 아니라 '순위(상위 몇 %)'로 나누나:
    사고 안 난 데이터를 사고 1건당 3개씩 만들어 넣어서, 모델 확률은 '진짜 사고 확률'이 아니라 '상대 위험 순위'임
    → valid 점수에서 상위 몇 %인지로 경계를 정하고, 그 경계의 확률값을 저장해서 새 예측에 적용
후보:
    A  높음 = 상위 5%,  보통 = 5~20%,  낮음 = 나머지
    B  높음 = 상위 10%, 보통 = 10~30%, 낮음 = 나머지
    C  높음 = 상위 15%, 보통 = 15~40%, 낮음 = 나머지
보는 것 (등급마다):
    사고 비율      그 등급 줄 중 실제 사고 비율 (높음 > 보통 > 낮음이어야 함)
    배수           사고 비율 ÷ 전체 평균 사고 비율 (높음이 평균의 몇 배 위험한가)
    포착률         실제 사고 전체 중 그 등급에 들어간 비율
    칸 수          높음 등급에 들어간 서로 다른 칸 개수 (적으면 경고가 몇몇 칸에만 몰림)
※ 약 30초 걸림
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
TUNING_PATH = f"{ROOT}/reports/tuning/최적_설정.json"
OUT_DIR = f"{ROOT}/reports/risk_level"
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATES = {"A": (0.05, 0.20), "B": (0.10, 0.30), "C": (0.15, 0.40)}   # (높음 비율, 높음+보통 비율)
CHOICE = "B"                     # 결과를 보고 고른 후보 (바꾸고 다시 실행하면 그 후보로 저장)
TIME_BANDS = {"새벽 (0~5시)": range(0, 6), "오전 (6~11시)": range(6, 12),
            "오후 (12~17시)": range(12, 18), "저녁 (18~23시)": range(18, 24)}

#  1. 튜닝 설정 + 데이터 불러오기 
assert os.path.exists(TUNING_PATH), f"{TUNING_PATH} 가 없어요 → model/tuning.py 를 먼저 실행하세요"
with open(TUNING_PATH, encoding="utf-8") as f:
    tuned = json.load(f)
FEATURES, CATS = tuned["features"], tuned["categorical"]
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    TARGET = json.load(f)["target"]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 변수 {len(FEATURES)}개")
print(f"튜닝 설정: {tuned['params']}, 나무 {tuned['n_estimators']}개")

#  2. 튜닝 설정으로 학습 → valid 위험 점수 
model = lgb.LGBMClassifier(objective="binary", n_estimators=tuned["n_estimators"], subsample_freq=1,
                           random_state=42, verbose=-1, **tuned["params"])
model.fit(train[FEATURES], train[TARGET], categorical_feature=CATS)
prob = model.predict_proba(valid[FEATURES])[:, 1]
y = valid[TARGET].values
base_rate = y.mean()
print(f"valid PR-AUC {average_precision_score(y, prob):.4f} / 전체 사고 비율 {base_rate:.1%}")

#  3. 후보별 등급 나누기 
LEVELS = ["높음", "보통", "낮음"]


def assign(p, cut_high, cut_mid):
    return np.where(p >= cut_high, "높음", np.where(p >= cut_mid, "보통", "낮음"))


rows, cuts = [], {}
for name, (hi, mid) in CANDIDATES.items():
    cut_high = float(np.quantile(prob, 1 - hi))          # 이 확률 이상 = 높음
    cut_mid = float(np.quantile(prob, 1 - mid))          # 이 확률 이상 = 보통 (높음 아래)
    cuts[name] = (cut_high, cut_mid)
    level = assign(prob, cut_high, cut_mid)
    for lv in LEVELS:
        m = level == lv
        rows.append({"후보": name, "등급": lv, "경계_확률": cut_high if lv == "높음" else cut_mid if lv == "보통" else 0.0,
                    "줄_비율": m.mean(), "사고_비율": y[m].mean(), "배수(평균 대비)": y[m].mean() / base_rate,
                    "포착률(실제 사고 중)": y[m].sum() / y.sum(), "칸_수": valid.loc[m, "grid_id"].nunique()})
table = pd.DataFrame(rows)
table.round(4).to_csv(f"{OUT_DIR}/등급_후보비교.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("\n[후보별 등급 비교] (valid)")
for name in CANDIDATES:
    t = table[table["후보"] == name].set_index("등급").drop(columns="후보")
    ok = t.loc["높음", "사고_비율"] > t.loc["보통", "사고_비율"] > t.loc["낮음", "사고_비율"]
    hi, mid = CANDIDATES[name]
    print(f"\n후보 {name} (높음 상위 {hi:.0%}, 보통 {hi:.0%}~{mid:.0%})  → 높음 > 보통 > 낮음 순서: {'O' if ok else 'X'}")
    print(t.round(4).to_string())

#  4. 고른 후보 자세히: 높음이 몇몇 칸·시간대에 몰리나 
cut_high, cut_mid = cuts[CHOICE]
level = assign(prob, cut_high, cut_mid)
high = level == "높음"
top_cells = valid.loc[high, "grid_id"].value_counts()
print(f"\n[후보 {CHOICE} 자세히]")
print(f"높음 등급 {high.sum():,}줄, 서로 다른 칸 {len(top_cells):,}개 "
    f"(valid 전체 칸 {valid['grid_id'].nunique():,}개 중 {len(top_cells) / valid['grid_id'].nunique():.1%})")
print(f"높음 줄 중 가장 많이 나온 칸 10개가 차지하는 비율: {top_cells.head(10).sum() / high.sum():.1%}")

band = valid["발생시"].map({h: b for b, hs in TIME_BANDS.items() for h in hs})
by_time = pd.DataFrame({
    "줄 수": band.value_counts(),
    "높음_비율": pd.Series(high).groupby(band.values).mean(),
    "사고_비율": pd.Series(y).groupby(band.values).mean(),
}).reindex(TIME_BANDS)
print("\n시간대별 높음 등급 비율 (사고 안 난 데이터는 24시간 고르게 만들어서, 차가 많은 낮 시간에 높음이 몰릴 수 있음)")
print(by_time.round(3).to_string())

#  5. 고른 후보 저장 
t = table[table["후보"] == CHOICE].set_index("등급")
out = {"후보": CHOICE, "높음_상위비율": CANDIDATES[CHOICE][0], "보통까지_상위비율": CANDIDATES[CHOICE][1],
    "cut_high": round(cut_high, 6), "cut_mid": round(cut_mid, 6),
    "규칙": "확률 ≥ cut_high → 높음 / cut_mid ≤ 확률 < cut_high → 보통 / 나머지 → 낮음",
    "valid_사고비율": {lv: round(float(t.loc[lv, "사고_비율"]), 4) for lv in LEVELS},
    "valid_포착률": {lv: round(float(t.loc[lv, "포착률(실제 사고 중)"]), 4) for lv in LEVELS},
    "model": {"params": tuned["params"], "n_estimators": tuned["n_estimators"], "features": FEATURES}}
with open(f"{OUT_DIR}/위험등급_기준.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n후보 {CHOICE} 저장: 확률 ≥ {cut_high:.4f} → 높음 / ≥ {cut_mid:.4f} → 보통 / 나머지 → 낮음")
print(f"저장 위치: {OUT_DIR}")

# 위험 등급 결과: 후보 B(높음 = 상위 10%, 확률 ≥ 0.518)를 고름. 높음 등급의 사고 비율은 59.8%로 평균(24.6%)의 2.4배이고,
#  실제 사고의 24%를 높음이, 59%를 높음+보통이 잡아냄. 높음은 609개 칸에 고르게 퍼져 있지만,
#  새벽에는 0.5%밖에 안 나와 낮 시간에 몰림