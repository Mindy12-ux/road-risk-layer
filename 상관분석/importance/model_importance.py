"""
3단계 - ② 모델별 변수 중요도 (LightGBM / XGBoost / RandomForest)
목적: 서로 다른 모델 3개가 '공통으로' 중요하다고 보는 변수 찾기
    → 한 모델만 중요하다고 하면 그 모델의 버릇일 수 있음. 셋 다 높으면 믿을 만함.

사용법 (ML project 폴더에서):
    python 상관분석/feat/model_importance.py

입력: data/model_input/train.csv (학습), valid.csv (채점), info.json
출력: reports/model_importance/
        모델별_중요도.csv              변수별 중요도(%)와 순위, 평균 순위
        모델별_점수.csv                모델 3개의 valid 점수
        01_모델별_중요도.png           변수별 중요도(%) 세 모델 나란히
        02_순위_히트맵.png             변수별 순위 (세 모델 비교)

설정: 변수 28개 (2단계 정리 반영), 가중치 없음 (3-① 결과)
중요도 계산 (모델마다 '얼마나 많이 도움이 됐나'를 세는 방식):
    LightGBM      gain: 그 변수로 나눴을 때 줄어든 오차의 합
    XGBoost       total_gain: 위와 같은 방식
    RandomForest  불순도 감소: 그 변수로 나눴을 때 사고/사고없음이 얼마나 깔끔하게 갈렸나
    → 단위가 달라서 각 모델 안에서 합계 100%로 맞춘 뒤 비교
※ RandomForest는 느려서 train의 30%만 뽑아 나무마다 학습 (max_samples=0.3)
"""

import json
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
OUT_DIR = f"{ROOT}/reports/model_importance"
os.makedirs(OUT_DIR, exist_ok=True)

DROP = ["주말", "칸_과거차대차비율", "도로_큰분류"]     # 2단계(다중공선성)에서 정리한 변수

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
FEATURES = [c for c in info["features"] if c not in DROP]
CATS = [c for c in info["categorical"] if c in FEATURES]

X_tr, y_tr = train[FEATURES].copy(), train[TARGET]
X_va, y_va = valid[FEATURES].copy(), valid[TARGET]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 변수 {len(FEATURES)}개")


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y.values[top].sum() / y.sum()


scores, importance = [], {}

#  2. LightGBM 
t0 = time.time()
m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                    min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    random_state=42, verbose=-1)
m.fit(X_tr, y_tr, categorical_feature=CATS)
p = m.predict_proba(X_va)[:, 1]
importance["LightGBM"] = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES)
scores.append({"모델": "LightGBM", "PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
            "상위10%_포착률": top_capture(y_va, p), "학습시간(초)": time.time() - t0})
print(f"LightGBM 끝 ({time.time() - t0:.0f}초)  PR-AUC {scores[-1]['PR-AUC']:.4f}", flush=True)

#  3. XGBoost (범주 변수는 category 형식으로 알려줌) 
Xc_tr, Xc_va = X_tr.copy(), X_va.copy()
for c in CATS:
    cats = sorted(X_tr[c].unique())
    Xc_tr[c] = pd.Categorical(X_tr[c], categories=cats)
    Xc_va[c] = pd.Categorical(X_va[c], categories=cats)
t0 = time.time()
m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, min_child_weight=10,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist", enable_categorical=True,
                    eval_metric="logloss", random_state=42, n_jobs=-1)
m.fit(Xc_tr, y_tr)
p = m.predict_proba(Xc_va)[:, 1]
gain = m.get_booster().get_score(importance_type="total_gain")
importance["XGBoost"] = pd.Series({c: gain.get(c, 0.0) for c in FEATURES})
scores.append({"모델": "XGBoost", "PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
            "상위10%_포착률": top_capture(y_va, p), "학습시간(초)": time.time() - t0})
print(f"XGBoost 끝 ({time.time() - t0:.0f}초)  PR-AUC {scores[-1]['PR-AUC']:.4f}", flush=True)

#  4. RandomForest (빈칸 그대로 처리 가능, 범주는 번호 그대로) 
t0 = time.time()
m = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_features="sqrt", max_samples=0.3,
                        n_jobs=-1, random_state=42)
m.fit(X_tr, y_tr)
p = m.predict_proba(X_va)[:, 1]
importance["RandomForest"] = pd.Series(m.feature_importances_, index=FEATURES)
scores.append({"모델": "RandomForest", "PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
            "상위10%_포착률": top_capture(y_va, p), "학습시간(초)": time.time() - t0})
print(f"RandomForest 끝 ({time.time() - t0:.0f}초)  PR-AUC {scores[-1]['PR-AUC']:.4f}", flush=True)

score_df = pd.DataFrame(scores).set_index("모델")
score_df.round(4).to_csv(f"{OUT_DIR}/모델별_점수.csv", encoding="utf-8-sig")
print("\n[모델별 valid 점수]")
print(score_df.round(4).to_string())

#  5. 중요도 정리: 모델마다 합계 100%로 → 순위 → 평균 순위 
MODELS = list(importance)
imp = pd.DataFrame({k: v / v.sum() * 100 for k, v in importance.items()})
rank = imp.rank(ascending=False).astype(int)
table = pd.concat([imp.add_suffix("_중요도(%)"), rank.add_suffix("_순위")], axis=1)
table["평균순위"] = rank.mean(axis=1)
table["세 모델 모두 상위10위"] = (rank <= 10).all(axis=1).map({True: "O", False: ""})
table["세 모델 모두 하위10위"] = (rank > len(FEATURES) - 10).all(axis=1).map({True: "O", False: ""})
table = table.sort_values("평균순위")
table.round(2).to_csv(f"{OUT_DIR}/모델별_중요도.csv", encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n[모델별 중요도] (평균 순위 순)")
print(table.round(2).to_string())

#  6. 그래프 ① 변수별 중요도(%) 세 모델 나란히 
COLOR = {"LightGBM": "#2a78d6", "XGBoost": "#eb6834", "RandomForest": "#1baf7a"}
order = table.index[::-1]
y = np.arange(len(order))
h = 0.27
fig, ax = plt.subplots(figsize=(11, 12))
for k, mname in enumerate(MODELS):
    ax.barh(y + (1 - k) * h, imp.loc[order, mname], h, label=mname, color=COLOR[mname])
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("중요도 (%, 모델마다 합계 100%)")
ax.set_title("모델별 변수 중요도 (위로 갈수록 세 모델 평균 순위가 높음)", loc="left")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_모델별_중요도.png", dpi=150)
plt.close()

#  7. 그래프 ② 순위 히트맵 
r = rank.loc[table.index]
fig, ax = plt.subplots(figsize=(6.5, 12))
im = ax.imshow(r.values, cmap="Blues_r", aspect="auto", vmin=1, vmax=len(FEATURES))
ax.set_xticks(range(len(MODELS)))
ax.set_xticklabels(MODELS)
ax.set_yticks(range(len(r)))
ax.set_yticklabels([f"{name}  (평균 {table.loc[name, '평균순위']:.1f})" for name in r.index], fontsize=8)
for i in range(len(r)):
    for j in range(len(MODELS)):
        v = r.iloc[i, j]
        ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v <= 10 else "black")
ax.set_title("변수별 순위 (1 = 가장 중요, 진할수록 중요)", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_순위_히트맵.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#LightGBM 끝 (4초)  PR-AUC 0.5105
#XGBoost 끝 (4초)  PR-AUC 0.5078
#RandomForest 끝 (7초)  PR-AUC 0.5060

# ── 모델별 중요도 결과 요약 (변수 28개, 가중치 없음, 기상·노면 Open-Meteo 통일 후) ──
# valid PR-AUC: LightGBM 0.511 > XGBoost 0.508 > RandomForest 0.506 → LightGBM을 기준 모델로 사용.
# 세 모델 모두 상위 10위 (8개, 공통으로 중요):
#   칸_작년까지같은시간대사고(평균 1위, 중요도 18~35%), 칸_작년까지사고(2위), 발생시(3.7위), 생활인구_합계(4위),
#   칸_도로연장_m(6.3위), 칸_직전1시간_속도비율(7위), 칸_평소속도_kmh(7.7위), 태양고도(9위).
#   → '원래 사고 나던 곳 + 시간 + 사람 많음 + 도로 규모 + 속도'가 핵심.
# 모델마다 의견이 갈림:
#   칸_작년까지큰사고: LightGBM 4위·RF 4위인데 XGBoost 19위 → 칸_작년까지사고와 겹쳐서(VIF 6.4) 모델이 둘 중 하나만 씀.
#   주변_작년까지사고: RF 6위, 나머지 11~15위 / 밤: RF 13위, 나머지 20~23위 (태양고도·발생시에 정보가 이미 있음).
#   요일: 12~20위, 월: 20~21위 (통일 전 부스팅 모델 9~12위에서 내려감 → 월은 날씨 흔적에 묻어 있던 몫이 빠짐).
# 1순위 변수 6개(이륜차·자전거PM·보행자·노인피해·과속신호위반·추돌 비율): 평균 13.3~18위로 중간 아래, 각 1~2% 수준.
# 세 모델 모두 하위 10위 (7개, 공통으로 덜 중요):
#   기상상태(28위), 노면상태(27위), 박명비율, 출퇴근, 칸_교량고가비율, 칸_간선도로비율, 월.
#   → 기상·노면은 통일 전 13·17위였는데 출처 흔적이 빠지면서 꼴찌로 내려감 (0.02~0.28%).
#   → drop-test에서 빼도 점수가 그대로인지 확인할 후보 (도로_작은분류도 평균 21.7위로 낮음).
# 주의: 이 중요도는 '학습할 때 많이 쓴 정도'라서, 겹치는 변수끼리는 중요도가 나뉘거나 한쪽으로 몰릴 수 있음
#   → permutation(3-③)과 drop-test(3-④)로 실제 점수 기여를 다시 확인.
