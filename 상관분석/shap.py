"""
4단계 - ① SHAP 값 계산 + ② 방향 확인 (LightGBM, 최종 세트 C 19개, valid로 계산)
목적: 줄(칸·시간)마다 '각 변수가 위험 점수를 얼마나 올렸나/내렸나'를 계산
    → 전체적으로 어떤 변수가 위험 이유로 많이 쓰이는지, 방향이 말이 되는지 확인
    → 다음 단계(4-③)에서 위험을 '올린' 변수만 골라 내비게이션 이유 문장으로 바꿀 예정

사용법 (ML project 폴더에서):
    python 상관분석/SHAP/shap_values.py

입력: data/model_input/train.csv (학습), valid.csv (SHAP 계산), info.json
출력: reports/SHAP/
        SHAP_중요도.csv                 변수별 평균 영향 크기, 위험을 올린 줄 비율
        SHAP_방향.csv                   숫자 변수: 값이 클수록 위험 ↑/↓
        SHAP_범주별.csv                 범주·0/1 변수: 범주마다 평균 SHAP
        01_SHAP_중요도.png              변수별 평균 |SHAP| (영향 크기)
        02_SHAP_점그래프.png            줄마다 SHAP (색 = 변수 값이 큰지 작은지)
        03_SHAP_범주별.png              범주별 평균 SHAP (비/맑음, 밤/낮 등)

SHAP 값 뜻: LightGBM 내장 기능(pred_contrib)으로 계산, 단위는 '위험 점수(로그 오즈)'
    한 줄의 위험 점수 = 기본값(전체 평균) + 변수 19개의 SHAP 합
    SHAP > 0 → 그 변수 때문에 위험이 올라감 (이유로 보여줄 후보)
    SHAP < 0 → 그 변수 때문에 위험이 내려감
방향 판단 (숫자 변수): 변수 값과 SHAP의 스피어만 r
    r > +0.3 → 값이 클수록 위험 ↑ / r < -0.3 → 값이 클수록 위험 ↓ / 그 사이 → 일정하지 않음
※ 약 40초 걸림
"""

import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
OUT_DIR = f"{ROOT}/reports/SHAP"
os.makedirs(OUT_DIR, exist_ok=True)

# 3-⑤에서 고른 최종 세트 C (설명 중심 19개)
FEATURES = ["월", "요일", "발생시", "밤", "기상상태", "노면상태", "도로_작은분류",
            "칸_작년까지사고", "칸_작년까지같은시간대사고", "칸_과거이륜차비율", "칸_과거보행자비율",
            "칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율",
            "생활인구_합계", "칸_평소속도_kmh", "칸_직전1시간_속도비율", "태양고도"]
N_DOTS = 20000            # 점그래프에 찍을 줄 수 (전부 찍으면 너무 느림)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
CATS = [c for c in info["categorical"] if c in FEATURES]
WEEKDAY_NAMES = {v: k[0] for k, v in info["weekdays"].items()}           # 0 → 월
NAMES = {c: dict(enumerate(info["category_codes"][c])) for c in CATS}   # 번호 → 이름
NAMES["요일"] = WEEKDAY_NAMES
NAMES["밤"] = {0: "낮 (6~19시)", 1: "밤 (20~5시)"}
GROUPED = CATS + ["요일", "밤"]                                          # 범주별로 보는 변수
NUMERIC = [c for c in FEATURES if c not in GROUPED]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 변수 {len(FEATURES)}개")

#  2. LightGBM 학습 (3단계와 같은 설정) 
model = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        random_state=42, verbose=-1)
model.fit(train[FEATURES], train[TARGET], categorical_feature=CATS)
X = valid[FEATURES]
print(f"valid PR-AUC: {average_precision_score(valid[TARGET], model.predict_proba(X)[:, 1]):.4f}")

#  3. SHAP 계산 
contrib = model.booster_.predict(X, pred_contrib=True)        # 열: 변수 19개 + 마지막 1개 = 기본값
shap = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)
bias = contrib[0, -1]
raw = model.booster_.predict(X, raw_score=True)
assert np.allclose(shap.sum(axis=1) + bias, raw, atol=1e-6), "SHAP 합 ≠ 위험 점수"
print(f"SHAP 계산 끝: 기본값(전체 평균 위험 점수) {bias:.3f}, 확인: 기본값 + SHAP 합 = 모델 위험 점수 (일치)")

#  4. 변수별 중요도 (평균 |SHAP|) 
imp = pd.DataFrame({
    "평균_|SHAP|": shap.abs().mean(),
    "위험_올린_줄비율": (shap > 0).mean(),
    "올릴때_평균": shap[shap > 0].mean(),
    "내릴때_평균": shap[shap < 0].mean(),
}).sort_values("평균_|SHAP|", ascending=False)
imp["순위"] = range(1, len(imp) + 1)
imp["중요도(%)"] = imp["평균_|SHAP|"] / imp["평균_|SHAP|"].sum() * 100
imp.round(4).to_csv(f"{OUT_DIR}/SHAP_중요도.csv", encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("\n[SHAP 중요도] (평균 |SHAP| 큰 순서)")
print(imp.round(4).to_string())

#  5. 방향 확인 ① 숫자 변수: 값이 클수록 위험 ↑ ? 
rows = []
for c in NUMERIC:
    ok = X[c].notna()
    r = spearmanr(X.loc[ok, c], shap.loc[ok, c]).statistic
    q = X.loc[ok, c].quantile([0.1, 0.9])
    low = shap.loc[ok & (X[c] <= q[0.1]), c].mean()
    high = shap.loc[ok & (X[c] >= q[0.9]), c].mean()
    rows.append({"변수": c, "값↔SHAP_스피어만": r,
                "방향": "값이 클수록 위험 ↑" if r > 0.3 else "값이 클수록 위험 ↓" if r < -0.3 else "일정하지 않음",
                "하위10%_평균SHAP": low, "상위10%_평균SHAP": high})
direction = pd.DataFrame(rows).set_index("변수").sort_values("값↔SHAP_스피어만", ascending=False)
direction.round(4).to_csv(f"{OUT_DIR}/SHAP_방향.csv", encoding="utf-8-sig")
print("\n[방향 ① 숫자 변수] (하위10% = 값이 작은 줄, 상위10% = 값이 큰 줄의 평균 SHAP)")
print(direction.round(4).to_string())

#  6. 방향 확인 ② 범주·0/1 변수: 범주마다 평균 SHAP 
rows = []
for c in GROUPED:
    for code, name in NAMES[c].items():
        m = X[c] == code
        if m.sum() == 0:
            continue
        rows.append({"변수": c, "범주": name, "줄 수": int(m.sum()), "평균_SHAP": shap.loc[m, c].mean(),
                    "위험_올린_줄비율": (shap.loc[m, c] > 0).mean()})
by_cat = pd.DataFrame(rows)
by_cat.round(4).to_csv(f"{OUT_DIR}/SHAP_범주별.csv", index=False, encoding="utf-8-sig")
print("\n[방향 ② 범주·0/1 변수] (평균 SHAP > 0 = 그 범주면 위험 ↑)")
print(by_cat.round(4).to_string(index=False))

#  7. 그래프 ① 중요도 막대 
plot = imp.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(plot.index, plot["평균_|SHAP|"], color="#2a78d6")
for y, (v, p) in enumerate(zip(plot["평균_|SHAP|"], plot["중요도(%)"])):
    ax.text(v + plot["평균_|SHAP|"].max() * 0.01, y, f"{v:.3f} ({p:.1f}%)", va="center", fontsize=7)
ax.set_xlabel("평균 |SHAP|  (위험 점수를 평균적으로 얼마나 움직이나, 로그 오즈)")
ax.set_title("SHAP 변수 중요도 (LightGBM, 최종 세트 C 19개, valid)", loc="left")
ax.set_xlim(0, plot["평균_|SHAP|"].max() * 1.2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_SHAP_중요도.png", dpi=150)
plt.close()

#  8. 그래프 ② 점그래프 (줄마다 SHAP, 색 = 변수 값) 
rng = np.random.default_rng(42)
pick = rng.choice(len(X), size=min(N_DOTS, len(X)), replace=False)
order = imp.index[::-1]
fig, ax = plt.subplots(figsize=(10, 8))
for y, c in enumerate(order):
    s = shap[c].values[pick]
    v = X[c].values[pick].astype(float)
    pct = pd.Series(v).rank(pct=True).values                     # 변수 값 → 0~1 (작음~큼)
    jitter = rng.normal(0, 0.12, len(s))
    nan = np.isnan(v)
    ax.scatter(s[nan], y + jitter[nan], c="#cccccc", s=2, alpha=0.4, lw=0)
    sc = ax.scatter(s[~nan], y + jitter[~nan], c=pct[~nan], cmap="coolwarm", vmin=0, vmax=1, s=2, alpha=0.5, lw=0)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=8)
ax.set_xlabel("SHAP (오른쪽 = 위험 ↑, 왼쪽 = 위험 ↓)")
ax.set_title(f"줄마다 SHAP (valid {len(pick):,}줄 무작위, 색: 파랑 = 값 작음 → 빨강 = 값 큼, 회색 = 빈칸)", loc="left",
            fontsize=10)
cb = fig.colorbar(sc, ax=ax, shrink=0.6)
cb.set_label("변수 값 (같은 변수 안에서의 순위)")
cb.set_ticks([0, 1])
cb.set_ticklabels(["작음", "큼"])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_SHAP_점그래프.png", dpi=150)
plt.close()

#  9. 그래프 ③ 범주별 평균 SHAP 
fig, axes = plt.subplots(1, len(GROUPED), figsize=(4 * len(GROUPED), 4.5))
for ax, c in zip(axes, GROUPED):
    d = by_cat[by_cat["변수"] == c]
    colors = ["#d93025" if v > 0 else "#2a78d6" for v in d["평균_SHAP"]]
    ax.barh(d["범주"], d["평균_SHAP"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    span = d["평균_SHAP"].abs().max()
    for y, (v, n) in enumerate(zip(d["평균_SHAP"], d["줄 수"])):
        ax.text(v + (span * 0.05 if v >= 0 else -span * 0.05), y, f"{v:+.3f}\n({n:,}줄)".replace("−", "-"),
                va="center", ha="left" if v >= 0 else "right", fontsize=6.5)
    ax.set_xlim(-span * 1.8, span * 1.8)
    ax.invert_yaxis()
    ax.set_title(c, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
fig.suptitle("범주별 평균 SHAP (빨강 = 위험 ↑, 파랑 = 위험 ↓)", x=0.01, ha="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_SHAP_범주별.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

# SHAP 결과: 위험을 가장 크게 움직이는 건 
# 작년까지사고(21.7%)·발생시(17.9%)·같은시간대사고(12.4%)·생활인구(9.9%)이고, 
# 과거 사고·사람·도로 규모가 클수록, 비·눈·젖은 노면·교차로일수록 위험이 올라가 이유 문장으로 쓸 수 있음.
# 반대로 밤·서리/결빙·보행자비율은 위험을 '내리는' 쪽(차가 적어서 생기는 효과)으로 나와서, 
# 이유 문장은 위험을 올린(SHAP > 0) 변수만 보여줘야 함.