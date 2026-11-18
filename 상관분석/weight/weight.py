"""
3단계 - ① 기준 모델 + 가중치 비교 (LightGBM)
목적: 사고(1) : 사고없음(0) = 1 : 3 불균형을 어떻게 다룰지 정하기
비교: 가중치 없음 / 약한 가중치(√3배) / balanced(3배)  → 사고 줄에 얼마나 더 무게를 줄지

사용법 (ML project 폴더에서):
    python 상관분석/feat/weight.py

입력: model/results/model_input/train.csv (학습), valid.csv (채점), info.json
출력: reports/feat_weight/
        가중치_비교.csv
        01_가중치_점수비교.png        지표별 점수
        02_PR곡선.png                 정밀도-재현율 곡선 (가중치별)

변수: 2단계 정리 반영 → 31개에서 주말, 칸_과거차대차비율, 도로_큰분류를 뺀 28개
평가 (valid):
    PR-AUC              사고를 얼마나 잘 골라내나 (메인 점수, 불균형 데이터용)
    ROC-AUC             사고와 사고없음을 얼마나 잘 구분하나
    상위10%_포착률       위험 점수 상위 10% 안에 실제 사고의 몇 %가 들어가나 (내비게이션 관점)
    로그손실             확률 숫자가 얼마나 정확한가 (낮을수록 좋음)
    재현율 / 정밀도 / F1  '확률 0.5 넘으면 사고'로 판정했을 때
"""

import json
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, log_loss, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score)

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/feat_weight"
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

X_tr, y_tr = train[FEATURES], train[TARGET]
X_va, y_va = valid[FEATURES], valid[TARGET]
n0, n1 = (y_tr == 0).sum(), (y_tr == 1).sum()
print(f"train {len(train):,}줄 (사고없음 {n0:,} / 사고 {n1:,} = 1 : {n0 / n1:.2f})")
print(f"valid {len(valid):,}줄 / 변수 {len(FEATURES)}개 (범주 {CATS})")

#  2. 가중치 3가지 
#   scale_pos_weight = 사고(1) 줄 하나를 사고없음 줄 몇 개만큼 중요하게 볼지
SETTINGS = {
    "가중치 없음": 1.0,
    "약한 가중치": float(np.sqrt(n0 / n1)),
    "balanced": float(n0 / n1),
}
COLOR = {"가중치 없음": "#2a78d6", "약한 가중치": "#eb6834", "balanced": "#1baf7a"}


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y.values[top].sum() / y.sum()


#  3. 학습 + 채점 
rows, probs = [], {}
for name, spw in SETTINGS.items():
    t0 = time.time()
    model = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                            min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            scale_pos_weight=spw, random_state=42, verbose=-1)
    model.fit(X_tr, y_tr, categorical_feature=CATS)
    p = model.predict_proba(X_va)[:, 1]
    pred = (p >= 0.5).astype(int)
    probs[name] = p
    rows.append({"설정": name, "사고 가중치": round(spw, 2),
                "PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
                "상위10%_포착률": top_capture(y_va, p), "로그손실": log_loss(y_va, p),
                "재현율": recall_score(y_va, pred), "정밀도": precision_score(y_va, pred),
                "F1": f1_score(y_va, pred), "사고로 판정한 비율": pred.mean()})
    r = rows[-1]
    print(f"\n[{name}] (사고 가중치 {spw:.2f}, {time.time() - t0:.0f}초)")
    print(f"   PR-AUC {r['PR-AUC']:.4f}   ROC-AUC {r['ROC-AUC']:.4f}   상위10% 포착 {r['상위10%_포착률']:.1%}   "
        f"로그손실 {r['로그손실']:.4f}")
    print(f"   (0.5 기준) 재현율 {r['재현율']:.1%}   정밀도 {r['정밀도']:.1%}   F1 {r['F1']:.4f}")

result = pd.DataFrame(rows).set_index("설정")
result.round(4).to_csv(f"{OUT_DIR}/가중치_비교.csv", encoding="utf-8-sig")
print("\n[가중치 비교]")
print(result.round(4).to_string())
print(f"\n참고: 아무렇게나 찍었을 때 PR-AUC ≈ {y_va.mean():.3f} (valid의 사고 비율), 상위10% 포착률 ≈ 10%")

#  4. 그래프 ① 지표별 점수 
metrics = ["PR-AUC", "ROC-AUC", "상위10%_포착률", "재현율", "정밀도", "F1"]
names = list(SETTINGS)
x = np.arange(len(metrics))
w = 0.26
fig, ax = plt.subplots(figsize=(13, 5))
for k, n in enumerate(names):
    vals = result.loc[n, metrics].values.astype(float)
    ax.bar(x + (k - 1) * w, vals, w, label=n, color=COLOR[n])
    for xi, v in zip(x + (k - 1) * w, vals):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(["PR-AUC\n(메인)", "ROC-AUC", "상위10%\n포착률", "재현율\n(0.5 기준)", "정밀도\n(0.5 기준)", "F1\n(0.5 기준)"])
ax.set_ylim(0, 1.05)
ax.axvline(2.5, color="#898781", ls=":", lw=0.8)
ax.text(1, 1.0, "← 기준선과 상관없는 점수 (순위 매기는 실력)", ha="center", fontsize=8, color="#555")
ax.text(4, 1.0, "'0.5 넘으면 사고' 기준 점수 →", ha="center", fontsize=8, color="#555")
ax.set_title("가중치별 점수 비교 (LightGBM, valid)", loc="left")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_가중치_점수비교.png", dpi=150)
plt.close()

#  5. 그래프 ② PR 곡선 
fig, ax = plt.subplots(figsize=(7, 6))
for n in names:
    prec, rec, _ = precision_recall_curve(y_va, probs[n])
    ax.plot(rec, prec, color=COLOR[n], lw=1.5, label=f"{n} (PR-AUC {result.loc[n, 'PR-AUC']:.3f})")
ax.axhline(y_va.mean(), color="#898781", ls="--", lw=0.8, label=f"아무렇게나 찍을 때 ({y_va.mean():.3f})")
ax.set_xlabel("재현율 (실제 사고 중 찾아낸 비율)")
ax.set_ylabel("정밀도 (사고라고 한 것 중 실제 사고 비율)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("정밀도-재현율 곡선 (valid)", loc="left")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_PR곡선.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#train 452,076줄 (사고없음 340,892 / 사고 111,184 = 1 : 3.07)
#valid 150,695줄 / 변수 28개 (범주 ['기상상태', '노면상태', '도로_작은분류'])
#
#[가중치 없음] (사고 가중치 1.00, 4초)
#   PR-AUC 0.5105   ROC-AUC 0.7847   상위10% 포착 24.2%   로그손실 0.4568
#   (0.5 기준) 재현율 28.4%   정밀도 58.3%   F1 0.3820
#
#[약한 가중치] (사고 가중치 1.75, 4초)
#   PR-AUC 0.5114   ROC-AUC 0.7849   상위10% 포착 24.2%   로그손실 0.4800
#   (0.5 기준) 재현율 55.4%   정밀도 49.9%   F1 0.5252
#
#[balanced] (사고 가중치 3.07, 4초)
#   PR-AUC 0.5113   ROC-AUC 0.7849   상위10% 포착 24.2%   로그손실 0.5533
#   (0.5 기준) 재현율 76.7%   정밀도 42.5%   F1 0.5467
#
#[가중치 비교]
#          사고 가중치  PR-AUC  ROC-AUC  상위10%_포착률    로그손실     재현율     정밀도      F1  사고로 판정한 비율
#설정                                                                                      
#가중치 없음      1.00  0.5105   0.7847     0.2424  0.4568  0.2840  0.5832  0.3820      0.1198
#약한 가중치      1.75  0.5114   0.7849     0.2423  0.4800  0.5541  0.4992  0.5252      0.2730
#balanced    3.07  0.5113   0.7849     0.2423  0.5533  0.7673  0.4247  0.5467      0.4444
#
#참고: 아무렇게나 찍었을 때 PR-AUC ≈ 0.246 (valid의 사고 비율), 상위10% 포착률 ≈ 10%

# ── 가중치 비교 결과 요약 (LightGBM, 변수 28개, valid, 기상·노면 Open-Meteo 통일 후) ──
# PR-AUC 0.5105 / 0.5114 / 0.5113, ROC-AUC 0.785, 상위10% 포착률 24.2%로 세 설정이 거의 같음
#   (차이 0.001 이하 = 씨앗만 바꿔도 생기는 흔들림 수준). 통일 전(0.530)보다 낮아진 건 날씨·노면 '출처 흔적'이 빠졌기 때문.
# 로그손실은 가중치 없음 0.457로 가장 좋고, 가중치를 줄수록 0.480 → 0.553으로 나빠짐 (확률이 부풀려짐).
# 0.5 기준 재현율은 28% → 55% → 77%로 오르지만 정밀도는 58% → 50% → 42%로 떨어짐 → 기준선만 옮겨진 효과.
# 가중치를 줘도 위험한 곳을 골라내는 순위 실력은 같고 확률만 덜 정확해지므로,
# 확률이 가장 정확한 '가중치 없음'을 쓰고 위험 등급 기준선은 나중에 따로 정함.
