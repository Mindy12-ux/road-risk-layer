"""
3단계 - ⑤ 최종 변수 세트 비교 (LightGBM, valid로 채점)
목적: drop-test는 변수를 '하나씩(묶음별로)' 뺀 것 → 여러 개를 '한꺼번에' 뺐을 때도 점수가 유지되나 확인
    → 점수가 같다면 변수가 적고, 내비게이션 이유로 보여주기 좋은 세트를 고름

사용법 (ML project 폴더에서):
    python 상관분석/importance/feature_set.py

입력: model/results/model_input/train.csv (학습), valid.csv (채점), info.json   (기상·노면 Open-Meteo 통일 후)
출력: reports/feature_set/
        세트별_점수.csv
        세트별_변수.csv                 세트마다 어떤 변수가 들어갔는지 (O 표시)
        01_세트별_PR-AUC.png            세트별 PR-AUC (막대 끝 선 = 씨앗 3개 흔들림)
        02_세트별_상위10%포착률.png      세트별 상위10% 포착률 (내비게이션 관점)

비교할 세트:
    A 전부 (28개)       2단계 정리 후 전부 → 기준
    B 점수 중심 (13개)  drop-test에서 '빼도 됨'인 변수 + 기여가 거의 없는 기상·노면을 모두 뺌
    C 설명 중심 (19개)  B + 내비게이션 이유로 보여주기 좋은 변수 6개
                        (비/눈, 노면, 밤, 같은 시간대 과거 사고, 보행자·이륜차 사고가 많던 곳)
    D 핵심 4개          permutation '중요' 4개만 → 나머지 변수들이 다 합쳐서 얼마나 보태는지 확인용
방법: 세트마다 씨앗 3개로 학습 → valid 점수 평균, A와 같은 씨앗끼리 짝지어 차이 계산
판단: A끼리 씨앗만 바꿨을 때 흔들리는 폭(잡음) = 2 × 표준편차 (최소 0.0005)
    A 대비 변화 ≥ -잡음 → A와 같은 수준 / 그보다 떨어지면 → 점수 손해
    A와 같은 수준인 세트 중에서, 설명하기 좋은 세트(C)를 우선 추천
※ 약 1분 걸림
"""

import json
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/feature_set"
os.makedirs(OUT_DIR, exist_ok=True)

DROP = ["주말", "칸_과거차대차비율", "도로_큰분류"]     # 2단계(다중공선성)에서 정리한 변수
SEEDS = [42, 7, 2024]

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
ALL = [c for c in info["features"] if c not in DROP]
y_tr, y_va = train[TARGET], valid[TARGET]

#  2. 세트 정의 
# B: drop-test에서 빼도 점수가 잡음 수준이었던 변수 + 기여가 거의 없는 기상·노면을 뺌
B_REMOVE = ["칸_작년까지큰사고", "칸_작년까지같은시간대사고", "주변_작년까지사고",   # 과거 사고 → 작년까지사고 하나만
            "밤", "박명비율", "출퇴근", "칸_교량고가비율",                           # 태양고도·발생시에 정보가 있음 / 기여 없음
            "칸_과거이륜차비율", "칸_과거보행자비율", "칸_과거노인피해비율",          # 1순위 6개 전부
            "칸_과거과속신호위반비율", "칸_과거자전거PM비율", "칸_과거추돌비율",
            "기상상태", "노면상태"]                                                   # 통일 후 기여 거의 없음
SET_B = [c for c in ALL if c not in B_REMOVE]
# C: B + 내비게이션 이유 문장으로 쓰기 좋은 변수
EXPLAIN = {"기상상태": "비·눈이 와요", "노면상태": "노면이 젖었/얼었어요", "밤": "밤이라 어두워요",
        "칸_작년까지같은시간대사고": "이 시간대에 사고가 잦던 곳이에요",
        "칸_과거보행자비율": "보행자 사고가 많던 곳이에요", "칸_과거이륜차비율": "오토바이 사고가 많던 곳이에요"}
SET_C = [c for c in ALL if c in SET_B or c in EXPLAIN]
SET_D = ["칸_작년까지사고", "발생시", "생활인구_합계", "칸_작년까지같은시간대사고"]

SETS = {"A 전부": ALL, "B 점수 중심": SET_B, "C 설명 중심": SET_C, "D 핵심 4개": SET_D}
COLOR = {"A 전부": "#898781", "B 점수 중심": "#2a78d6", "C 설명 중심": "#1baf7a", "D 핵심 4개": "#eb6834"}
for name, cols in SETS.items():
    assert all(c in ALL for c in cols), name
    print(f"{name} ({len(cols)}개): {cols}")
print(f"\ntrain {len(train):,}줄 / valid {len(valid):,}줄")


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y.values[top].sum() / y.sum()


def run(features):
    """씨앗 3개로 학습 → 씨앗별 점수 표"""
    cats = [c for c in info["categorical"] if c in features]
    rows = []
    for seed in SEEDS:
        m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                            min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            random_state=seed, verbose=-1)
        m.fit(train[features], y_tr, categorical_feature=cats)
        p = m.predict_proba(valid[features])[:, 1]
        rows.append({"PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
                    "상위10%_포착률": top_capture(y_va, p), "로그손실": log_loss(y_va, p)})
    return pd.DataFrame(rows)


#  3. 세트별 학습 + 채점 
scores = {}
for name, cols in SETS.items():
    t0 = time.time()
    scores[name] = run(cols)
    s = scores[name].mean()
    print(f"{name:<10} ({len(cols):>2}개)  PR-AUC {s['PR-AUC']:.4f}  ROC-AUC {s['ROC-AUC']:.4f}  "
        f"상위10% {s['상위10%_포착률']:.1%}  로그손실 {s['로그손실']:.4f}  ({time.time() - t0:.0f}초)", flush=True)

base = scores["A 전부"]
NOISE = max(2 * base["PR-AUC"].std(), 0.0005)
rows = []
for name, cols in SETS.items():
    s = scores[name]
    diff = s["PR-AUC"].values - base["PR-AUC"].values          # 같은 씨앗끼리 짝지어 차이
    rows.append({"세트": name, "변수 수": len(cols),
                "PR-AUC": s["PR-AUC"].mean(), "PR-AUC_표준편차": s["PR-AUC"].std(),
                "A대비_PR-AUC변화": diff.mean(),
                "ROC-AUC": s["ROC-AUC"].mean(), "상위10%_포착률": s["상위10%_포착률"].mean(),
                "A대비_상위10%변화": s["상위10%_포착률"].mean() - base["상위10%_포착률"].mean(),
                "로그손실": s["로그손실"].mean()})
result = pd.DataFrame(rows).set_index("세트")
result["판단"] = ["기준" if n == "A 전부" else ("A와 같은 수준" if v >= -NOISE else "점수 손해")
                for n, v in result["A대비_PR-AUC변화"].items()]
result.round(5).to_csv(f"{OUT_DIR}/세트별_점수.csv", encoding="utf-8-sig")

members = pd.DataFrame({name: ["O" if c in cols else "" for c in ALL] for name, cols in SETS.items()}, index=ALL)
members["이유 문장 (C에서 추가)"] = [EXPLAIN.get(c, "") for c in ALL]
members.to_csv(f"{OUT_DIR}/세트별_변수.csv", encoding="utf-8-sig")

pd.set_option("display.width", 250)
print(f"\n[세트별 점수] (valid, 씨앗 3개 평균, 잡음 ±{NOISE:.4f})")
print(result.round(4).to_string())
print("\n[세트별 변수] (O = 들어감)")
print(members.to_string())
same = [n for n in result.index if result.loc[n, "판단"] == "A와 같은 수준"]
print(f"\nA와 같은 수준인 세트: {same if same else '없음'}")
print(f"참고: 아무렇게나 찍었을 때 PR-AUC ≈ {y_va.mean():.3f}, 상위10% 포착률 ≈ 10%")


#  4. 그래프 
def draw(col, err, ylabel, title, path, pct=False):
    names = list(result.index)
    vals = result[col] * (100 if pct else 1)
    errs = None if err is None else result[err] * (100 if pct else 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, vals, yerr=errs, color=[COLOR[n] for n in names], width=0.6,
        error_kw={"elinewidth": 1, "ecolor": "#333", "capsize": 4})
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * 0.25 + (0.3 if pct else 0.002)
    ax.set_ylim(lo - pad, hi + pad)                              # 차이가 작아서 아래를 잘라 확대
    for x, (n, v) in enumerate(vals.items()):
        label = f"{v:.1f}%" if pct else f"{v:.4f}"
        ax.text(x, v + pad * 0.15, label, ha="center", fontsize=9)
        ax.text(x, lo - pad * 0.85, f"{result.loc[n, '변수 수']}개", ha="center", fontsize=8, color="#555")
    if not pct:
        a = result.loc["A 전부", col]
        ax.axhspan(a - NOISE, a + NOISE, color="#898781", alpha=0.15, lw=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


draw("PR-AUC", "PR-AUC_표준편차", f"valid PR-AUC (회색 띠 = A ± 잡음 {NOISE:.4f})",
    "세트별 PR-AUC (y축 아래를 잘라 확대, 막대 아래 = 변수 수)", f"{OUT_DIR}/01_세트별_PR-AUC.png")
draw("상위10%_포착률", None, "상위10% 포착률 (%)",
    "세트별 상위10% 포착률 (위험 점수 상위 10% 안에 잡힌 실제 사고 비율)",
    f"{OUT_DIR}/02_세트별_상위10%포착률.png", pct=True)

print(f"\n저장 위치: {OUT_DIR}")

# Feature set 결과: C 설명 중심(19개)이 PR-AUC 0.5100으로 전부(28개, 0.5105)와 같은 수준이라, 
# 변수 9개를 줄이면서도 비·밤·보행자 같은 내비게이션 이유 변수를 담은 C를 최종 세트로 선택함.