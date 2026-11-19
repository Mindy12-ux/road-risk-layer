"""
3단계 - ④ Drop-test (변수를 '빼고 다시 학습'했을 때 점수 변화, LightGBM, valid로 채점)
목적: permutation은 '섞기'만 해서, 비슷한 변수가 대신 맞혀주면 중요도가 작게 나옴
    → 여기서는 아예 빼고 처음부터 다시 학습해서 "정말 없어도 되나"를 확인
    → 빼도 점수가 그대로(또는 오르면) 빼는 게 이득 (변수가 적을수록 설명·관리가 쉬움)

사용법 (ML project 폴더에서):
    python 상관분석/feat/drop_test.py

입력: model/results/model_input/train.csv (학습), valid.csv (채점), info.json
출력: reports/drop_test/
        drop_test_결과.csv
        01_drop_PR-AUC변화.png        실험별 PR-AUC 변화 (기준 28개 대비)
        02_drop_상위10%포착률변화.png  실험별 상위10% 포착률 변화 (내비게이션 관점)

설정: 기준 = 변수 28개, 가중치 없음, LightGBM (3-②와 같은 설정)
방법: 실험마다 후보 변수를 빼고 다시 학습 → valid 점수 → 기준과의 차이
    씨앗(random_state) 3개로 반복해서 평균 (학습 때 줄·변수를 무작위로 뽑아서 점수가 조금씩 흔들림)
판단: 기준 모델끼리 씨앗만 바꿨을 때 흔들리는 폭(잡음) = 2 × 표준편차
    변화 ≥ -잡음       → 빼도 됨 (점수 차이가 잡음 수준)
    -0.005 < 변화 < -잡음 → 조금 손해
    변화 ≤ -0.005      → 빼면 손해
※ 학습을 실험 수 × 3번 해서 약 5~10분 걸림
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
OUT_DIR = f"{ROOT}/reports/drop_test"
os.makedirs(OUT_DIR, exist_ok=True)

DROP = ["주말", "칸_과거차대차비율", "도로_큰분류"]     # 2단계(다중공선성)에서 정리한 변수
SEEDS = [42, 7, 2024]

#  실험 목록: (실험 묶음, 실험 이름, 뺄 변수) 
P1 = ["칸_과거이륜차비율", "칸_과거보행자비율", "칸_과거노인피해비율", "칸_과거과속신호위반비율",
    "칸_과거자전거PM비율", "칸_과거추돌비율"]
LOW5 = ["박명비율", "칸_교량고가비율", "칸_간선도로비율", "출퇴근", "도로_작은분류"]
EXPERIMENTS = [
    ("① 과거 사고", "- 칸_작년까지큰사고", ["칸_작년까지큰사고"]),
    ("① 과거 사고", "- 칸_작년까지같은시간대사고", ["칸_작년까지같은시간대사고"]),
    ("① 과거 사고", "- 주변_작년까지사고", ["주변_작년까지사고"]),
    ("① 과거 사고", "- 칸_작년까지사고", ["칸_작년까지사고"]),
    ("① 과거 사고", "- 큰사고 + 같은시간대 (작년까지사고·주변만 남김)",
    ["칸_작년까지큰사고", "칸_작년까지같은시간대사고"]),
    ("② 밤/해", "- 밤", ["밤"]),
    ("② 밤/해", "- 태양고도", ["태양고도"]),
    ("② 밤/해", "- 밤 + 박명비율 (태양고도만 남김)", ["밤", "박명비율"]),
    ("③ 1순위", "- 칸_과거과속신호위반비율", ["칸_과거과속신호위반비율"]),
    ("③ 1순위", "- 1순위 6개 전부", P1),
    ("④ 하위 변수", "- 박명비율", ["박명비율"]),
    ("④ 하위 변수", "- 칸_교량고가비율", ["칸_교량고가비율"]),
    ("④ 하위 변수", "- 칸_간선도로비율", ["칸_간선도로비율"]),
    ("④ 하위 변수", "- 출퇴근", ["출퇴근"]),
    ("④ 하위 변수", "- 도로_작은분류", ["도로_작은분류"]),
    ("④ 하위 변수", "- 하위 5개 전부", LOW5),
    ("⑤ 날씨/노면", "- 노면상태", ["노면상태"]),
    ("⑤ 날씨/노면", "- 기상상태", ["기상상태"]),
    ("⑤ 날씨/노면", "- 노면상태 + 기상상태", ["노면상태", "기상상태"]),
    ("⑥ 날짜", "- 월", ["월"]),
]
COLOR = {"① 과거 사고": "#2a78d6", "② 밤/해": "#7b5cd6", "③ 1순위": "#eb6834",
        "④ 하위 변수": "#1baf7a", "⑤ 날씨/노면": "#d93025", "⑥ 날짜": "#898781"}

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
FEATURES = [c for c in info["features"] if c not in DROP]
y_tr, y_va = train[TARGET], valid[TARGET]
for _, _, cols in EXPERIMENTS:
    assert all(c in FEATURES for c in cols), cols
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 기준 변수 {len(FEATURES)}개 / 실험 {len(EXPERIMENTS)}개")


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y.values[top].sum() / y.sum()


def run(features):
    """변수 목록으로 씨앗 3개 학습 → 점수 평균/표준편차"""
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
    df = pd.DataFrame(rows)
    return df.mean(), df.std(), df["PR-AUC"].values


#  2. 기준 모델 (28개) 
t0 = time.time()
base_mean, base_std, base_pr = run(FEATURES)
NOISE = max(2 * base_std["PR-AUC"], 0.0005)
print(f"\n[기준 28개] PR-AUC {base_mean['PR-AUC']:.4f} (씨앗별 {np.round(base_pr, 4)})  "
    f"상위10% {base_mean['상위10%_포착률']:.1%}  ({time.time() - t0:.0f}초)")
print(f"잡음 폭 (씨앗만 바꿨을 때 흔들림 × 2): ±{NOISE:.4f}")

# ── 3. 실험 ─────────────────────────────────────────────
rows = []
for group, name, cols in EXPERIMENTS:
    t0 = time.time()
    mean, std, pr = run([c for c in FEATURES if c not in cols])
    # 씨앗끼리 짝지어 차이 (같은 씨앗 → 무작위 부분이 같아서 비교가 더 공정)
    diff = pr - base_pr
    rows.append({"실험 묶음": group, "실험": name, "뺀 변수 수": len(cols), "남은 변수 수": len(FEATURES) - len(cols),
                "PR-AUC": mean["PR-AUC"], "PR-AUC_변화": diff.mean(), "변화_표준편차": diff.std(ddof=1),
                "ROC-AUC_변화": mean["ROC-AUC"] - base_mean["ROC-AUC"],
                "상위10%포착_변화": mean["상위10%_포착률"] - base_mean["상위10%_포착률"],
                "로그손실_변화": mean["로그손실"] - base_mean["로그손실"]})
    print(f"{name:<40} PR-AUC 변화 {diff.mean():+.4f}  상위10% {rows[-1]['상위10%포착_변화']:+.2%}  "
        f"({time.time() - t0:.0f}초)", flush=True)

result = pd.DataFrame(rows).set_index("실험")


def judge(v):
    if v >= -NOISE:
        return "빼도 됨 (잡음 수준)"
    if v > -0.005:
        return "조금 손해"
    return "빼면 손해"


result["판단"] = result["PR-AUC_변화"].map(judge)
result.round(5).to_csv(f"{OUT_DIR}/drop_test_결과.csv", encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(f"\n[drop-test 결과] (기준 PR-AUC {base_mean['PR-AUC']:.4f}, 잡음 ±{NOISE:.4f})")
print(result.drop(columns=["PR-AUC"]).round(4).to_string())


#  4. 그래프 
def draw(col, err, xlabel, title, path, pct=False):
    plot = result.iloc[::-1]
    vals = plot[col] * (100 if pct else 1)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.barh(plot.index, vals, xerr=None if err is None else plot[err],
            color=[COLOR[g] for g in plot["실험 묶음"]], error_kw={"elinewidth": 0.8, "ecolor": "#333"})
    span = vals.abs().max()
    errs = np.zeros(len(vals)) if err is None else plot[err].values
    for y, (v, e) in enumerate(zip(vals, errs)):
        label = (f"{v:+.2f}%p" if pct else f"{v:+.4f}").replace("−", "-")
        gap = e + span * 0.02                               # 오차 막대 끝 바깥에 숫자 표시
        ax.text(v + gap if v >= 0 else v - gap, y, label, va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    if not pct:
        ax.axvspan(-NOISE, NOISE, color="#898781", alpha=0.15, lw=0)
    ax.set_xlim(-span * 1.25, max(span * 0.35, vals.max() * 1.3))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
    ax.legend(handles, COLOR.keys(), loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


draw("PR-AUC_변화", "변화_표준편차",
    f"valid PR-AUC 변화 (기준 28개 {base_mean['PR-AUC']:.3f} 대비, 회색 띠 = 잡음 ±{NOISE:.4f}, 왼쪽일수록 빼면 손해)",
    "Drop-test: 변수를 빼고 다시 학습하면 점수가 얼마나 바뀌나 (LightGBM, 씨앗 3개 평균)",
    f"{OUT_DIR}/01_drop_PR-AUC변화.png")
draw("상위10%포착_변화", None,
    f"상위10% 포착률 변화 (%p, 기준 {base_mean['상위10%_포착률']:.1%} 대비)",
    "Drop-test: 위험 점수 상위 10% 안에 잡히는 실제 사고 비율 변화",
    f"{OUT_DIR}/02_drop_상위10%포착률변화.png", pct=True)

print(f"\n저장 위치: {OUT_DIR}")

# ── Drop-test 결과 요약 (LightGBM, 씨앗 3개, 기준 28개 valid PR-AUC 0.5105, 잡음 ±0.0005, 기상·노면 Open-Meteo 통일 후) ──
# 어떤 변수를 빼도 PR-AUC 변화가 0.002 이하라 '빼면 크게 손해'인 변수는 없음 (통일 전 노면+기상 -0.020은 출처 흔적이었음).
# 가장 손해가 큰 것도 칸_작년까지사고 -0.0017, 하위 5개 전부 -0.0014, 노면+기상 -0.0012, 기상 -0.0009, 월 -0.0009 정도.
# 큰사고·같은시간대·주변사고, 밤, 태양고도, 1순위 6개 전부, 과속신호위반, 박명·교량고가·출퇴근은 빼도 잡음 수준
#   → 과거 사고 변수는 '칸_작년까지사고' 하나만 남겨도 되고, 1순위 변수는 점수에는 도움이 안 됨.
# 따라서 최종 변수는 점수보다 '내비게이션 이유로 보여줄 가치'(비·밤·보행자 등)를 기준으로 고르면 됨 → 3-5 세트 비교에서 확인.
