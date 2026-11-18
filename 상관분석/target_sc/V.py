"""
상관분석 1단계 - ③ 크라메르 V (Cramér's V)
정답: 사고여부 (0 = 사고없음, 1 = 사고)
대상: 이름(범주)으로 된 변수 5개
    기상상태, 노면상태, 도로_큰분류, 도로_작은분류 + 요일
    (요일은 숫자 0~6이지만 '순서'보다 '어느 요일인가'가 중요한 범주라 여기서도 봅니다)

사용법 (ML project 폴더에서):
    python 상관분석/target_sc/V.py

입력: model/results/model_input/train.csv, info.json   (train만 사용)
출력: reports/target_V/
        크라메르V_결과.csv
        01_크라메르V_막대.png        변수별 V
        02_범주별_사고비율.png       변수마다 범주(예: 맑음/비/눈)별 사고 비율

크라메르 V 계산:
    1) 변수 × 사고여부 표(교차표)를 만들고 카이제곱(χ²) 값을 구함
    χ² = 각 칸의 (실제 줄 수 − 관계가 없을 때 기대되는 줄 수)² ÷ 기대 줄 수 를 모두 더한 값
    2) V = √( χ² ÷ (전체 줄 수 × (min(행 수, 열 수) − 1)) )
    → 0이면 관계 없음, 1이면 완벽한 관계. 방향(+/−)은 없음.

판단 기준 (Cohen, 정답이 2개라 자유도 1 기준): V < 0.1 거의 없음 / 0.1~0.3 약함 / 0.3~0.5 중간 / 0.5 이상 강함
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/target_V"
os.makedirs(OUT_DIR, exist_ok=True)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]

# 번호 → 이름표 (그래프에 '0, 1, 2' 대신 '맑음, 비, 눈'으로 보이게)
NAMES = dict(info["category_codes"])
NAMES["요일"] = ["월", "화", "수", "목", "금", "토", "일"]
CATS = list(info["categorical"]) + ["요일"]
print(f"train {len(train):,}줄 / 크라메르 V 대상 변수 {len(CATS)}개: {CATS}")

#  2. 크라메르 V 계산 
def strength(v):
    return "강함" if v >= 0.5 else "중간" if v >= 0.3 else "약함" if v >= 0.1 else "거의 없음"

rows, rate_tables = [], {}
for col in CATS:
    table = pd.crosstab(train[col], train[TARGET])           # 범주 × 사고여부 표
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    # 범주별 사고 비율 (그래프·해석용)
    rt = pd.DataFrame({"줄 수": table.sum(axis=1), "사고 비율": table[1] / table.sum(axis=1)})
    rt.index = [NAMES[col][i] for i in rt.index]
    rate_tables[col] = rt

    top, low = rt["사고 비율"].idxmax(), rt["사고 비율"].idxmin()
    rows.append({"변수": col, "범주 수": table.shape[0], "카이제곱": chi2, "크라메르_V": v, "판단": strength(v),
                "사고비율 최고 범주": f"{top} ({rt.loc[top, '사고 비율']:.1%})",
                "사고비율 최저 범주": f"{low} ({rt.loc[low, '사고 비율']:.1%})"})
    print(f"\n[{col}] 범주별 줄 수와 사고 비율")
    print(rt.assign(**{"사고 비율": rt["사고 비율"].map("{:.1%}".format)}).to_string())

result = pd.DataFrame(rows).sort_values("크라메르_V", ascending=False).reset_index(drop=True)
result.round(4).to_csv(f"{OUT_DIR}/크라메르V_결과.csv", index=False, encoding="utf-8-sig")
print("\n[크라메르 V 결과] (V 큰 순서)")
print(result.round(4).to_string(index=False))

#  3. 그래프 ① V 막대 
fig, ax = plt.subplots(figsize=(8, 4))
plot = result.iloc[::-1]
ax.barh(plot["변수"], plot["크라메르_V"], color="#2a78d6")
for y, v in enumerate(plot["크라메르_V"]):
    ax.text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9)
for t in (0.1, 0.3):
    ax.axvline(t, color="#898781", ls="--", lw=0.8)
ax.set_xlim(0, max(0.35, result["크라메르_V"].max() * 1.25))
ax.set_title("크라메르 V: 범주 변수 vs 사고여부 (train)", loc="left")
ax.set_xlabel("V  (점선: 0.1 약함, 0.3 중간)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_크라메르V_막대.png", dpi=150)
plt.close()

#  4. 그래프 ② 변수마다 범주별 사고 비율 
base_rate = train[TARGET].mean()
order = result["변수"].tolist()
fig, axes = plt.subplots(len(order), 1, figsize=(11, 3.2 * len(order)))
for ax, col in zip(axes, order):
    rt = rate_tables[col]
    ax.bar(rt.index, rt["사고 비율"], color="#2a78d6")
    for i, (r, cnt) in enumerate(zip(rt["사고 비율"], rt["줄 수"])):
        ax.text(i, r + 0.01, f"{r:.0%}\n({cnt:,}줄)", ha="center", fontsize=7)
    ax.axhline(base_rate, color="#898781", ls="--", lw=0.8)
    v = result.loc[result["변수"] == col, "크라메르_V"].iloc[0]
    ax.set_title(f"{col}  (V = {v:.3f})", loc="left", fontsize=10)
    ax.set_ylabel("사고 비율")
    ax.set_ylim(0, min(1.0, rt["사고 비율"].max() * 1.35))
    ax.tick_params(axis="x", labelsize=8)
fig.suptitle(f"범주별 사고 비율 (점선 = 전체 평균 {base_rate:.1%}, 괄호 = 줄 수)", x=0.01, ha="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_범주별_사고비율.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#train 452,076줄 / 크라메르 V 대상 변수 5개: ['기상상태', '노면상태', '도로_큰분류', '도로_작은분류', '요일']
#
#[기상상태] 범주별 줄 수와 사고 비율
#       줄 수  사고 비율
#맑음  430118  24.5%
#비    20453  26.8%
#눈     1505  24.8%
#
#[노면상태] 범주별 줄 수와 사고 비율
#          줄 수  사고 비율
#건조     408717  24.5%
#젖음/습기   38459  26.4%
#서리/결빙    2825  17.4%
#적설       2075  25.7%
#
#[도로_큰분류] 범주별 줄 수와 사고 비율
#        줄 수  사고 비율
#단일로  238095  23.1%
#교차로  170134  27.1%
#기타    43517  23.0%
#미분류     330  22.4%
#
#[도로_작은분류] 범주별 줄 수와 사고 비율
#              줄 수  사고 비율
#기타         262675  23.7%
#교차로안        93791  26.9%
#교차로부근       58632  27.7%
#교차로횡단보도내    17711  26.5%
#교량           6382  15.2%
#터널           7181   5.4%
#고가도로위        2365  21.9%
#지하차도(도로)내    3009  24.3%
#미분류           330  22.4%
#
#[요일] 범주별 줄 수와 사고 비율
#     줄 수  사고 비율
#월  64749  24.7%
#화  65149  25.1%
#수  65393  25.3%
#목  65145  25.5%
#금  66786  27.2%
#토  64416  24.7%
#일  60438  19.2%
#
#[크라메르 V 결과] (V 큰 순서)
#     변수  범주 수      카이제곱  크라메르_V    판단    사고비율 최고 범주    사고비율 최저 범주
#도로_작은분류     9 2432.1696  0.0733 거의 없음 교차로부근 (27.7%)     터널 (5.4%)
#     요일     7 1261.9537  0.0528 거의 없음     금 (27.2%)     일 (19.2%)
# 도로_큰분류     4  930.7803  0.0454 거의 없음   교차로 (27.1%)   미분류 (22.4%)
#   노면상태     4  152.7189  0.0184 거의 없음 젖음/습기 (26.4%) 서리/결빙 (17.4%)
#   기상상태     3   56.6776  0.0112 거의 없음     비 (26.8%)    맑음 (24.5%)

# ── 크라메르 V 분석 결과 요약 (train, 범주 변수 5개 vs 사고여부 0/1, 기상·노면 Open-Meteo 통일 후) ──
# 5개 모두 V < 0.1로 '관계 거의 없음' 수준.
#   도로_작은분류(0.073) > 요일(0.053) > 도로_큰분류(0.045) > 노면상태(0.018) > 기상상태(0.011) 순.
# 단, V가 작은 이유는 대부분의 줄이 한 범주에 몰려 있기 때문(맑음 95%, 건조 90%)이고,
#   범주별 사고 비율을 보면 의미 있는 차이가 있음.
# 도로_작은분류: 교차로부근 27.7%, 교차로안 26.9% → 교차로 주변이 평균(24.6%)보다 높음.
#   터널 5.4%, 교량 15.2%로 매우 낮음.
# 요일: 금요일 27.2%로 가장 높고 일요일 19.2%로 가장 낮음 → 교통량 차이를 반영한 것으로 해석.
# 기상상태: 비 26.8% vs 맑음 24.5% → 비 오는 날 약 1.1배로 작지만 위험 쪽 방향.
#   눈 24.8%는 맑음과 거의 같음 (눈 오는 날은 차가 덜 다니는 영향이 섞인 것으로 보임).
#   통일 전에는 비 31.7%·눈 31.2%로 크게 나왔지만, 사고 난 줄(경찰 기록)과 안 난 줄(Open-Meteo)의
#   출처 차이 때문에 부풀려진 값이었음 (weather_check.py 참고).
# 노면상태: 젖음/습기 26.4%로 약간 높고, 서리/결빙 17.4%로 낮음, 적설 25.7%는 평균 수준.
#   통일 전 결빙·적설 12%대로 낮게 나왔던 것도 출처 차이 영향이 컸음.
#   → 결빙이 낮은 건 얼은 날 차가 적게 다니는 영향(교통량)으로 보이며 '얼면 안전함'으로 해석하면 안 됨.
# 결론: 날씨·노면은 '비·젖은 노면이면 조금 더 위험' 정도의 약한 신호만 남음.
#   범주 변수는 V만 보고 제외하지 않고, 3단계(모델 중요도·drop-test)에서 재확인.
