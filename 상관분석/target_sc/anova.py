"""
상관분석 1단계 - ④ ANOVA (분산분석) + 에타제곱 η²
정답: 사고여부 (0 = 사고없음, 1 = 사고)
대상: 숫자 변수 24개 (스피어만과 같은 변수)

사용법 (ML project 폴더에서):
    python 상관분석/target_sc/anova.py

입력: model/results/model_input/train.csv, info.json   (train만 사용)
출력: reports/target_anova/
        ANOVA_결과.csv
        01_에타제곱_막대.png          변수별 η²
        02_사고별_분포비교.png        η²가 큰 변수 6개: 사고없음 vs 사고 그룹의 값 분포(상자그림)

ANOVA가 보는 것: "사고 난 그룹과 안 난 그룹의 평균이 얼마나 다른가"
    F   = 그룹 사이 차이 ÷ 그룹 안 흩어짐        (클수록 두 그룹 평균 차이가 뚜렷함)
    η²  = 그룹 사이 차이 ÷ 전체 흩어짐 (0~1)      (변수의 흩어짐 중 '사고 여부'로 설명되는 비율)
    ※ 그룹이 2개라 t-검정과 같은 결과. 줄이 45만 개라 p값은 거의 0 → p값 대신 η² 크기로 판단.

판단 기준 (Cohen): η² < 0.01 거의 없음 / 0.01~0.06 작음 / 0.06~0.14 중간 / 0.14 이상 큼
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/target_anova"
os.makedirs(OUT_DIR, exist_ok=True)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
BINARY = ["주말", "밤", "출퇴근"]
NUMERIC = [c for c in info["features"] if c not in BINARY and c not in info["categorical"]]

groups = info["feature_groups"]
GROUP_OF = {c: "기본" for c in groups["기본"]}
GROUP_OF.update({c: "1순위" for c in groups["1순위"]})
GROUP_OF.update({c: "2순위" for cols in groups["2순위"].values() for c in cols})
COLOR = {"기본": "#2a78d6", "1순위": "#eb6834", "2순위": "#1baf7a"}
print(f"train {len(train):,}줄 / ANOVA 대상 변수 {len(NUMERIC)}개")

#  2. ANOVA + η² 계산 
def strength(e):
    return "큼" if e >= 0.14 else "중간" if e >= 0.06 else "작음" if e >= 0.01 else "거의 없음"

rows = []
for col in NUMERIC:
    d = train[[col, TARGET]].dropna()                  # 빈칸(2020년 과거 기록 등)은 빼고 계산
    g0, g1 = d.loc[d[TARGET] == 0, col], d.loc[d[TARGET] == 1, col]
    f_stat = f_oneway(g0, g1).statistic
    grand = d[col].mean()
    ss_between = len(g0) * (g0.mean() - grand) ** 2 + len(g1) * (g1.mean() - grand) ** 2
    ss_total = ((d[col] - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    rows.append({"변수": col, "묶음": GROUP_OF[col], "F": f_stat, "η²": eta2, "판단": strength(eta2),
                "사고없음_평균": g0.mean(), "사고_평균": g1.mean(),
                "평균 차이(사고-없음)": g1.mean() - g0.mean(), "사용한 줄": len(d)})

result = pd.DataFrame(rows).sort_values("η²", ascending=False).reset_index(drop=True)
result.round(4).to_csv(f"{OUT_DIR}/ANOVA_결과.csv", index=False, encoding="utf-8-sig")
print("\n[ANOVA 결과] (η² 큰 순서)")
print(result.round(4).to_string(index=False))

#  3. 그래프 ① η² 막대 
fig, ax = plt.subplots(figsize=(10, 9))
plot = result.iloc[::-1]
ax.barh(plot["변수"], plot["η²"], color=[COLOR[g] for g in plot["묶음"]])
for y, v in enumerate(plot["η²"]):
    ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
for t in (0.01, 0.06, 0.14):
    ax.axvline(t, color="#898781", ls="--", lw=0.8)
ax.set_xlim(0, max(0.16, result["η²"].max() * 1.2))
ax.set_title("ANOVA 에타제곱 η²: 변수 vs 사고여부 (train)", loc="left")
ax.set_xlabel("η²  (점선: 0.01 작음, 0.06 중간, 0.14 큼)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, COLOR.keys(), loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_에타제곱_막대.png", dpi=150)
plt.close()

#  4. 그래프 ② η²가 큰 변수 6개: 사고없음 vs 사고 분포 
top = result["변수"].head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), top):
    d = train[[col, TARGET]].dropna()
    data = [d.loc[d[TARGET] == 0, col], d.loc[d[TARGET] == 1, col]]
    bp = ax.boxplot(data, tick_labels=["사고없음 (0)", "사고 (1)"], showfliers=False, patch_artist=True,
                    medianprops={"color": "black"})
    for patch, c in zip(bp["boxes"], ["#c3c2bd", COLOR[GROUP_OF[col]]]):
        patch.set_facecolor(c)
    means = [x.mean() for x in data]
    ax.scatter([1, 2], means, color="white", edgecolor="black", zorder=3, label="평균")
    e = result.loc[result["변수"] == col, "η²"].iloc[0]
    ax.set_title(f"{col}  (η² = {e:.3f})", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
fig.suptitle("η²가 큰 변수 6개: 사고없음 vs 사고 그룹의 값 분포 (상자 = 중간 50%, 선 = 중앙값, 점 = 평균)",
            x=0.01, ha="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_사고별_분포비교.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#train 452,076줄 / ANOVA 대상 변수 24개
#
#[ANOVA 결과] (η² 큰 순서)
#           변수  묶음          F     η²    판단   사고없음_평균     사고_평균 평균 차이(사고-없음)  사용한 줄
#칸_작년까지같은시간대사고  기본 60014.9278 0.1376    중간    1.8945    4.3649       2.4704 376090
#     칸_작년까지사고  기본 53920.4056 0.1254    중간   15.1773   29.1882 14.0109 376090
#    칸_작년까지큰사고  기본 45217.5618 0.1073    중간    3.6954    6.6922   2.9968 376090
#    주변_작년까지사고  기본 33888.0519 0.0827    중간  133.3592  197.6167  64.2575 376090
#      생활인구_합계 2순위 31878.5145 0.0659    중간 5045.3339 7722.0631     2676.7291 452076
#     칸_도로연장_m 2순위 21276.5145 0.0449    작음 2895.6034 3663.9742      768.3708 452076
#      칸_최대차로수 2순위 12730.3945 0.0277    작음    3.1484    3.5928 0.4444 446396
#          발생시  기본  9216.6411 0.0200    작음   11.4859   13.6947        2.2088 452076
#   칸_평소속도_kmh 2순위  7535.2612 0.0181    작음   29.9544   26.0535       -3.9009 409345
#         태양고도 2순위  6321.6948 0.0138    작음    0.2303   10.3759       10.1456 452076
# 칸_직전1시간_속도비율 2순위  1980.3834 0.0048 거의 없음    1.0000    0.9854     -0.0146 409355
#  칸_과거자전거PM비율 1순위   809.6891 0.0021 거의 없음    0.0975    0.0902    -0.0073 376090
# 칸_과거과속신호위반비율 1순위   769.9350 0.0020 거의 없음    0.1133    0.1203        0.0070 376090
#     칸_과거추돌비율 1순위   732.7046 0.0019 거의 없음    0.1617    0.1533   -0.0084 376090
#     칸_간선도로비율 2순위   756.9058 0.0017 거의 없음    0.1717    0.1460   -0.0257 446396
#     칸_교량고가비율 2순위   559.1038 0.0013 거의 없음    0.0315    0.0222   -0.0093 446396
#    칸_과거이륜차비율 1순위   384.4738 0.0010 거의 없음    0.2488    0.2569     0.0081 376090
#           요일  기본   264.6343 0.0006 거의 없음    2.9986    2.8873       -0.1113 452076
#   칸_과거노인피해비율 1순위   211.1120 0.0006 거의 없음    0.1309    0.1275     -0.0034 376090
#            월  기본   190.5513 0.0004 거의 없음    6.4975    6.6616        0.1641 452076
#    칸_과거보행자비율 1순위    26.0391 0.0001 거의 없음    0.2173    0.2194     0.0020 376090
#    칸_과거차대차비율 1순위    24.7081 0.0001 거의 없음    0.7526    0.7546     0.0021 376090
# 칸_평균제한속도_kmh 2순위    18.9510 0.0000 거의 없음   42.3604   42.1896   -0.1707 446396
#         박명비율 2순위     5.6038 0.0000 거의 없음    0.0398    0.0388-0.0010 452076

# ── ANOVA η² 분석 결과 요약 (train, 숫자 변수 24개, 사고없음 vs 사고 두 그룹 평균 비교) ──
# 과거 사고 기록 4개가 '중간' 수준으로 가장 큼:
#   칸_작년까지같은시간대사고(0.138), 칸_작년까지사고(0.125), 칸_작년까지큰사고(0.107), 주변_작년까지사고(0.083).
#   → 사고 난 줄은 같은 시간대 과거 사고가 평균 4.4건으로, 사고 안 난 줄(1.9건)의 약 2.3배.
# 생활인구_합계(0.066)도 '중간' → 사고 난 곳 평균 약 7,700명 vs 안 난 곳 약 5,000명 (약 1.5배).
# 칸_도로연장_m(0.045), 칸_최대차로수(0.028), 발생시(0.020), 칸_평소속도_kmh(0.018), 태양고도(0.014)는 '작음'.
#   → 도로가 길고 넓을수록, 늦은 시간일수록 사고 증가. 평소속도는 사고 난 곳이 약 4km/h 더 느림(도심 도로 영향).
# 1순위 변수 7개와 월, 요일, 간선도로비율, 교량고가비율, 평균제한속도, 박명비율 등 나머지 14개는 η² < 0.01로 '거의 없음'.
# 스피어만 결과와 순위가 거의 같음 → 두 기법 모두 '과거 사고 기록 > 생활인구 > 도로 규모' 순으로 중요하다고 일치.
# 1순위 변수는 '사고 종류'를 설명하는 변수라 평균 차이가 작은 게 자연스러움 → 3단계(모델 중요도·drop-test)에서 재확인 예정.