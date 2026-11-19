"""
상관분석 1단계 - ② 파이 계수 (φ)
정답: 사고여부 (0 = 사고없음, 1 = 사고)
대상: 0/1 변수 3개 (주말, 밤, 출퇴근)

사용법 (ML project 폴더에서):
    python 상관분석/target_sc/phi.py

입력: data/model_input/train.csv, info.json   (train만 사용)
출력: reports/target_phi/
        파이_결과.csv
        01_파이_막대.png          변수별 φ
        02_0과1_사고비율.png      변수가 0일 때와 1일 때 사고 비율 비교

파이 계산 (2×2 표):
                사고없음(0)  사고(1)
    변수 = 0         a          b
    변수 = 1         c          d
    φ = (a·d − b·c) / √((a+b)(c+d)(a+c)(b+d))
    +면 '변수가 1일 때 사고가 더 많음', −면 '변수가 1일 때 사고가 더 적음'

판단 기준 (Cohen): |φ| < 0.1 거의 없음 / 0.1~0.3 약함 / 0.3~0.5 중간 / 0.5 이상 강함
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
OUT_DIR = f"{ROOT}/reports/target_phi"
os.makedirs(OUT_DIR, exist_ok=True)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
BINARY = ["주말", "밤", "출퇴근"]
MEANING = {"주말": "토·일요일", "밤": "20시~5시", "출퇴근": "7~9시, 17~19시"}
print(f"train {len(train):,}줄 / 파이 대상 변수 {len(BINARY)}개")

#  2. 파이 계산 
def strength(v):
    a = abs(v)
    return "강함" if a >= 0.5 else "중간" if a >= 0.3 else "약함" if a >= 0.1 else "거의 없음"

rows = []
for col in BINARY:
    t = pd.crosstab(train[col], train[TARGET])            # 2×2 표
    a, b = t.loc[0, 0], t.loc[0, 1]
    c, d = t.loc[1, 0], t.loc[1, 1]
    phi = (a * d - b * c) / np.sqrt(float(a + b) * (c + d) * (a + c) * (b + d))
    rows.append({"변수": col, "뜻": MEANING[col], "파이_φ": phi, "|φ|": abs(phi), "판단": strength(phi),
                "0일때_사고비율": b / (a + b), "1일때_사고비율": d / (c + d),
                "1인 줄 비율": (c + d) / len(train)})
    print(f"\n[{col}] 2×2 표 (행: {col} 0/1, 열: 사고없음/사고)")
    print(t.to_string())

result = pd.DataFrame(rows).sort_values("|φ|", ascending=False).reset_index(drop=True)
result.round(4).to_csv(f"{OUT_DIR}/파이_결과.csv", index=False, encoding="utf-8-sig")
print("\n[파이 결과] (|φ| 큰 순서)")
print(result.drop(columns="|φ|").round(4).to_string(index=False))

#  3. 그래프 ① 파이 막대 
fig, ax = plt.subplots(figsize=(8, 3.5))
plot = result.iloc[::-1]
ax.barh(plot["변수"], plot["파이_φ"], color=["#2a78d6" if v >= 0 else "#eb6834" for v in plot["파이_φ"]])
for y, v in enumerate(plot["파이_φ"]):
    ax.text(v + (0.003 if v >= 0 else -0.003), y, f"{v:.3f}".replace("−", "-"),
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
for t in (0.1, 0.3):
    ax.axvline(t, color="#898781", ls="--", lw=0.8)
    ax.axvline(-t, color="#898781", ls="--", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-0.35, 0.35)
ax.set_title("파이 계수 φ: 0/1 변수 vs 사고여부 (train)", loc="left")
ax.set_xlabel("φ  (점선: ±0.1 약함, ±0.3 중간 / 파랑 = 1일 때 사고↑, 주황 = 1일 때 사고↓)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_파이_막대.png", dpi=150)
plt.close()

#  4. 그래프 ② 0일 때 vs 1일 때 사고 비율 
base_rate = train[TARGET].mean()
fig, axes = plt.subplots(1, len(BINARY), figsize=(12, 4), sharey=True)
for ax, (_, r) in zip(axes, result.iterrows()):
    vals = [r["0일때_사고비율"], r["1일때_사고비율"]]
    ax.bar([f"아님 (0)", f"{r['변수']} (1)"], vals, color=["#c3c2bd", "#2a78d6"], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=10)
    ax.axhline(base_rate, color="#898781", ls="--", lw=0.8)
    ax.set_title(f"{r['변수']} ({r['뜻']})  φ = {r['파이_φ']:.3f}".replace("−", "-"), fontsize=10, loc="left")
axes[0].set_ylabel("사고 비율")
axes[0].set_ylim(0, max(result[["0일때_사고비율", "1일때_사고비율"]].max()) * 1.25)
fig.suptitle(f"변수가 0일 때 vs 1일 때 사고 비율 (점선 = 전체 평균 {base_rate:.1%})", x=0.01, ha="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_0과1_사고비율.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")
#
#train 452,076줄 / 파이 대상 변수 3개
#
#[주말] 2×2 표 (행: 주말 0/1, 열: 사고없음/사고)
#사고여부       0      1
#주말                 
#0     243531  83691
#1      97361  27493
#
#[밤] 2×2 표 (행: 밤 0/1, 열: 사고없음/사고)
#사고여부       0      1
#밤                  
#0     198494  81957
#1     142398  29227
#
#[출퇴근] 2×2 표 (행: 출퇴근 0/1, 열: 사고없음/사고)
#사고여부       0      1
#출퇴근                
#0     256034  74775
#1      84858  36409
#
#[파이 결과] (|φ| 큰 순서)
# 변수            뜻    파이_φ    판단  0일때_사고비율  1일때_사고비율  1인 줄비율
#  밤       20시~5시 -0.1374    약함    0.2922    0.1703   0.3796
#출퇴근 7~9시, 17~19시  0.0763 거의 없음    0.2260    0.3002   0.2682
# 주말        토·일요일 -0.0369 거의 없음    0.2558    0.2202   0.2762
#
#저장 위치: /Users/woojin/ML project/reports/target_phi

# ── 파이 φ 분석 결과 요약 (train, 0/1 변수 3개 vs 사고여부 0/1) ──
# 밤(20시~5시)만 φ = -0.137로 '약한 관계' → 밤이 아닐 때 사고 비율 29.2%, 밤일 때 17.0%로 밤에 더 낮음.
# 출퇴근(φ = 0.076): 출퇴근 시간 사고 비율 30.0% vs 평소 22.6%로 높지만 '거의 없음' 수준.
# 주말(φ = -0.037): 주말 22.0% vs 평일 25.6%로 차이가 작아 '거의 없음'.
# 주의: 밤의 음수 값은 '밤이 더 안전함'이 아님.
#   사고 안 난 데이터는 24시간에 고르게 무작위로 만들었는데, 실제 사고는 차가 적은 밤에 덜 남
#   → 이 값은 '밤에 다니는 차가 적음(교통량 차이)'을 반영한 것으로 해석.
#   출퇴근의 양수 값도 교통량이 많은 영향이 섞여 있음.
# 서비스 관점: 모델이 그대로 배우면 '밤이라 안전함' 같은 잘못된 이유를 보여줄 수 있음
#   → SHAP으로 이유 문장을 만들 때 '밤' 변수 표현 방식을 다시 검토할 예정.