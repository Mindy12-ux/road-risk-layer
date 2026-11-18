"""
상관분석 1단계 - ① 스피어만 순위 상관계수
정답: 사고여부 (0 = 사고없음, 1 = 사고)
대상: 숫자·순서가 있는 변수 24개 (0/1 변수는 파이, 범주 변수는 크라메르 V에서 따로 봅니다)

사용법 (ML project 폴더에서):
    python analysis/target_sc/01_spearman.py

입력: model/results/model_input/train.csv, info.json   (train만 사용 → valid·test는 건드리지 않음)
출력: reports/target_anal_re/spearman/
        스피어만_결과.csv
        01_스피어만_막대.png         변수별 관계 크기 (정렬)
        02_구간별_사고비율.png       관계가 큰 변수 6개: 값이 커질수록 사고 비율이 어떻게 변하나

판단 기준 (Cohen): |r| < 0.1 거의 없음 / 0.1~0.3 약함 / 0.3~0.5 중간 / 0.5 이상 강함
※ 데이터가 45만 줄이라 p값은 거의 항상 0에 가까워서, p값 대신 r의 크기로 판단합니다.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/target_anal_re/spearman"
os.makedirs(OUT_DIR, exist_ok=True)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]

# 스피어만 대상: 0/1 변수(주말·밤·출퇴근)와 범주 변수(기상·노면·도로 분류)는 뺌
BINARY = ["주말", "밤", "출퇴근"]
NUMERIC = [c for c in info["features"] if c not in BINARY and c not in info["categorical"]]

# 그래프 색: 변수가 어느 묶음인지
groups = info["feature_groups"]
GROUP_OF = {c: "기본" for c in groups["기본"]}
GROUP_OF.update({c: "1순위" for c in groups["1순위"]})
GROUP_OF.update({c: "2순위" for cols in groups["2순위"].values() for c in cols})
COLOR = {"기본": "#2a78d6", "1순위": "#eb6834", "2순위": "#1baf7a"}

print(f"train {len(train):,}줄 / 스피어만 대상 변수 {len(NUMERIC)}개")

#  2. 스피어만 계산 
def strength(r):
    a = abs(r)
    return "강함" if a >= 0.5 else "중간" if a >= 0.3 else "약함" if a >= 0.1 else "거의 없음"

rows = []
for col in NUMERIC:
    ok = train[col].notna()                       # 빈칸(2020년 과거 기록 등)은 빼고 계산
    r = spearmanr(train.loc[ok, col], train.loc[ok, TARGET]).statistic
    rows.append({"변수": col, "묶음": GROUP_OF[col], "스피어만_r": r, "|r|": abs(r),
                "판단": strength(r), "방향": "클수록 사고↑" if r > 0 else "클수록 사고↓",
                "사용한 줄": int(ok.sum())})

result = pd.DataFrame(rows).sort_values("|r|", ascending=False).reset_index(drop=True)
result.round(4).to_csv(f"{OUT_DIR}/스피어만_결과.csv", index=False, encoding="utf-8-sig")
print("\n[스피어만 결과] (|r| 큰 순서)")
print(result.drop(columns="|r|").round(4).to_string(index=False))

#  3. 그래프 ① 막대 
fig, ax = plt.subplots(figsize=(10, 9))
plot = result.iloc[::-1]                           # 큰 값이 위로
ax.barh(plot["변수"], plot["스피어만_r"], color=[COLOR[g] for g in plot["묶음"]])
for y, v in enumerate(plot["스피어만_r"]):
    ax.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:.3f}".replace("−", "-"),
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
for t in (0.1, 0.3):                               # Cohen 기준선
    ax.axvline(t, color="#898781", ls="--", lw=0.8)
    ax.axvline(-t, color="#898781", ls="--", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("스피어만 상관계수: 변수 vs 사고여부 (train)", loc="left")
ax.set_xlabel("r  (점선: ±0.1 약함, ±0.3 중간)")
lim = max(0.35, result["|r|"].max() * 1.25)
ax.set_xlim(-lim, lim)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, COLOR.keys(), loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_스피어만_막대.png", dpi=150)
plt.close()

#  4. 그래프 ② 관계가 큰 변수 6개: 구간별 사고 비율 
top = result["변수"].head(6).tolist()
base_rate = train[TARGET].mean()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), top):
    d = train[[col, TARGET]].dropna()
    if d[col].nunique() <= 24:                     # 값 종류가 적으면 그대로 (예: 발생시 0~23)
        g = d.groupby(col)[TARGET].mean()
        x = g.index.astype(str)
    else:                                          # 많으면 5개 구간으로 (값 순서대로 20%씩)
        bins = pd.qcut(d[col].rank(method="first"), 5, labels=["하위20%", "20~40%", "40~60%", "60~80%", "상위20%"])
        g = d.groupby(bins, observed=True)[TARGET].mean()
        x = g.index.astype(str)
    ax.bar(x, g.values, color=COLOR[GROUP_OF[col]])
    ax.axhline(base_rate, color="#898781", ls="--", lw=0.8)
    r = result.loc[result["변수"] == col, "스피어만_r"].iloc[0]
    ax.set_title(f"{col}  (r = {r:.3f})".replace("−", "-"), fontsize=10, loc="left")
    ax.set_ylabel("사고 비율")
    ax.tick_params(axis="x", labelsize=7, rotation=45 if len(x) > 6 else 0)
fig.suptitle(f"관계가 큰 변수 6개: 값에 따른 사고 비율 (점선 = 전체 평균 {base_rate:.1%})", x=0.01, ha="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_구간별_사고비율.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

# ── 스피어만 분석 결과 요약 (train, 숫자 변수 24개 vs 사고여부 0/1) ──
# 과거 사고 기록 변수가 가장 강함: 칸_작년까지같은시간대사고(0.38), 칸_작년까지사고(0.37),
#   칸_작년까지큰사고(0.34)가 '중간' 수준 → 원래 사고가 나던 칸일수록 사고 가능성이 뚜렷하게 높음.
# 주변_작년까지사고(0.29), 생활인구_합계(0.27), 칸_도로연장_m(0.21), 칸_최대차로수(0.16),
#   발생시(0.14), 태양고도(0.12)는 '약한 양의 관계' → 사람이 많고, 도로가 길고 넓고, 늦은 시간일수록 사고 증가.
# 칸_평소속도_kmh(-0.16)만 유일하게 약한 음의 관계 → 평소 빠른 도로(도시고속도로 등)보다
#   신호·보행자가 많아 느린 도심 도로에서 사고가 더 잦은 것으로 해석.
# 1순위 변수 7개(이륜차·보행자·노인·과속신호위반·자전거PM·차대차·추돌 비율)와 월, 요일,
#   간선도로비율, 박명비율 등 나머지 14개는 |r| < 0.1로 '관계 거의 없음'.
# 단, 1순위 변수는 '사고가 날까'보다 '어떤 종류의 사고가 나는 곳인가'를 설명하는 변수라 r이 낮은 게 자연스러움
#   → 바로 제외하지 않고 3단계(모델 중요도·drop-test)에서 재확인 예정.