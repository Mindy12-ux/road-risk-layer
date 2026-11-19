"""
상관분석 2단계 - ① 피어슨 상관계수 행렬 + 히트맵 (다중공선성 체크)
목적: 변수 '두 개씩' 짝지어 얼마나 비슷한지 확인 → 너무 비슷한 짝(겹치는 변수) 찾기
대상: 변수 31개 (범주 변수 4개는 원-핫으로 쪼갬 → 계산용 칸 43개)

사용법 (ML project 폴더에서):
    python 상관분석/multi/pearson.py

입력: model/results/model_input/train.csv, info.json   (train만 사용)
출력: reports/multi_pearson/
        피어슨_행렬.csv               모든 짝의 r
        피어슨_높은짝.csv             |r| ≥ 0.5인 짝 목록 (큰 순서)
        01_피어슨_히트맵.png          전체 행렬 (|r| ≥ 0.5만 숫자 표시)
        02_높은짝_막대.png            |r| 큰 짝 상위 20개

원-핫: 이름으로 된 변수는 범주마다 0/1 칸을 만듦 (예: 기상상태 → 기상상태=비, 기상상태=눈)
    첫 번째 범주(맑음, 건조, 단일로, 기타)는 기준이라 뺌 → 칸끼리 자동으로 겹치는 것 방지

판단 기준: |r| < 0.7 괜찮음 / 0.7~0.9 주의 / 0.9 이상 심각 (둘 중 하나는 빼는 것을 검토)
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/multi_pearson"
os.makedirs(OUT_DIR, exist_ok=True)

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
FEATURES, CATS, CODES = info["features"], info["categorical"], info["category_codes"]

#  2. 범주 변수 원-핫 (첫 범주 제외) 
X = train[[c for c in FEATURES if c not in CATS]].astype(float)
for col in CATS:
    for code, name in enumerate(CODES[col]):
        if code == 0:                                   # 기준 범주는 뺌
            continue
        X[f"{col}={name}"] = (train[col] == code).astype(float)
print(f"train {len(train):,}줄 / 변수 {len(FEATURES)}개 → 원-핫 후 계산용 칸 {X.shape[1]}개")

#  3. 피어슨 행렬 
corr = X.corr(method="pearson")                         # 빈칸은 그 짝에서만 빼고 계산
corr.round(3).to_csv(f"{OUT_DIR}/피어슨_행렬.csv", encoding="utf-8-sig")

# 짝 목록 (같은 변수에서 나온 원-핫 칸끼리는 뺌: 예) 노면상태=젖음 ↔ 노면상태=결빙)
def origin(c):
    return c.split("=")[0]

def level(r):
    a = abs(r)
    return "심각" if a >= 0.9 else "주의" if a >= 0.7 else "참고"

pairs = []
cols = corr.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        a, b = cols[i], cols[j]
        if origin(a) == origin(b):
            continue
        r = corr.iloc[i, j]
        if abs(r) >= 0.5:
            pairs.append({"변수1": a, "변수2": b, "피어슨_r": r, "|r|": abs(r), "판단": level(r)})
pairs = pd.DataFrame(pairs).sort_values("|r|", ascending=False).reset_index(drop=True)
pairs.round(4).to_csv(f"{OUT_DIR}/피어슨_높은짝.csv", index=False, encoding="utf-8-sig")
print("\n[|r| ≥ 0.5인 짝] (큰 순서)")
print(pairs.drop(columns="|r|").round(3).to_string(index=False))
print(f"\n심각(≥0.9) {sum(pairs['판단'] == '심각')}쌍 / 주의(0.7~0.9) {sum(pairs['판단'] == '주의')}쌍 / "
    f"참고(0.5~0.7) {sum(pairs['판단'] == '참고')}쌍")

#  4. 그래프 ① 히트맵 (아래 삼각형만) 
n = len(cols)
mask = np.triu(np.ones((n, n), dtype=bool))             # 위 삼각형은 같은 값이라 가림
show = np.ma.masked_array(corr.values, mask=mask)
fig, ax = plt.subplots(figsize=(18, 16))
im = ax.imshow(show, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(cols, rotation=90, fontsize=7)
ax.set_yticklabels(cols, fontsize=7)
for i in range(n):
    for j in range(i):
        r = corr.iloc[i, j]
        if abs(r) >= 0.5:
            ax.text(j, i, f"{r:.2f}".replace("−", "-"), ha="center", va="center", fontsize=5.5,
                    color="white" if abs(r) >= 0.7 else "black")
fig.colorbar(im, ax=ax, shrink=0.6, label="피어슨 r  (빨강 = 같이 커짐, 파랑 = 반대로 움직임)")
ax.set_title("피어슨 상관계수 히트맵 (train, |r| ≥ 0.5만 숫자 표시)", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_피어슨_히트맵.png", dpi=150)
plt.close()

#  5. 그래프 ② |r| 큰 짝 상위 20개 
top = pairs.head(20).iloc[::-1]
color = {"심각": "#d93025", "주의": "#eb6834", "참고": "#898781"}
fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(top) + 1)))
ax.barh([f"{a}  ↔  {b}" for a, b in zip(top["변수1"], top["변수2"])], top["|r|"],
        color=[color[l] for l in top["판단"]])
for y, (v, r) in enumerate(zip(top["|r|"], top["피어슨_r"])):
    ax.text(v + 0.005, y, f"{r:.2f}".replace("−", "-"), va="center", fontsize=8)
for t in (0.7, 0.9):
    ax.axvline(t, color="#898781", ls="--", lw=0.8)
ax.set_xlim(0, 1.05)
ax.set_xlabel("|r|  (점선: 0.7 주의, 0.9 심각 / 막대 옆 숫자 = 실제 r, 음수면 반대로 움직임)")
ax.set_title("너무 비슷한 짝 상위 20개", loc="left")
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_높은짝_막대.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#train 452,076줄 / 변수 31개 → 원-핫 후 계산용 칸 43개
#
#[|r| ≥ 0.5인 짝] (큰 순서)
#          변수1           변수2  피어슨_r 판단
#   도로_큰분류=미분류   도로_작은분류=미분류  1.000 심각
#    칸_과거보행자비율     칸_과거차대차비율 -0.954 심각
#     칸_작년까지사고     칸_작년까지큰사고  0.890 주의
#     칸_작년까지사고 칸_작년까지같은시간대사고  0.814 주의
#            밤          태양고도 -0.809 주의
#           요일            주말  0.783 주의
#     칸_작년까지사고     주변_작년까지사고  0.752 주의
#    칸_작년까지큰사고 칸_작년까지같은시간대사고  0.724 주의
#       기상상태=비    노면상태=젖음/습기  0.711 주의
#    칸_작년까지큰사고     주변_작년까지사고  0.660 참고
#   도로_큰분류=교차로  도로_작은분류=교차로안  0.659 참고
# 칸_평균제한속도_kmh      칸_간선도로비율  0.656 참고
#       기상상태=눈       노면상태=적설  0.654 참고
#칸_작년까지같은시간대사고     주변_작년까지사고  0.612 참고
#    주변_작년까지사고       생활인구_합계  0.611 참고
#     칸_작년까지사고       생활인구_합계  0.578 참고
#    칸_작년까지큰사고       생활인구_합계  0.563 참고
#    칸_과거차대차비율      칸_과거추돌비율  0.530 참고
#    칸_과거보행자비율      칸_과거추돌비율 -0.525 참고
#칸_작년까지같은시간대사고       생활인구_합계  0.524 참고
#
#심각(≥0.9) 2쌍 / 주의(0.7~0.9) 7쌍 / 참고(0.5~0.7) 11쌍

# ── 피어슨 상관계수 분석 결과 요약 (train, 변수 31개 → 원-핫 후 43칸, |r| ≥ 0.5인 짝 20쌍, 기상·노면 Open-Meteo 통일 후) ──
# 심각(|r| ≥ 0.9) 2쌍:
#   도로_큰분류=미분류 ↔ 도로_작은분류=미분류 (1.00): 같은 줄(330줄)이라 완전히 같은 정보
#     → 둘 중 하나의 '미분류'는 중복. 예전 분석에서도 삭제 후보였음.
#   칸_과거보행자비율 ↔ 칸_과거차대차비율 (-0.95): 차대차 + 차대사람 + 차량단독 = 100%라 거의 반대 정보
#     → 1단계 관계도 둘 다 약함(0.014 vs 0.003). 설명하기 쉬운 '보행자'를 남기고 차대차 제외 검토.
# 주의(0.7~0.9) 7쌍:
#   과거 사고 기록 4개끼리 서로 0.72~0.89로 겹침 (칸_작년까지사고 ↔ 큰사고 0.89, ↔ 같은시간대사고 0.81, ↔ 주변 0.75 등)
#     → 모두 1단계에서 가장 강한 변수라 바로 빼지 않고 VIF와 3단계(drop-test)에서 결정.
#   밤 ↔ 태양고도 (-0.81): 둘 다 '어두운가'를 나타냄 → 하나만 남기는 것 검토 (밤이 이유 문장으로 쓰기 쉬움).
#   요일 ↔ 주말 (0.78): 주말 = 토·일요일이라 요일 안에 포함된 정보 → 주말 제외 검토 (예전 분석과 같은 결론).
#   기상상태=비 ↔ 노면상태=젖음/습기 (0.71, 통일 전 0.75): 비가 오면 노면이 젖는 자연스러운 관계
#     → 둘 다 같은 Open-Meteo 규칙에서 나와서 겹치지만, 내비게이션 이유('비가 와요', '노면이 젖었어요')로 쓸 수 있어 VIF 결과 보고 판단.
# 참고(0.5~0.7) 11쌍: 생활인구_합계 ↔ 과거 사고 기록(0.52~0.61), 평균제한속도 ↔ 간선도로비율(0.66),
#   기상상태=눈 ↔ 노면상태=적설(0.65), 차대차/보행자 ↔ 추돌비율(±0.53) 등 → 지금은 유지.
# 기상·노면 통일 전후로 짝 목록과 개수(심각 2 / 주의 7 / 참고 11)는 같고, 비↔젖음·눈↔적설 값만 조금 바뀜.
# 결론: 삭제 후보 = 도로 '미분류' 중복, 칸_과거차대차비율, 주말, (밤 또는 태양고도 중 하나)
#   → VIF로 다시 확인하고, 최종 결정은 3단계 모델 비교 후.
