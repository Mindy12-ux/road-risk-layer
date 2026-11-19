"""
상관분석 2단계 - ② VIF (분산팽창계수, 다중공선성 체크)
목적: 변수 '하나'를 나머지 '전부'로 얼마나 맞힐 수 있는지 확인 → 다른 변수들과 정보가 겹치는 정도
대상: 변수 31개 (범주 변수 4개는 원-핫으로 쪼갬 → 계산용 칸 43개, 피어슨과 같은 방식)

사용법 (ML project 폴더에서):
    python 상관분석/multi/VIF.py

입력: data/model_input/train.csv, info.json   (train만 사용)
출력: reports/multi_VIF/
        VIF_결과.csv                  정리 전 / 정리 후 VIF
        01_VIF_전체.png               43칸 전체 VIF
        02_VIF_정리후.png             피어슨 삭제 후보를 뺀 뒤 다시 계산한 VIF

VIF 계산: 칸 하나를 나머지 칸 전부로 맞히는 직선(회귀)을 만들고, 맞힌 정도 R²를 구함
    VIF = 1 ÷ (1 − R²)
    예) R² = 0.9 (나머지로 90% 맞힘) → VIF = 10  →  이 변수 정보의 90%가 다른 변수에 이미 있음

판단 기준: VIF < 5 괜찮음 / 5~10 주의 / 10 이상 심각

※ 사용하는 줄: 2021년 이후 (2020년은 과거 기록 11개가 전부 빈칸이라 제외)
남은 빈칸(통행속도 약 9.5%, 도로구조 약 1.2%)은 그 칸의 평균으로 채워서 계산
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
OUT_DIR = f"{ROOT}/reports/multi_VIF"
os.makedirs(OUT_DIR, exist_ok=True)

# 피어슨 결과에서 나온 삭제 후보 (정리 후 VIF 계산 때 뺌)
DROP = ["도로_작은분류=미분류",      # 도로_큰분류=미분류와 완전히 같음 (r = 1.00)
        "도로_큰분류=교차로",         # = 교차로안 + 교차로부근 + 교차로횡단보도내 (작은분류 3칸의 합과 완전히 같음)
        "칸_과거차대차비율",          # 칸_과거보행자비율과 거의 반대 (r = -0.95)
        "주말"]                       # 요일에 포함된 정보 (r = 0.78)

#  1. 데이터 불러오기 + 원-핫 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
FEATURES, CATS, CODES = info["features"], info["categorical"], info["category_codes"]

train = train[train["연도"] > 2020].reset_index(drop=True)
X = train[[c for c in FEATURES if c not in CATS]].astype(float)
for col in CATS:
    for code, name in enumerate(CODES[col]):
        if code == 0:
            continue
        X[f"{col}={name}"] = (train[col] == code).astype(float)
X = X.fillna(X.mean())
print(f"사용한 줄 {len(X):,} (2021년 이후) / 계산용 칸 {X.shape[1]}개")


#  2. VIF 계산 함수 
def vif_table(X):
    Z = ((X - X.mean()) / X.std()).values           # 단위를 맞춤 (표준화)
    out = {}
    for i, col in enumerate(X.columns):
        y = Z[:, i]
        A = np.c_[np.ones(len(Z)), np.delete(Z, i, axis=1)]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        out[col] = 1 / max(1 - r2, 1e-6)              # R²가 1이면 무한대 → 아주 큰 값으로
    return pd.Series(out)


def level(v):
    return "심각" if v >= 10 else "주의" if v >= 5 else "괜찮음"


before = vif_table(X)
after = vif_table(X.drop(columns=DROP))

result = pd.DataFrame({"VIF_정리전": before, "VIF_정리후": after})
result["판단_정리전"] = result["VIF_정리전"].map(level)
result["판단_정리후"] = result["VIF_정리후"].map(lambda v: level(v) if pd.notna(v) else "삭제")
result = result.sort_values("VIF_정리전", ascending=False)
result.round(2).to_csv(f"{OUT_DIR}/VIF_결과.csv", encoding="utf-8-sig")

pd.set_option("display.width", 200)
print("\n[VIF 결과] (정리 전 큰 순서, 정리 후 '삭제' = 피어슨 삭제 후보)")
print(result.round(2).to_string())
for name, s in [("정리 전", before), ("정리 후", after)]:
    print(f"{name}: 심각 {sum(s >= 10)}개 / 주의 {sum((s >= 5) & (s < 10))}개 / 괜찮음 {sum(s < 5)}개")


#  3. 그래프 
COLOR = {"심각": "#d93025", "주의": "#eb6834", "괜찮음": "#2a78d6"}

def draw(s, title, path):
    s = s.sort_values()
    shown = s.clip(upper=30)                         # 너무 큰 값은 30에서 자름 (숫자는 그대로 표시)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.3 * len(s) + 1)))
    ax.barh(s.index, shown, color=[COLOR[level(v)] for v in s])
    for y, v in enumerate(s):
        label = "∞ (완전히 겹침)" if v >= 1e5 else f"{v:.1f}"
        ax.text(min(v, 30) + 0.3, y, label, va="center", fontsize=7)
    for t in (5, 10):
        ax.axvline(t, color="#898781", ls="--", lw=0.8)
    ax.set_xlim(0, 34)
    ax.set_xlabel("VIF  (점선: 5 주의, 10 심각 / 30 넘는 값은 막대를 30에서 자름)")
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

draw(before, f"VIF 정리 전 (칸 {len(before)}개)", f"{OUT_DIR}/01_VIF_전체.png")
draw(after, f"VIF 정리 후 (삭제 후보 {len(DROP)}개 뺌, 칸 {len(after)}개)", f"{OUT_DIR}/02_VIF_정리후.png")

print(f"\n저장 위치: {OUT_DIR}")

#사용한 줄 376,090 (2021년 이후) / 계산용 칸 43개
#
#[VIF 결과] (정리 전 큰 순서, 정리 후 '삭제' = 피어슨 삭제 후보)
#                      VIF_정리전  VIF_정리후 판단_정리전 판단_정리후
#도로_큰분류=미분류         1000000.00     1.00     심각    괜찮음
#도로_작은분류=교차로부근      1000000.00     1.13     심각    괜찮음
#도로_작은분류=교차로안       1000000.00     1.18     심각    괜찮음
#도로_작은분류=교차로횡단보도내   1000000.00     1.05     심각    괜찮음
#도로_작은분류=미분류        1000000.00      NaN     심각     삭제
#도로_큰분류=교차로         1000000.00      NaN     심각     삭제
#칸_과거보행자비율               12.00     1.92     심각    괜찮음
#칸_과거차대차비율               11.95      NaN     심각     삭제
#칸_작년까지사고                10.83    10.79     심각     심각
#칸_작년까지큰사고                6.39     6.36     주의     주의
#밤                        5.03     5.03     주의     주의
#칸_작년까지같은시간대사고            4.02     4.02    괜찮음    괜찮음
#태양고도                     3.91     3.91    괜찮음    괜찮음
#주변_작년까지사고                2.86     2.83    괜찮음    괜찮음
#주말                       2.68      NaN    괜찮음     삭제
#요일                       2.60     1.03    괜찮음    괜찮음
#칸_평균제한속도_kmh             2.38     2.38    괜찮음    괜찮음
#칸_간선도로비율                 2.12     2.12    괜찮음    괜찮음
#생활인구_합계                  2.04     2.04    괜찮음    괜찮음
#노면상태=젖음/습기               2.04     2.04    괜찮음    괜찮음
#기상상태=비                   2.02     2.02    괜찮음    괜찮음
#칸_최대차로수                  1.87     1.87    괜찮음    괜찮음
#노면상태=적설                  1.75     1.75    괜찮음    괜찮음
#기상상태=눈                   1.75     1.75    괜찮음    괜찮음
#출퇴근                      1.75     1.75    괜찮음    괜찮음
#칸_과거추돌비율                 1.70     1.67    괜찮음    괜찮음
#칸_평소속도_kmh               1.69     1.68    괜찮음    괜찮음
#칸_과거이륜차비율                1.68     1.68    괜찮음    괜찮음
#칸_도로연장_m                 1.56     1.55    괜찮음    괜찮음
#칸_교량고가비율                 1.36     1.36    괜찮음    괜찮음
#도로_작은분류=교량               1.27     1.27    괜찮음    괜찮음
#발생시                      1.27     1.27    괜찮음    괜찮음
#칸_과거노인피해비율               1.23     1.23    괜찮음    괜찮음
#칸_과거과속신호위반비율             1.20     1.19    괜찮음    괜찮음
#도로_작은분류=터널               1.14     1.14    괜찮음    괜찮음
#칸_과거자전거PM비율              1.12     1.11    괜찮음    괜찮음
#도로_큰분류=기타                1.09     1.09    괜찮음    괜찮음
#박명비율                     1.08     1.08    괜찮음    괜찮음
#칸_직전1시간_속도비율             1.07     1.03    괜찮음    괜찮음
#월                        1.02     1.02    괜찮음    괜찮음
#도로_작은분류=고가도로위            1.02     1.02    괜찮음    괜찮음
#노면상태=서리/결빙               1.01     1.01    괜찮음    괜찮음
#도로_작은분류=지하차도(도로)내        1.01     1.01    괜찮음    괜찮음
#정리 전: 심각 9개 / 주의 2개 / 괜찮음 32개
#정리 후: 심각 1개 / 주의 2개 / 괜찮음 36개

# ── VIF 분석 결과 요약 (train 2021년 이후 376,090줄, 변수 31개 → 원-핫 후 43칸, 기상·노면 Open-Meteo 통일 후) ──
# 정리 전: 심각(≥10) 9개 / 주의(5~10) 2개 / 괜찮음 32개.
#   도로 분류 6칸이 VIF ∞ (완전히 겹침):
#     도로_큰분류=미분류 = 도로_작은분류=미분류 (같은 330줄),
#     도로_큰분류=교차로 = 작은분류의 교차로안 + 교차로부근 + 교차로횡단보도내 (세 칸의 합과 완전히 같음)
#     → 피어슨(두 개씩 비교)으로는 못 찾고 VIF(여러 칸 한꺼번에 비교)로 새로 발견.
#   칸_과거보행자비율(12.0) ↔ 칸_과거차대차비율(11.9): 서로 거의 반대 정보라 둘 다 높게 나옴.
#   칸_작년까지사고(10.8): 큰사고·같은시간대사고·주변사고 정보가 합쳐져 대부분 설명됨.
# 정리 후 (삭제 후보 4개 제외: 도로_작은분류=미분류, 도로_큰분류=교차로, 칸_과거차대차비율, 주말):
#   심각 1개 / 주의 2개 / 괜찮음 36개로 크게 개선.
#   도로 분류 칸들 ∞ → 1.0~1.2, 칸_과거보행자비율 12.0 → 1.9, 요일 2.6 → 1.0으로 겹침 해소.
# 기상·노면 통일 후: 기상상태=비 2.27 → 2.02, 노면상태=젖음/습기 2.29 → 2.04 등 조금 낮아졌을 뿐 모두 '괜찮음' 그대로
#   → 날씨·노면 통일은 다중공선성 결론에 영향 없음.
# 남은 문제:
#   칸_작년까지사고(10.8, 심각): 1단계에서 가장 강한 변수 중 하나라 바로 빼지 않음
#     → 3단계 drop-test에서 '칸_작년까지사고'를 뺄지, 비슷한 과거 기록 변수 중 하나를 뺄지 결정.
#   칸_작년까지큰사고(6.4), 밤(5.0)은 '주의' 수준 → 밤은 태양고도·발생시와 겹침, 3단계에서 함께 확인.
# 결론: 삭제 확정 후보 = 도로 '미분류' 중복, 도로_큰분류(교차로 정보가 작은분류에 다 있음), 칸_과거차대차비율, 주말
#   → 도로_큰분류는 '단일로/기타' 구분만 남으므로, 모델에서는 도로_큰분류 변수 자체를 빼는 것을 검토.
