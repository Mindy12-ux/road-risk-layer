"""
3단계 - ③ Permutation 중요도 (LightGBM, valid로 측정)
목적: 변수 하나의 값을 '마구 뒤섞었을 때' 점수(PR-AUC)가 얼마나 떨어지나
    → 많이 떨어지면 모델이 그 변수에 정말 의지하는 것. 거의 안 떨어지면 없어도 되는 변수.
    → 3-②의 중요도(학습 때 많이 쓴 정도)와 달리 '실제 점수 기여'를 봄. valid로 재서 과대평가도 적음.

사용법 (ML project 폴더에서):
    python 상관분석/feat/permutation.py

입력: model/results/model_input/train.csv (학습), valid.csv (측정), info.json
출력: reports/model_permutation/
        permutation_결과.csv
        01_permutation_막대.png       변수별 PR-AUC 하락 (막대 끝 선 = 5번 반복의 흩어짐)
        02_gain_vs_permutation.png    3-② 중요도(gain)와 permutation 비교

설정: 변수 28개, 가중치 없음, LightGBM (3-②에서 가장 좋았던 모델)
방법: 변수마다 valid의 그 칸만 무작위로 섞기 → PR-AUC 다시 계산 → 원래 점수와의 차이
    5번 반복해서 평균 (섞는 운에 따라 결과가 조금씩 달라서)
판단: PR-AUC 하락 ≤ 0  → 섞어도 점수가 그대로 (없어도 되는 변수 후보)
    하락이 반복 흩어짐(표준편차)보다 작으면 → 우연과 구분 안 됨
※ 약 1분 걸림
"""

import json
import os
import time

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

import platform
plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # 저장소 맨 위 폴더
DATA_DIR = f"{ROOT}/model/results/model_input"   # dvc pull로 받은 데이터
OUT_DIR = f"{ROOT}/reports/model_permutation"
os.makedirs(OUT_DIR, exist_ok=True)

DROP = ["주말", "칸_과거차대차비율", "도로_큰분류"]     # 2단계(다중공선성)에서 정리한 변수
N_REPEATS = 5

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
FEATURES = [c for c in info["features"] if c not in DROP]
CATS = [c for c in info["categorical"] if c in FEATURES]

groups = info["feature_groups"]
GROUP_OF = {c: "기본" for c in groups["기본"]}
GROUP_OF.update({c: "1순위" for c in groups["1순위"]})
GROUP_OF.update({c: "2순위" for cols in groups["2순위"].values() for c in cols})
COLOR = {"기본": "#2a78d6", "1순위": "#eb6834", "2순위": "#1baf7a"}

X_tr, y_tr = train[FEATURES], train[TARGET]
X_va, y_va = valid[FEATURES], valid[TARGET]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 변수 {len(FEATURES)}개")

#  2. LightGBM 학습 (3-②와 같은 설정) 
model = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        random_state=42, verbose=-1)
model.fit(X_tr, y_tr, categorical_feature=CATS)
base = average_precision_score(y_va, model.predict_proba(X_va)[:, 1])
print(f"원래 valid PR-AUC: {base:.4f}")

#  3. Permutation 중요도 
t0 = time.time()
perm = permutation_importance(model, X_va, y_va, scoring="average_precision",
                            n_repeats=N_REPEATS, random_state=42, n_jobs=1)
print(f"permutation 끝 ({time.time() - t0:.0f}초)")

gain = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
result = pd.DataFrame({
    "묶음": [GROUP_OF[c] for c in FEATURES],
    "PR-AUC_하락": perm.importances_mean,
    "하락_표준편차": perm.importances_std,
    "하락_비율(%)": perm.importances_mean / base * 100,
    "gain_중요도(%)": (gain / gain.sum() * 100).values,
}, index=FEATURES)


def judge(r):
    if r["PR-AUC_하락"] <= 0:
        return "없어도 됨 (섞어도 그대로)"
    if r["PR-AUC_하락"] < 2 * r["하락_표준편차"]:
        return "우연과 구분 안 됨"
    if r["PR-AUC_하락"] >= 0.01:
        return "중요"
    return "조금 도움"


result["판단"] = result.apply(judge, axis=1)
result["permutation_순위"] = result["PR-AUC_하락"].rank(ascending=False).astype(int)
result["gain_순위"] = result["gain_중요도(%)"].rank(ascending=False).astype(int)
result = result.sort_values("PR-AUC_하락", ascending=False)
result.round(5).to_csv(f"{OUT_DIR}/permutation_결과.csv", encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n[permutation 결과] (PR-AUC 하락 큰 순서)")
print(result.round(4).to_string())
print("\n판단별 개수:", result["판단"].value_counts().to_dict())

#  4. 그래프 ① PR-AUC 하락 막대 
plot = result.iloc[::-1]
fig, ax = plt.subplots(figsize=(10, 10))
ax.barh(plot.index, plot["PR-AUC_하락"], xerr=plot["하락_표준편차"],
        color=[COLOR[g] for g in plot["묶음"]], error_kw={"elinewidth": 0.8, "ecolor": "#333"})
for y, v in enumerate(plot["PR-AUC_하락"]):
    ax.text(max(v, 0) + 0.002, y, f"{v:.4f}".replace("−", "-"), va="center", fontsize=7)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(0.01, color="#898781", ls="--", lw=0.8)
ax.set_xlabel(f"valid PR-AUC 하락 (원래 {base:.3f}에서 얼마나 떨어지나, 점선 = 0.01)")
ax.set_title(f"Permutation 중요도: 변수를 섞으면 점수가 얼마나 떨어지나 (LightGBM, {N_REPEATS}번 평균)", loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, COLOR.keys(), loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_permutation_막대.png", dpi=150)
plt.close()

#  5. 그래프 ② gain 중요도 vs permutation 
fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(result["gain_중요도(%)"], result["PR-AUC_하락"], c=[COLOR[g] for g in result["묶음"]], s=40)
for name, r in result.iterrows():
    if r["PR-AUC_하락"] < 0.004 and r["gain_중요도(%)"] < 3:   # 아래쪽에 몰린 작은 변수는 이름 생략 (CSV에 있음)
        continue
    ax.annotate(name, (r["gain_중요도(%)"], r["PR-AUC_하락"]), fontsize=7, xytext=(4, 2),
                textcoords="offset points")
ax.set_xscale("log")
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("gain 중요도 (%, 3-② 방식: 학습 때 많이 쓴 정도)")
ax.set_ylabel("permutation PR-AUC 하락 (실제 점수 기여)")
ax.set_title("학습 때 많이 쓴 정도 vs 실제 점수 기여 (오른쪽 위일수록 둘 다 중요)", loc="left")
ax.legend(handles, COLOR.keys(), loc="upper left", frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_gain_vs_permutation.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

#원래 valid PR-AUC: 0.5105
#permutation 끝 (40초)
#
#[permutation 결과] (PR-AUC 하락 큰 순서)
#                묶음  PR-AUC_하락  하락_표준편차  하락_비율(%)  gain_중요도(%)               판단  permutation_순위  gain_순위
#칸_작년까지사고        기본     0.0695   0.0014   13.6056      13.1569               중요               1        2
#발생시             기본     0.0400   0.0004    7.8353       8.1341               중요               2        3
#생활인구_합계        2순위     0.0252   0.0009    4.9424       5.4740               중요               3        5
#칸_작년까지같은시간대사고   기본     0.0183   0.0004    3.5856      34.8554               중요               4        1
#칸_도로연장_m       2순위     0.0071   0.0003    1.3918       4.0840            조금 도움               5        6
#칸_직전1시간_속도비율   2순위     0.0057   0.0004    1.1121       3.7427            조금 도움               6        7
#요일              기본     0.0053   0.0006    1.0403       1.6214            조금 도움               7       12
#태양고도           2순위     0.0035   0.0003    0.6913       2.2571            조금 도움               8        9
#칸_평균제한속도_kmh   2순위     0.0034   0.0004    0.6691       1.8444            조금 도움               9       10
#칸_평소속도_kmh     2순위     0.0031   0.0002    0.6067       2.4726            조금 도움              10        8
#칸_최대차로수        2순위     0.0029   0.0002    0.5741       1.2540            조금 도움              11       18
#월               기본     0.0025   0.0004    0.4899       0.9764            조금 도움              12       21
#주변_작년까지사고       기본     0.0022   0.0004    0.4255       1.7168            조금 도움              13       11
#칸_작년까지큰사고       기본     0.0021   0.0005    0.4041       6.3381            조금 도움              14        4
#칸_과거노인피해비율     1순위     0.0016   0.0001    0.3185       1.3232            조금 도움              15       17
#칸_과거이륜차비율      1순위     0.0016   0.0001    0.3175       1.3787            조금 도움              16       14
#칸_과거보행자비율      1순위     0.0014   0.0002    0.2741       1.5855            조금 도움              17       13
#도로_작은분류         기본     0.0014   0.0002    0.2703       0.6243            조금 도움              18       23
#밤               기본     0.0011   0.0001    0.2218       1.1747            조금 도움              19       20
#칸_간선도로비율       2순위     0.0011   0.0003    0.2187       0.7310            조금 도움              20       22
#출퇴근             기본     0.0010   0.0001    0.2042       0.2600            조금 도움              21       26
#칸_과거자전거PM비율    1순위     0.0010   0.0002    0.1921       1.3697            조금 도움              22       15
#칸_교량고가비율       2순위     0.0006   0.0002    0.1156       0.4105            조금 도움              23       24
#노면상태            기본     0.0005   0.0001    0.1018       0.1866            조금 도움              24       27
#칸_과거추돌비율       1순위     0.0005   0.0004    0.0967       1.2517       우연과 구분 안 됨              25       19
#칸_과거과속신호위반비율   1순위     0.0004   0.0002    0.0783       1.3324            조금 도움              26       16
#기상상태            기본     0.0001   0.0001    0.0238       0.1001            조금 도움              27       28
#박명비율           2순위    -0.0001   0.0000   -0.0248       0.3436  없어도 됨 (섞어도 그대로)              28       25
#
#판단별 개수: {'조금 도움': 22, '중요': 4, '우연과 구분 안 됨': 1, '없어도 됨 (섞어도 그대로)': 1}

# ── Permutation 중요도 결과 요약 (LightGBM, 변수 28개, valid PR-AUC 0.511, 기상·노면 Open-Meteo 통일 후) ──
# '중요'(PR-AUC 하락 ≥ 0.01)는 4개: 칸_작년까지사고 0.070(점수의 13.6%, 압도적 1위), 발생시 0.040,
#   생활인구_합계 0.025, 칸_작년까지같은시간대사고 0.018.
# 칸_작년까지같은시간대사고는 gain 1위(35%)인데 permutation 4위 → 작년까지사고와 정보가 겹쳐서 섞어도 대신 맞혀줌.
#   칸_작년까지큰사고도 gain 4위 → permutation 14위(0.002)로 같은 이유.
# '조금 도움'(0~0.01): 도로연장 0.007, 직전1시간 속도비율 0.006, 요일 0.005, 태양고도·제한속도·평소속도·최대차로수 0.003,
#   월 0.0025, 주변사고 0.002, 1순위 변수(노인·이륜차·보행자·자전거PM 0.001~0.0016), 밤·출퇴근·간선 약 0.001.
# 통일 전 2위·6위였던 노면상태(0.0005, 24위)와 기상상태(0.0001, 27위)는 거의 기여 없음으로 떨어짐
#   → 통일 전 점수는 '출처 흔적'이었음이 다시 확인됨. 월도 4위 → 12위로 내려감.
# 칸_과거추돌비율은 '우연과 구분 안 됨', 박명비율은 섞어도 점수가 그대로라 '없어도 됨'.
# → 실제 점수는 '과거 사고 + 시간 + 생활인구'가 대부분을 만들고, 나머지는 drop-test로 뺄지 확인.
