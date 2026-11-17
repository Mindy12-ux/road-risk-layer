"""
4단계 - ③ 이유 문장 만들기 + ④ 예시 점검 (LightGBM, 최종 세트 C 19개, valid로 계산)
목적: 줄(칸·시간)마다 SHAP으로 '위험을 올린' 이유를 골라 내비게이션 문장으로 바꾸기
    예) 위험 확률 78% → ① 최근 3년간 사고가 많았던 곳이에요 ② 18시 무렵에 사고가 잦은 곳이에요
                        ③ 지금 사람이 많은 곳이에요  + [오늘] 비가 와요
     이유를 두 종류로 나눔:
    기본 이유 (장소·시간): 과거 사고, 시간대 사고, 사람 많음, 큰 도로, 교차로, 요일 → 큰 순서로 최대 3개
    오늘 상황 (그날그날 바뀜): 날씨·노면, 막힘 → 위험을 조금이라도 올렸으면(SHAP > 0) 항상 따로 덧붙임
    → 날씨는 영향이 작아서 기본 이유 3개 안에는 거의 못 들어가지만, 기획안대로 '오늘은 비가 와요'를 꼭 보여주기 위함

사용법 (ML project 폴더에서):
    python 상관분석/SHAP/shap_reason.py

입력: data/model_input/train.csv (학습), valid.csv (이유 계산), info.json
출력: reports/SHAP_reason/
        이유_예시_위험상위.csv          위험 확률 가장 높은 20줄 + 이유 3개 + 오늘 상황
        이유_예시_비오는날.csv          비 오는 날 줄 중 위험 상위 20줄 + 이유 3개 + 오늘 상황
        이유_빈도.csv                   이유마다 얼마나 자주 나오나 (전체 / 위험 상위 10%)
        01_이유별_빈도.png              이유별 등장 비율 (기본 이유 / 오늘 상황)
        02_이유_개수.png                줄마다 기본 이유가 몇 개 나오나 (0~3개)
        03_예시_한줄.png                비 오는 날 위험 상위 줄 하나의 이유 묶음별 SHAP

방법:
1) SHAP 계산 (4-①과 같은 모델)
2) 비슷한 변수를 '이유 묶음'으로 합침 (묶음 SHAP = 들어간 변수 SHAP의 합)
    예) 큰 도로 = 도로연장 + 최대차로수 + 제한속도 + 간선도로비율
3) 묶음마다 문장 조건 확인 (예: 날씨는 비·눈·젖은 노면일 때만, 막힘은 속도가 평소의 95% 미만일 때만)
4) 기본 이유: 묶음 SHAP > MIN_SHAP 인 것 중 큰 순서로 최대 3개 → 문장으로 바꿈
5) 오늘 상황: 조건이 맞고 묶음 SHAP > TODAY_MIN 이면 '[오늘] ...' 문장으로 덧붙임
이유에서 빼는 변수 (모델 계산에는 그대로 씀):
    월, 태양고도, 평소속도 → 사용자에게 문장으로 설명하기 어려움
    밤, 보행자비율, 이륜차비율 → 4-②에서 위험을 '내리는' 쪽이거나 방향이 일정하지 않았음
※ 약 40초 걸림
"""

import json
import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
OUT_DIR = f"{ROOT}/reports/SHAP_reason"
os.makedirs(OUT_DIR, exist_ok=True)

# 3-⑤에서 고른 최종 세트 C (설명 중심 19개)
FEATURES = ["월", "요일", "발생시", "밤", "기상상태", "노면상태", "도로_작은분류",
            "칸_작년까지사고", "칸_작년까지같은시간대사고", "칸_과거이륜차비율", "칸_과거보행자비율",
            "칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율",
            "생활인구_합계", "칸_평소속도_kmh", "칸_직전1시간_속도비율", "태양고도"]
MIN_SHAP = 0.05           # 이 값보다 작게 올린 이유는 너무 약해서 안 보여줌
TOP_K = 3                 # 줄마다 보여줄 기본 이유 최대 개수
TODAY = ["날씨·노면", "막힘"]   # 오늘 상황 이유 (기본 이유 3개와 따로 덧붙임)
TODAY_MIN = 0.0           # 오늘 상황은 위험을 조금이라도 올렸으면 보여줌
N_EXAMPLES = 20

#  1. 데이터 불러오기 + 학습 (4-①과 같은 설정) 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
CATS = [c for c in info["categorical"] if c in FEATURES]
CODES = info["category_codes"]
WEEKDAY = {v: k[0] for k, v in info["weekdays"].items()}                 # 0 → 월

model = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.1, num_leaves=63,
                        min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                        random_state=42, verbose=-1)
model.fit(train[FEATURES], train[TARGET], categorical_feature=CATS)
X = valid[FEATURES]
prob = model.predict_proba(X)[:, 1]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / valid PR-AUC {average_precision_score(valid[TARGET], prob):.4f}")

contrib = model.booster_.predict(X, pred_contrib=True)
shap = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X.index)

#  2. 이유 묶음 + 문장 조건 + 문장 
code = {c: {n: i for i, n in enumerate(CODES[c])} for c in CATS}           # 이름 → 번호
CROSS = [code["도로_작은분류"][n] for n in ["교차로안", "교차로부근", "교차로횡단보도내"]]
SKY = {code["기상상태"]["비"]: "비가 와요", code["기상상태"]["눈"]: "눈이 와요"}
SURF = {code["노면상태"]["젖음/습기"]: "노면이 젖어 있어요", code["노면상태"]["적설"]: "눈이 쌓여 있어요",
        code["노면상태"]["서리/결빙"]: "노면이 얼었을 수 있어요"}


def weather_text(r):
    return SKY.get(r["기상상태"]) or SURF.get(r["노면상태"])


# 이유 묶음: (들어가는 변수, 문장을 보여줄 조건, 문장 만드는 함수)
REASONS = {
    "과거 사고": (["칸_작년까지사고"],
            X["칸_작년까지사고"].notna(),
            lambda r: f"최근 3년간 사고가 많았던 곳이에요 (1년 평균 {r['칸_작년까지사고']:.0f}건)"),
    "시간대 사고": (["칸_작년까지같은시간대사고", "발생시"],
            pd.Series(True, index=X.index),
            lambda r: f"{int(r['발생시'])}시 무렵에 사고가 잦은 곳이에요"),
    "사람 많음": (["생활인구_합계"],
            X["생활인구_합계"].notna(),
            lambda r: f"지금 사람이 많은 곳이에요 (생활인구 약 {r['생활인구_합계']:,.0f}명)"),
    "큰 도로": (["칸_도로연장_m", "칸_최대차로수", "칸_평균제한속도_kmh", "칸_간선도로비율"],
            X["칸_도로연장_m"].notna(),
            lambda r: "차가 많이 다니는 큰 도로예요"),
    "막힘": (["칸_직전1시간_속도비율"],
        X["칸_직전1시간_속도비율"] < 0.95,
        lambda r: f"평소보다 차가 막혀요 (속도가 평소의 {r['칸_직전1시간_속도비율']:.0%})"),
    "교차로": (["도로_작은분류"],
            X["도로_작은분류"].isin(CROSS),
            lambda r: "교차로 근처예요"),
    "날씨·노면": (["기상상태", "노면상태"],
            X["기상상태"].isin(SKY) | X["노면상태"].isin(SURF),
            weather_text),
    "요일": (["요일"],
        pd.Series(True, index=X.index),
        lambda r: f"{WEEKDAY[int(r['요일'])]}요일은 사고가 잦은 요일이에요"),
}
BASE_REASONS = [g for g in REASONS if g not in TODAY]
NOT_REASON = [c for c in FEATURES if c not in sum((v[0] for v in REASONS.values()), [])]
print(f"기본 이유 {len(BASE_REASONS)}개: {BASE_REASONS}  /  오늘 상황 {len(TODAY)}개: {TODAY}")
print(f"이유에서 뺀 변수 (모델 계산에는 사용): {NOT_REASON}")

# 묶음 SHAP (조건이 안 맞으면 이유 후보에서 제외)
G = pd.DataFrame({g: shap[cols].sum(axis=1) for g, (cols, _, _) in REASONS.items()})
OK = pd.DataFrame({g: cond.values for g, (_, cond, _) in REASONS.items()}, index=X.index)
cand = G[BASE_REASONS].where(OK[BASE_REASONS] & (G[BASE_REASONS] > MIN_SHAP))   # 기본 이유 후보 (나머지 빈칸)
today_ok = OK[TODAY] & (G[TODAY] > TODAY_MIN)                                     # 오늘 상황 후보

#  3. 줄마다 기본 이유 최대 3개 + 오늘 상황 고르기 
order = np.argsort(-cand.fillna(-np.inf).values, axis=1)[:, :TOP_K]
names = np.array(BASE_REASONS)
picked = []
for i, idx in enumerate(order):
    row = [names[j] for j in idx if not np.isnan(cand.values[i, j])]
    picked.append(row)
n_reason = np.array([len(p) for p in picked])
today_vals = G[TODAY].where(today_ok).values
picked_today = [[TODAY[j] for j in np.argsort(-np.nan_to_num(v, nan=-np.inf)) if not np.isnan(v[j])]
                for v in today_vals]


def sentences(i):
    r = X.iloc[i]
    return [REASONS[g][2](r) for g in picked[i]]


def today_sentences(i):
    r = X.iloc[i]
    return ["[오늘] " + REASONS[g][2](r) for g in picked_today[i]]


def example_table(rows):
    """같은 칸·날짜·시간 줄이 여러 개면(같은 때 사고가 여러 건) 하나만 보여줌"""
    out, seen = [], set()
    for i in rows:
        key = (valid["grid_id"].iloc[i], valid["date"].iloc[i], int(valid["발생시"].iloc[i]))
        if key in seen:
            continue
        seen.add(key)
        s = sentences(i) + [""] * TOP_K
        out.append({"grid_id": valid["grid_id"].iloc[i], "date": valid["date"].iloc[i],
                    "발생시": int(valid["발생시"].iloc[i]), "위험확률": round(prob[i], 3),
                    "실제_사고": int(valid[TARGET].iloc[i]),
                    **{f"이유{k + 1}": s[k] for k in range(TOP_K)},
                    "오늘_상황": " / ".join(today_sentences(i))})
        if len(out) == N_EXAMPLES:
            break
    return pd.DataFrame(out)


#  4. 예시 점검 ① 위험 확률 상위 
top = np.argsort(-prob)[:N_EXAMPLES * 3]
ex_top = example_table(top)
ex_top.to_csv(f"{OUT_DIR}/이유_예시_위험상위.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 45)
print(f"\n[예시 ① 위험 확률 상위 {N_EXAMPLES}줄]")
print(ex_top.to_string(index=False))

#  5. 예시 점검 ② 비 오는 날 
rain = np.where(X["기상상태"].values == code["기상상태"]["비"])[0]
rain_top = rain[np.argsort(-prob[rain])][:N_EXAMPLES * 3]
ex_rain = example_table(rain_top)
ex_rain.to_csv(f"{OUT_DIR}/이유_예시_비오는날.csv", index=False, encoding="utf-8-sig")
print(f"\n[예시 ② 비 오는 날 중 위험 상위 {N_EXAMPLES}줄]")
print(ex_rain.to_string(index=False))
rain_has = np.mean(["날씨·노면" in picked_today[i] for i in rain])
print(f"비 오는 날 {len(rain):,}줄 중 '[오늘] 비가 와요'가 나온 비율: {rain_has:.1%}  "
      f"(나머지는 비가 위험을 올리지 않은 줄 → 안 보여줌)")

#  6. 이유 빈도 + 개수 
high = prob >= np.quantile(prob, 0.9)                                     # 위험 상위 10%
both = [p + t for p, t in zip(picked, picked_today)]                      # 기본 이유 + 오늘 상황
hi = np.where(high)[0]
freq = pd.DataFrame({
    "종류": ["오늘 상황" if g in TODAY else "기본 이유" for g in REASONS],
    "전체_등장비율": [np.mean([g in p for p in both]) for g in REASONS],
    "상위10%_등장비율": [np.mean([g in both[i] for i in hi]) for g in REASONS],
    "조건맞는줄_중_등장비율": [np.mean([g in both[i] for i in np.where(OK[g])[0]]) for g in REASONS],
    "묶음SHAP_평균(올릴때)": [G.loc[G[g] > 0, g].mean() for g in REASONS],
}, index=list(REASONS)).sort_values(["종류", "상위10%_등장비율"], ascending=[True, False])
freq.round(4).to_csv(f"{OUT_DIR}/이유_빈도.csv", encoding="utf-8-sig")
print("\n[이유 빈도] (등장비율 = 보여준 줄의 비율, 조건맞는줄 = 예) 날씨는 비·눈·젖은 노면인 줄 중에서)")
print(freq.round(3).to_string())

count = pd.DataFrame({
    "전체": pd.Series(n_reason).value_counts(normalize=True).sort_index(),  # 기본 이유 개수
    "위험 상위10%": pd.Series(n_reason[high]).value_counts(normalize=True).sort_index(),
}).reindex(range(TOP_K + 1)).fillna(0)
print("\n[줄마다 기본 이유 개수] (비율)")
print(f"오늘 상황이 1개 이상 붙는 줄: 전체 {np.mean([len(t) > 0 for t in picked_today]):.1%}")
print(count.round(3).to_string())

#  7. 그래프 ① 이유별 빈도 
plot = freq.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
y = np.arange(len(plot))
ax.barh(y + 0.2, plot["상위10%_등장비율"] * 100, 0.4, color="#d93025", label="위험 상위 10% 줄")
ax.barh(y - 0.2, plot["전체_등장비율"] * 100, 0.4, color="#898781", label="전체 줄")
for yi, (a, b) in enumerate(zip(plot["상위10%_등장비율"], plot["전체_등장비율"])):
    ax.text(a * 100 + 1, yi + 0.2, f"{a:.0%}", va="center", fontsize=7)
    ax.text(b * 100 + 1, yi - 0.2, f"{b:.0%}", va="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels([f"[오늘] {n}" if n in TODAY else n for n in plot.index])
ax.set_xlim(0, 110)
ax.set_xlabel("이유를 보여준 줄의 비율 (%)")
ax.set_title(f"이유별 등장 비율 (valid, 기본 이유 = SHAP > {MIN_SHAP} 중 최대 {TOP_K}개, "
            f"[오늘] = SHAP > {TODAY_MIN}이면 항상)", loc="left", fontsize=10)
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_이유별_빈도.png", dpi=150)
plt.close()

#  8. 그래프 ② 이유 개수 
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(TOP_K + 1)
ax.bar(x - 0.2, count["전체"] * 100, 0.4, color="#898781", label="전체 줄")
ax.bar(x + 0.2, count["위험 상위10%"] * 100, 0.4, color="#d93025", label="위험 상위 10% 줄")
for xi, (a, b) in enumerate(zip(count["전체"], count["위험 상위10%"])):
    ax.text(xi - 0.2, a * 100 + 1, f"{a:.0%}", ha="center", fontsize=8)
    ax.text(xi + 0.2, b * 100 + 1, f"{b:.0%}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{k}개" for k in x])
ax.set_ylabel("줄 비율 (%)")
ax.set_ylim(0, 105)
ax.set_title("줄마다 보여줄 기본 이유가 몇 개 나오나 (오늘 상황 제외)", loc="left")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_이유_개수.png", dpi=150)
plt.close()

#  9. 그래프 ③ 비 오는 날 예시 한 줄의 묶음별 SHAP 
rain_today = [j for j in rain_top if "날씨·노면" in picked_today[j]]
i = rain_today[0] if rain_today else top[0]                               # 비 오는 날 + 오늘 상황이 붙은 위험 상위 줄
g = G.iloc[i].sort_values()
rest = shap.iloc[i][NOT_REASON].sum()
g["(이유에서 뺀 변수 합)"] = rest
g = g.sort_values()
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#d93025" if (n in picked[i] or n in picked_today[i]) else ("#f4a6a0" if v > 0 else "#2a78d6") for n, v in g.items()]
ax.barh(g.index, g.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
for yi, v in enumerate(g.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), yi, f"{v:+.2f}".replace("−", "-"), va="center",
            ha="left" if v >= 0 else "right", fontsize=8)
span = np.abs(g.values).max()
ax.set_xlim(-span * 1.3, span * 1.3)
ax.set_xlabel("묶음 SHAP (진한 빨강 = 보여줄 이유, 연한 빨강 = 올렸지만 안 보여줌, 파랑 = 위험 ↓)")
title = "\n".join([f"  {k + 1}. {t}" for k, t in enumerate(sentences(i))] + [f"  + {t}" for t in today_sentences(i)])
ax.set_title(f"비 오는 날 예시 ({valid['date'].iloc[i]} {int(valid['발생시'].iloc[i])}시, 칸 {valid['grid_id'].iloc[i]}, "
            f"위험 {prob[i]:.0%})\n{title}", loc="left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_예시_한줄.png", dpi=150)
plt.close()

print(f"\n저장 위치: {OUT_DIR}")

# ── 이유 문장 결과 요약 (LightGBM, 최종 세트 C 19개, valid, SHAP 기반) ──
# 이유를 '기본 이유(장소·시간) 최대 3개'와 '[오늘] 상황(날씨·노면, 막힘)'으로 나눠서 보여줌.
# 기본 이유는 묶음 SHAP > 0.05인 것 중 큰 순서로 고르고, 오늘 상황은 위험을 조금이라도 올렸으면(SHAP > 0) 항상 덧붙임.
# 위험 상위 10% 줄은 99.4%가 기본 이유 3개를 모두 받고, 주로 나오는 이유는
#   시간대 사고(99%), 과거 사고(86%), 사람 많음(64%), 큰 도로(49%)임 → 위험한 곳은 '원래 사고 나던 곳 + 그 시간대 + 사람 많음'으로 설명됨.
# 요일(1%)·교차로(0%)는 위험 상위 줄에서는 거의 안 나오고, 이유가 하나도 없는 줄(9.8%)은 대부분 위험이 낮은 줄임.
# 날씨는 영향이 작아(SHAP 평균 +0.015) 기본 이유 3개 안에는 못 들어가지만, [오늘] 줄로 따로 빼서
#   비 오는 날 위험 상위 20줄 중 19줄, 비 오는 날 전체의 67.4%에 '[오늘] 비가 와요'가 표시됨.
# 나머지 32.6%는 모델상 비가 위험을 올리지 않은 줄이라 근거 없는 이유가 되지 않도록 표시하지 않음.
# [오늘] 막힘은 속도가 평소의 95% 미만인 줄의 거의 전부(99.7%), 위험 상위 10% 줄의 42%에 표시됨.
# 예) 2022-08-30 13시 비 오는 날 위험 89% → 최근 3년간 사고가 많았던 곳(1년 평균 98건) / 13시 무렵 사고가 잦은 곳 /
#   사람이 많은 곳(약 38,000명) + [오늘] 평소보다 차가 막혀요(92%) + [오늘] 비가 와요.
# 주의: 위험 상위는 몇몇 칸(1916_3888, 1915_3890 등)이 반복되고 이유 문장도 비슷함 → 서비스 화면에서 문장 다양성은 추후 검토.