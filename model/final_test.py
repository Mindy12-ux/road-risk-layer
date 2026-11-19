"""
5단계 - ③ test로 최종 성능 확인 (한 번만!)
목적: 지금까지 한 번도 안 쓴 test로 '진짜 성능'을 확인 → 발표·보고서에 쓸 최종 점수

사용법 (ML project 폴더에서, model/tuning.py → model/risk_level.py 를 먼저 실행해야 함):
    python model/final_test.py

입력: data/model_input/train.csv, valid.csv, test.csv, info.json
    reports/tuning/최적_설정.json      (5-① 튜닝 설정 + 나무 개수 + 변수 목록)
    reports/risk_level/위험등급_기준.json (5-② 높음/보통 경계 확률값)
출력: reports/final_test/
        test_점수.csv          모델 2개(train만 / train+valid)의 valid·test 점수
        test_등급.csv          5-② 경계를 test에 그대로 적용한 등급별 사고 비율, 포착률
        test_심각도별_포착.csv  경상/중상/사망 사고가 각각 어느 등급에 들어갔나 (참고용)

모델 2개를 만드는 이유:
    ① train만으로 학습 (5-②에서 경계를 정한 그 모델)  → valid 점수와 test 점수를 같은 모델로 비교 (부풀려졌나 확인)
    ② train + valid 합쳐서 학습 (서비스에 쓸 최종 모델) → 데이터가 많아져서 조금 더 좋아지나 확인
    ※ ②는 확률 크기가 조금 달라질 수 있어서, 같은 경계를 썼을 때 높음 비율이 10%에서 얼마나 벗어나는지도 봄
판단:
    test PR-AUC가 valid보다 0.01 넘게 떨어지면 → valid에 너무 맞춘 것 (튜닝 다시 생각)
    test 등급에서 높음 > 보통 > 낮음 순서가 지켜지고, 높음 줄 비율이 10% 근처면 → 경계 그대로 사용
※ test는 이번에 한 번만 보고, 결과를 보고 설정을 다시 바꾸지 않음 (바꾸면 test도 '연습 문제'가 됨)
※ 약 1분 걸림
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
TUNING_PATH = f"{ROOT}/reports/tuning/최적_설정.json"
LEVEL_PATH = f"{ROOT}/reports/risk_level/위험등급_기준.json"
OUT_DIR = f"{ROOT}/reports/final_test"
os.makedirs(OUT_DIR, exist_ok=True)

LEVELS = ["높음", "보통", "낮음"]
SEVERITY = {1: "경상", 2: "중상", 3: "사망"}

#  1. 설정 + 데이터 불러오기 
for p, step in [(TUNING_PATH, "model/tuning.py"), (LEVEL_PATH, "model/risk_level.py")]:
    assert os.path.exists(p), f"{p} 가 없어요 → {step} 를 먼저 실행하세요"
with open(TUNING_PATH, encoding="utf-8") as f:
    tuned = json.load(f)
with open(LEVEL_PATH, encoding="utf-8") as f:
    cut = json.load(f)
FEATURES, CATS = tuned["features"], tuned["categorical"]
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
test = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    TARGET = json.load(f)["target"]
both = pd.concat([train, valid], ignore_index=True)
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / test {len(test):,}줄 / 변수 {len(FEATURES)}개")
print(f"위험 등급 경계 (후보 {cut['후보']}): 확률 ≥ {cut['cut_high']:.4f} → 높음 / ≥ {cut['cut_mid']:.4f} → 보통")


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y[top].sum() / y.sum()


def scores(y, p):
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "상위10%_포착률": top_capture(y, p), "로그손실": log_loss(y, p)}


def fit(df):
    m = lgb.LGBMClassifier(objective="binary", n_estimators=tuned["n_estimators"], subsample_freq=1,
                           random_state=42, verbose=-1, **tuned["params"])
    m.fit(df[FEATURES], df[TARGET], categorical_feature=CATS)
    return m


def assign(p):
    return np.where(p >= cut["cut_high"], "높음", np.where(p >= cut["cut_mid"], "보통", "낮음"))


#  2. 모델 2개 학습 + 채점 
y_va, y_te = valid[TARGET].values, test[TARGET].values
m1 = fit(train)                               # ① train만
m2 = fit(both)                                # ② train + valid (최종 모델)
p1_va = m1.predict_proba(valid[FEATURES])[:, 1]
p1_te = m1.predict_proba(test[FEATURES])[:, 1]
p2_te = m2.predict_proba(test[FEATURES])[:, 1]

rows = [{"모델": "① train만", "채점": "valid", **scores(y_va, p1_va)},
        {"모델": "① train만", "채점": "test", **scores(y_te, p1_te)},
        {"모델": "② train+valid (최종)", "채점": "test", **scores(y_te, p2_te)}]
score_table = pd.DataFrame(rows)
score_table.round(5).to_csv(f"{OUT_DIR}/test_점수.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("\n[점수]")
print(score_table.round(4).to_string(index=False))
drop = score_table.loc[0, "PR-AUC"] - score_table.loc[1, "PR-AUC"]
gain = score_table.loc[2, "PR-AUC"] - score_table.loc[1, "PR-AUC"]
print(f"같은 모델 ①의 valid → test PR-AUC 변화: {-drop:+.4f}  "
    f"→ {'valid에 너무 맞춤 (주의)' if drop > 0.01 else '부풀려지지 않음 (test에서도 비슷)'}")
print(f"train+valid로 다시 학습한 효과 (② - ①, test): {gain:+.4f}")
print(f"참고: 아무렇게나 찍었을 때 PR-AUC ≈ {y_te.mean():.3f}, 상위10% 포착률 ≈ 10%")

#  3. 5-② 경계를 test에 그대로 적용 
base_rate = y_te.mean()
lv_rows = []
for name, p in [("① train만", p1_te), ("② train+valid (최종)", p2_te)]:
    level = assign(p)
    for lv in LEVELS:
        m = level == lv
        lv_rows.append({"모델": name, "등급": lv, "줄_비율": m.mean(), "사고_비율": y_te[m].mean(),
                        "배수(평균 대비)": y_te[m].mean() / base_rate, "포착률(실제 사고 중)": y_te[m].sum() / y_te.sum(),
                        "valid_사고비율": cut["valid_사고비율"][lv], "valid_포착률": cut["valid_포착률"][lv]})
lv_table = pd.DataFrame(lv_rows)
lv_table.round(4).to_csv(f"{OUT_DIR}/test_등급.csv", index=False, encoding="utf-8-sig")
print(f"\n[test 등급] (5-② 경계 그대로, valid에서 정할 때 높음 줄 비율 = {cut['높음_상위비율']:.0%})")
for name in lv_table["모델"].unique():
    t = lv_table[lv_table["모델"] == name].set_index("등급").drop(columns="모델")
    ok = t.loc["높음", "사고_비율"] > t.loc["보통", "사고_비율"] > t.loc["낮음", "사고_비율"]
    print(f"\n{name}  → 높음 > 보통 > 낮음 순서: {'O' if ok else 'X'}, "
        f"높음 줄 비율 {t.loc['높음', '줄_비율']:.1%} (목표 {cut['높음_상위비율']:.0%})")
    print(t.round(4).to_string())

#  4. (참고) 심각도별로 어느 등급에 들어갔나 — 최종 모델 ② 
level = assign(p2_te)
sev = test["심각도_참고"].values
sev_rows = []
for code, label in SEVERITY.items():
    m = sev == code
    sev_rows.append({"심각도": label, "사고 수": int(m.sum()),
                     **{f"{lv}_비율": (level[m] == lv).mean() for lv in LEVELS},
                    "높음+보통_비율": (level[m] != "낮음").mean()})
sev_table = pd.DataFrame(sev_rows)
sev_table.round(4).to_csv(f"{OUT_DIR}/test_심각도별_포착.csv", index=False, encoding="utf-8-sig")
print("\n[참고] 심각도별로 어느 등급에 들어갔나 (최종 모델 ②, test 실제 사고만)")
print(sev_table.round(3).to_string(index=False))

print(f"\n저장 위치: {OUT_DIR}")

# test 결과: 최종 모델(train+valid로 학습)의 test PR-AUC는 0.518로 valid(0.513)와 비슷해 점수가 부풀려지지 않았음. 
# valid에서 정한 경계(B)를 test에 그대로 써도 높음 등급이 전체의 10.1%이고, 
# 사고 비율 60.5%로 평균(24.6%)의 2.5배라 경계를 그대로 써도 됨. 단, 사망사고는 높음+보통에 34%만 잡혀 한계로 남음