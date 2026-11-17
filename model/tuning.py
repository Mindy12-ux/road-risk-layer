"""
5단계 - ① 하이퍼파라미터 튜닝 (LightGBM, 최종 세트 C 19개, valid로 채점)
목적: 지금까지 쓴 비교용 기본 설정(나무 300개, 잎 63개 등)을 우리 데이터에 맞게 조정해서 점수 올리기

사용법 (ML project 폴더에서):
    python 모델/tuning.py

입력: data/model_input/train.csv (학습), valid.csv (채점), info.json
출력: reports/tuning/
        튜닝_결과.csv                   시도한 설정 전부 + valid 점수 (좋은 순서)
        최적_설정.json                  가장 좋은 설정 + 나무 개수 → 5-③ 최종 학습에서 그대로 사용

방법: 무작위 탐색 (random search)
    설정 후보 표(SPACE)에서 무작위로 N_TRIALS개 조합을 뽑아 하나씩 학습 → valid PR-AUC 비교
    나무 개수는 '조기 종료'로 정함: valid 점수가 100번 연속 안 오르면 멈추고, 가장 좋았던 개수를 씀
    마지막에 1등 설정을 씨앗 3개로 다시 학습해서 기본 설정과 비교 (운이 좋아서 1등한 건 아닌지 확인)
조정하는 설정 (뜻):
    learning_rate      나무 하나가 배우는 정도 (작을수록 천천히 꼼꼼히, 나무가 더 많이 필요)
    num_leaves         나무 하나의 잎 개수 (클수록 복잡한 규칙)
    min_child_samples  잎 하나에 최소 몇 줄 (클수록 우연한 규칙을 덜 외움)
    subsample          나무마다 줄을 몇 %만 뽑아 쓸지
    colsample_bytree   나무마다 변수를 몇 %만 뽑아 쓸지
    reg_lambda / reg_alpha   규제 (클수록 너무 튀는 규칙을 눌러줌)
※ valid로 설정을 고르므로 valid 점수는 조금 부풀려질 수 있음 → 진짜 성능은 5-③ test로 한 번만 확인
※ 약 7분 걸림
"""

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import ParameterSampler

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
OUT_DIR = f"{ROOT}/reports/tuning"
os.makedirs(OUT_DIR, exist_ok=True)

# 3-⑤에서 고른 최종 세트 C (설명 중심 19개)
FEATURES = ["월", "요일", "발생시", "밤", "기상상태", "노면상태", "도로_작은분류",
            "칸_작년까지사고", "칸_작년까지같은시간대사고", "칸_과거이륜차비율", "칸_과거보행자비율",
            "칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율",
            "생활인구_합계", "칸_평소속도_kmh", "칸_직전1시간_속도비율", "태양고도"]
N_TRIALS = 25
MAX_TREES = 3000          # 조기 종료가 없을 때의 최대 나무 개수
EARLY_STOP = 100          # valid 점수가 이만큼 연속 안 오르면 멈춤
SEEDS = [42, 7, 2024]     # 마지막 확인용

BASE = dict(learning_rate=0.1, num_leaves=63, min_child_samples=100, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0)   # 지금까지 쓴 기본 설정 (나무 300개)
SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [31, 63, 127, 255],
    "min_child_samples": [20, 50, 100, 200, 500],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    "reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}

#  1. 데이터 불러오기 
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
CATS = [c for c in info["categorical"] if c in FEATURES]
X_tr, y_tr = train[FEATURES], train[TARGET]
X_va, y_va = valid[FEATURES], valid[TARGET]
print(f"train {len(train):,}줄 / valid {len(valid):,}줄 / 변수 {len(FEATURES)}개 / 시도 {N_TRIALS}번")


def top_capture(y, p, frac=0.10):
    top = np.argsort(-p)[: int(len(p) * frac)]
    return y.values[top].sum() / y.sum()


def fit(params, seed=42, n_trees=None):
    """n_trees가 없으면 조기 종료로 나무 개수를 정함"""
    m = lgb.LGBMClassifier(objective="binary", n_estimators=n_trees or MAX_TREES, subsample_freq=1,
                           random_state=seed, verbose=-1, **params)
    if n_trees:
        m.fit(X_tr, y_tr, categorical_feature=CATS)
    else:
        m.fit(X_tr, y_tr, categorical_feature=CATS, eval_X=X_va, eval_y=y_va, eval_metric="average_precision",
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
    p = m.predict_proba(X_va)[:, 1]
    trees = m.best_iteration_ if not n_trees else n_trees
    return {"PR-AUC": average_precision_score(y_va, p), "ROC-AUC": roc_auc_score(y_va, p),
            "상위10%_포착률": top_capture(y_va, p), "로그손실": log_loss(y_va, p), "나무_개수": trees}


#  2. 기준: 지금까지 쓴 기본 설정 
t0 = time.time()
base = fit(BASE, n_trees=300)
print(f"\n[기본 설정] PR-AUC {base['PR-AUC']:.4f}  상위10% {base['상위10%_포착률']:.1%}  ({time.time() - t0:.0f}초)")

#  3. 무작위 탐색 
rows = []
for k, params in enumerate(ParameterSampler(SPACE, n_iter=N_TRIALS, random_state=42), start=1):
    t0 = time.time()
    r = fit(params)
    rows.append({"시도": k, **params, **r, "학습시간(초)": time.time() - t0})
    print(f"[{k:>2}/{N_TRIALS}] PR-AUC {r['PR-AUC']:.4f}  나무 {r['나무_개수']:>4}개  "
        f"lr {params['learning_rate']}  잎 {params['num_leaves']}  최소줄 {params['min_child_samples']}  "
        f"({time.time() - t0:.0f}초)", flush=True)

result = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
result.round(5).to_csv(f"{OUT_DIR}/튜닝_결과.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("\n[튜닝 결과] (valid PR-AUC 좋은 순서, 상위 10개)")
print(result.head(10).round(4).to_string(index=False))

#  4. 1등 설정 확인: 씨앗 3개로 기본 설정과 비교 
best = result.iloc[0]
best_params = {k: (int(best[k]) if k in ("num_leaves", "min_child_samples") else float(best[k])) for k in SPACE}
best_trees = int(best["나무_개수"])
check = []
for seed in SEEDS:
    b = fit(BASE, seed=seed, n_trees=300)
    t = fit(best_params, seed=seed, n_trees=best_trees)
    check.append({"씨앗": seed, "기본_PR-AUC": b["PR-AUC"], "튜닝_PR-AUC": t["PR-AUC"],
                "기본_상위10%": b["상위10%_포착률"], "튜닝_상위10%": t["상위10%_포착률"],
                "기본_로그손실": b["로그손실"], "튜닝_로그손실": t["로그손실"]})
check = pd.DataFrame(check).set_index("씨앗")
gain = (check["튜닝_PR-AUC"] - check["기본_PR-AUC"])
print(f"\n[1등 설정 확인] 씨앗 3개 (같은 씨앗끼리 비교)")
print(check.round(4).to_string())
print(f"PR-AUC 향상: 평균 {gain.mean():+.4f} (씨앗별 {np.round(gain.values, 4)})  "
    f"→ {'세 씨앗 모두 향상' if (gain > 0).all() else '씨앗마다 다름 (운일 수 있음)'}")

best_json = {"params": best_params, "n_estimators": best_trees, "features": FEATURES, "categorical": CATS,
            "valid_PR-AUC_씨앗3평균": round(float(check["튜닝_PR-AUC"].mean()), 4),
            "기본설정_valid_PR-AUC_씨앗3평균": round(float(check["기본_PR-AUC"].mean()), 4)}
with open(f"{OUT_DIR}/최적_설정.json", "w", encoding="utf-8") as f:
    json.dump(best_json, f, ensure_ascii=False, indent=2)
print(f"\n최적 설정: {best_params}, 나무 {best_trees}개 → 최적_설정.json 저장")

print(f"\n저장 위치: {OUT_DIR}")

# 튜닝 결과: 무작위 탐색 25번 중 1등 설정(lr 0.03, 잎 127, 변수 50%, 규제 λ5, 나무 404개)이 
# valid PR-AUC 0.5098 → 0.5127로 씨앗 3개 모두 올랐지만(+0.003) 향상 폭이 작아, 점수를 더 올리려면 설정보다 새 데이터·변수가 필요함.