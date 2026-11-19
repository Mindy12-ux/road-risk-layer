"""2단계 LightGBM 모델: 1단계 사고 여부 → 2단계 사고 심각도(경상/중상/사망).

입력: data/model (scripts/make_model_data.py 결과 = dataset_v2 형식 + 팀원 변수)
- train.csv(2021~23), valid.csv(2024), test.csv(2025)  (이름에 _v2가 붙어 있어도 읽음)
- 라벨 '사고심각도': 0 무사고, 1 경상, 2 중상, 3 사망
- 위도·경도·연도는 이미 제거됨(좌표 암기 방지). 위치는 cx, cy와 칸 이력 피처로만 들어감.
- 사망이 0.16%로 희소해서 한 번에 4클래스로 풀지 않고 두 단계로 나눔.
    1단계: 사고(1~3) vs 무사고(0)          → 이진 분류, 평가 AUC
    2단계: 사고 행만 모아서 경상/중상/사망 → 3클래스, 클래스 가중치로 희소 클래스 보정
    최종 확률: P(경상) = P(사고) × P(경상|사고) ... 처럼 곱해서 4클래스 확률을 만듦.
- 검증(2024)으로 조기 종료, 테스트(2025)는 마지막에 한 번만 평가.

사용 (저장소 맨 위 폴더에서):
    pip install lightgbm scikit-learn pandas
    python model/train_two_stage.py                              # data/model → results/model
    python model/train_two_stage.py --drop 칸_작년까지큰사고 실제밤  # 상관관계 보고 뺄 변수 지정
    python model/train_two_stage.py --only-base --out results/model_base   # 팀원 변수 없이 (비교용)
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

TARGET = "사고심각도"
# 숫자로 적혀 있지만 크기 순서가 없는 범주형 컬럼 (데이터에 있는 것만 씀). 문자 컬럼도 자동으로 범주형.
CAT_ALL = ["요일", "기상상태", "노면상태", "도로_큰분류", "도로_작은분류", "칸_최고도로등급", "칸_속도출처"]
CAT = []  # load()에서 채움
BASE_FEATS = ["월", "요일", "발생시", "주말", "밤", "출퇴근", "기상상태", "노면상태", "도로_큰분류", "도로_작은분류",
              "칸_작년까지사고", "칸_작년까지큰사고", "칸_작년까지같은시간대사고", "주변_작년까지사고", "cx", "cy"]
ROOT = Path(__file__).resolve().parents[1]
SEV_NAMES = ["경상", "중상", "사망"]
SEED = 42
WEIGHT = "balanced"  # main()에서 --weight 옵션으로 덮어씀


def find_csv(data_dir, k):
    """train_v2.csv처럼 _v2가 붙은 이름을 먼저 찾고, 없으면 train.csv를 씀."""
    for name in (f"{k}_v2.csv", f"{k}.csv"):
        p = Path(data_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{data_dir}에 {k}_v2.csv 또는 {k}.csv가 없어요. --data로 폴더를 지정하세요.")


def load(data_dir, drop=(), only_base=False):
    rd = lambda k: pd.read_csv(find_csv(data_dir, k), encoding="utf-8-sig", low_memory=False)
    tr, va, te = rd("train"), rd("valid"), rd("test")
    if "사고여부" in tr and TARGET not in tr:
        raise SystemExit("[오류] 라벨이 '사고여부'예요. data/merged 말고 make_model_data.py 결과(data/model)를 넣으세요.")
    wrong = [c for c in drop if c not in tr.columns]
    if wrong:
        raise SystemExit(f"[오류] --drop 에 없는 컬럼: {wrong}")
    feats = [c for c in tr.columns if c != TARGET and c not in drop and (not only_base or c in BASE_FEATS)]
    CAT[:] = [c for c in CAT_ALL if c in feats] + [c for c in feats if tr[c].dtype == object and c not in CAT_ALL]
    for d in (tr, va, te):
        for c in CAT:
            d[c] = d[c].astype("category")
    # 범주 값 목록을 학습 데이터 기준으로 맞춰 둠(검증·테스트에 새 값이 있어도 깨지지 않게)
    for c in CAT:
        cats = sorted(set(tr[c].cat.categories) | set(va[c].cat.categories) | set(te[c].cat.categories))
        for d in (tr, va, te):
            d[c] = d[c].cat.set_categories(cats)
    return tr, va, te, feats


def stage1(tr, va, feats):
    """사고 여부(이진) 모델."""
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=SEED)
    dtr = lgb.Dataset(tr[feats], (tr[TARGET] > 0).astype(int), categorical_feature=CAT)
    dva = lgb.Dataset(va[feats], (va[TARGET] > 0).astype(int), categorical_feature=CAT, reference=dtr)
    return lgb.train(params, dtr, 3000, valid_sets=[dva], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def stage2(tr, va, feats):
    """사고 행만으로 심각도(경상 0 / 중상 1 / 사망 2) 3클래스 모델."""
    a_tr, a_va = tr[tr[TARGET] > 0], va[va[TARGET] > 0]
    y_tr, y_va = a_tr[TARGET] - 1, a_va[TARGET] - 1
    # 클래스 가중치: balanced는 희소 클래스(사망)를 예측하게 만들지만 경상 정확도가 떨어짐.
    # sqrt는 그 중간(대신 argmax로는 사망을 거의 예측하지 않음). --weight 옵션으로 바꿔 볼 수 있음.
    f = (lambda w: w) if WEIGHT == "balanced" else np.sqrt if WEIGHT == "sqrt" else np.ones_like
    w_tr = f(compute_sample_weight("balanced", y_tr))
    w_va = f(compute_sample_weight("balanced", y_va))
    params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=31,
                  min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=5.0, verbose=-1, seed=SEED)
    dtr = lgb.Dataset(a_tr[feats], y_tr, weight=w_tr, categorical_feature=CAT)
    dva = lgb.Dataset(a_va[feats], y_va, weight=w_va, categorical_feature=CAT, reference=dtr)
    return lgb.train(params, dtr, 3000, valid_sets=[dva], valid_names=["valid"],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])


def predict4(m1, m2, X):
    """두 모델을 곱해서 [무사고, 경상, 중상, 사망] 4클래스 확률을 만듦."""
    p_acc = m1.predict(X, num_iteration=m1.best_iteration)
    p_sev = m2.predict(X, num_iteration=m2.best_iteration)
    return np.column_stack([1 - p_acc, p_acc[:, None] * p_sev])


def evaluate(name, d, m1, m2, feats):
    y = d[TARGET].to_numpy()
    p4 = predict4(m1, m2, d[feats])
    p_acc = 1 - p4[:, 0]
    acc = d[TARGET] > 0
    p_sev = m2.predict(d.loc[acc, feats], num_iteration=m2.best_iteration)
    y_sev = y[acc] - 1
    big = (y_sev >= 1).astype(int)  # 큰 사고 = 중상 + 사망
    r = {
        "1단계_사고여부_AUC": roc_auc_score(acc, p_acc),
        "1단계_사고여부_PR_AUC": average_precision_score(acc, p_acc),
        "2단계_큰사고(중상+사망)_AUC": roc_auc_score(big, p_sev[:, 1] + p_sev[:, 2]),
        "2단계_사망_AUC": roc_auc_score(y_sev == 2, p_sev[:, 2]),
        "2단계_macro_F1": f1_score(y_sev, p_sev.argmax(1), average="macro"),
        "전체_4클래스_macro_F1": f1_score(y, p4.argmax(1), average="macro"),
    }
    print(f"\n===== {name} =====")
    for k, v in r.items():
        print(f"{k:28s} {v:.4f}")
    print("\n2단계 혼동행렬 (행=실제, 열=예측: 경상/중상/사망)")
    print(confusion_matrix(y_sev, p_sev.argmax(1)))
    print(classification_report(y_sev, p_sev.argmax(1), target_names=SEV_NAMES, digits=3, zero_division=0))
    return r, p4


def importance(m, feats):
    return (pd.DataFrame({"피처": feats, "gain": m.feature_importance("gain")})
            .assign(비율=lambda x: (x.gain / x.gain.sum()).round(4))
            .sort_values("gain", ascending=False).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "model"), help="train/valid/test.csv가 있는 폴더 (기본: data/model)")
    ap.add_argument("--out", default=str(ROOT / "results" / "model"))
    ap.add_argument("--drop", nargs="+", default=[], metavar="컬럼", help="학습에서 뺄 변수들")
    ap.add_argument("--only-base", action="store_true", help="원래 16개 변수만 사용 (팀원 변수 효과 비교용)")
    ap.add_argument("--weight", choices=["balanced", "sqrt", "none"], default="balanced",
                    help="2단계 클래스 가중치 (기본 balanced)")
    a = ap.parse_args()
    global WEIGHT
    WEIGHT = a.weight
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    tr, va, te, feats = load(a.data, a.drop, a.only_base)
    print(f"피처 {len(feats)}개: {feats}")
    print(f"train {len(tr):,} / valid {len(va):,} / test {len(te):,}")

    print("\n[1단계] 사고 여부 학습")
    m1 = stage1(tr, va, feats)
    print("\n[2단계] 사고 심각도 학습 (사고 행만)")
    m2 = stage2(tr, va, feats)

    res = {}
    res["valid_2024"], _ = evaluate("검증 2024", va, m1, m2, feats)
    res["test_2025"], p4 = evaluate("테스트 2025", te, m1, m2, feats)

    m1.save_model(str(out / "stage1_accident.txt"), num_iteration=m1.best_iteration)
    m2.save_model(str(out / "stage2_severity.txt"), num_iteration=m2.best_iteration)
    importance(m1, feats).to_csv(out / "importance_stage1.csv", index=False, encoding="utf-8-sig")
    importance(m2, feats).to_csv(out / "importance_stage2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(p4, columns=["P_무사고", "P_경상", "P_중상", "P_사망"]).assign(
        실제=te[TARGET].to_numpy()).to_csv(out / "test_2025_predictions.csv", index=False, encoding="utf-8-sig")
    res["설정"] = {"data": a.data, "weight": a.weight, "drop": a.drop, "only_base": a.only_base, "피처": feats}
    (out / "metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n1단계 피처 중요도(gain) 상위 10\n", importance(m1, feats).head(10).to_string())
    print("\n2단계 피처 중요도(gain) 상위 10\n", importance(m2, feats).head(10).to_string())
    print(f"\n저장 위치: {out.resolve()}")
    print("※ 무사고를 일부만 표본 추출했으므로 P_사고는 실제 확률보다 높게 나옴. 위험도 '순위'로 쓰거나 표본 비율로 보정할 것.")


if __name__ == "__main__":
    main()
