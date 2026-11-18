"""
모델 성능 페이지용 '성적표' 파일 만들기 → web/data/performance_static.json
목적: 이미 저장한 최종 모델(model/results/models)을 다시 학습하지 않고 불러와서,
      2025 test로 채점한 숫자를 웹 '모델 성능' 페이지가 읽는 JSON 하나로 모아 두기

사용법 (road-risk-layer 폴더에서, dvc pull 로 model/results 를 받은 뒤):
    python model/make_performance.py

입력: model/results/models/        사고위험_모델.txt, 위험등급_기준.json, 모델_정보.json
      model/results/model_input/test.csv   (사고여부, 심각도_참고 포함)
출력: web/data/performance_static.json
        metrics        ROC-AUC, PR-AUC, 무작위 기준(= test 사고 비율)
        grades         높음/보통/낮음마다 줄 비율, 사고 비율, 평균 대비 배수, 포착률
        deciles        점수 10구간(하위→상위)별 실제 사고 비율 (막대그래프)
        fatal_capture  사망사고 중 높음+보통에 들어간 비율
        service_check  make_features.py 확인 결과 (칸+시각만으로 변수 재생성) — 숫자는 아래 상수에서 옮겨 적음
※ 모델 파일은 읽기만 함. 모델을 다시 학습(save_model.py)했을 때만 다시 실행하면 됨
※ 약 10초
"""

import json
import os
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model", "results", "models")
TEST_PATH = os.path.join(ROOT, "model", "results", "model_input", "test.csv")
OUT_PATH = os.path.join(ROOT, "web", "data", "performance_static.json")
TARGET, SEVERITY = "사고여부", "심각도_참고"
LEVELS = ["높음", "보통", "낮음"]
RANK_RANGE = {"높음": "상위 10%", "보통": "상위 10~30%", "낮음": "나머지 70%"}   # risk_level.py 후보 B

# make_features.py check() 결과 (서비스 방식 변수로 test 다시 채점) — 다시 돌리면 여기 숫자만 바꾸기
SERVICE_CHECK = {"pr_auc_before": 0.5183, "pr_auc_after": 0.5156,
                 "roc_auc_before": 0.7852, "roc_auc_after": 0.7840}


def load_json(name):
    with open(os.path.join(MODEL_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def booster():
    path = os.path.join(MODEL_DIR, "사고위험_모델.txt")
    with open(path, encoding="utf-8") as f:            # Windows 한글 경로 우회 (web/app.py 와 같은 방식)
        return lgb.Booster(model_str=f.read())


def main():
    info, cut = load_json("모델_정보.json"), load_json("위험등급_기준.json")
    features = info["features"]
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    y = test[TARGET].values
    p = booster().predict(test[features].astype(float))
    base = y.mean()
    grade = np.where(p >= cut["cut_high"], "높음", np.where(p >= cut["cut_mid"], "보통", "낮음"))

    grades = []
    for lv in LEVELS:
        m = grade == lv
        rate = y[m].mean() if m.any() else None
        grades.append({"grade": lv, "rank_range": RANK_RANGE[lv], "row_share": round(m.mean(), 4),
                       "accident_rate": None if rate is None else round(rate, 4),
                       "lift": None if rate is None else round(rate / base, 2),
                       "capture": round(y[m].sum() / y.sum(), 4)})

    order = np.argsort(p)                                # 낮은 점수 → 높은 점수
    deciles = [round(y[idx].mean(), 4) for idx in np.array_split(order, 10)]

    fatal = None
    if SEVERITY in test:
        m = test[SEVERITY].values == 3
        fatal = round((grade[m] != "낮음").mean(), 4) if m.any() else None

    out = {
        "model": {"name": "LightGBM 이진 분류", "n_features": len(features),
                  "train_years": "2023-2024", "test_year": 2025,
                  "trained_on": info.get("학습날짜") or info.get("학습_날짜"),
                  "made_on": date.today().isoformat()},
        "metrics": {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                    "pr_auc_baseline": round(base, 4), "n_test_rows": int(len(y))},
        "grades": grades,
        "deciles": deciles,
        "fatal_capture_high_mid": fatal,
        "service_check": SERVICE_CHECK,
        "notes": {"score_is_rank": True, "placeholder": False},
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"test {len(y):,}줄 → ROC-AUC {out['metrics']['roc_auc']}, PR-AUC {out['metrics']['pr_auc']}")
    for g in grades:
        print(f"  {g['grade']}: 줄 {g['row_share']:.1%}, 사고 비율 {g['accident_rate']}, 배수 {g['lift']}")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
