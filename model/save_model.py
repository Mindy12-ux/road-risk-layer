"""
5단계 - ④ 서비스용 최종 모델 + 규칙 파일 저장
목적: Flask가 다시 학습하지 않고 파일만 불러서 바로 예측할 수 있게 models/ 폴더에 모아 두기

사용법 (ML project 폴더에서, model/tuning.py → model/risk_level.py → model/final_test.py 다음에):
    python model/save_model.py

입력: data/model_input/train.csv, valid.csv, test.csv, info.json
    reports/tuning/최적_설정.json       (튜닝 설정 + 나무 개수 + 변수 목록)
    reports/risk_level/위험등급_기준.json (높음/보통 경계)
    reports/final_test/test_점수.csv     (있으면 모델_정보.json에 test 점수 같이 기록)
출력: models/
        사고위험_모델.txt       train + valid 로 학습한 최종 LightGBM (5-③의 ② 모델과 같은 방법)
        위험등급_기준.json      높음/보통 경계 확률값 (5-② 결과 그대로 복사)
        이유문장_규칙.json      이유 묶음, 문장 조건, 문장 틀, MIN_SHAP / TOP_K / TODAY_MIN (4-③ 규칙)
        모델_정보.json          변수 19개 순서, 범주형 변수, 범주 목록(번호 ↔ 이름), 설정, 학습 날짜, test 점수
확인 (저장이 잘 됐나):
    ① 저장한 모델을 다시 불러서 test를 예측 → 학습 직후 예측과 똑같은가
    ② model/predictor.py(Flask용)로 test 일부를 예측 → 등급 비율, 이유 개수, 비 오는 날 '[오늘] 비가 와요' 비율
    ③ 예시 3줄 출력 (위험 가장 높은 줄 / 비 오는 날 / 낮음 줄)
※ 범주 번호(기상상태, 노면상태, 도로_작은분류)는 학습 때와 똑같아야 함 → 모델_정보.json의 범주 목록을 Flask에서도 그대로 사용
※ 약 1분 걸림
"""

import json
import os
import shutil
import sys
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = "/Users/woojin/ML project"
DATA_DIR = f"{ROOT}/data/model_input"
TUNING_PATH = f"{ROOT}/reports/tuning/최적_설정.json"
LEVEL_PATH = f"{ROOT}/reports/risk_level/위험등급_기준.json"
TEST_SCORE_PATH = f"{ROOT}/reports/final_test/test_점수.csv"
MODEL_DIR = f"{ROOT}/models"
os.makedirs(MODEL_DIR, exist_ok=True)
N_CHECK = 5000            # predictor 확인에 쓸 test 줄 수 (전부 하면 느려서 일부만)

#  1. 설정 + 데이터 불러오기 
for p, step in [(TUNING_PATH, "model/tuning.py"), (LEVEL_PATH, "model/risk_level.py")]:
    assert os.path.exists(p), f"{p} 가 없어요 → {step} 를 먼저 실행하세요"
with open(TUNING_PATH, encoding="utf-8") as f:
    tuned = json.load(f)
FEATURES, CATS = tuned["features"], tuned["categorical"]
train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"{DATA_DIR}/valid.csv", encoding="utf-8-sig")
test = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig")
with open(f"{DATA_DIR}/info.json", encoding="utf-8") as f:
    info = json.load(f)
TARGET = info["target"]
both = pd.concat([train, valid], ignore_index=True)
print(f"최종 학습 데이터 (train + valid) {len(both):,}줄 / 변수 {len(FEATURES)}개 / 나무 {tuned['n_estimators']}개")

#  2. 최종 모델 학습 + 저장 
model = lgb.LGBMClassifier(objective="binary", n_estimators=tuned["n_estimators"], subsample_freq=1,
                           random_state=42, verbose=-1, **tuned["params"])
model.fit(both[FEATURES], both[TARGET], categorical_feature=CATS)
model.booster_.save_model(f"{MODEL_DIR}/사고위험_모델.txt")

#  3. 위험 등급 기준 복사 
shutil.copy(LEVEL_PATH, f"{MODEL_DIR}/위험등급_기준.json")

#  4. 이유 문장 규칙 (4-③ shap_reason.py와 같은 규칙) 
CODES = info["category_codes"]
code = {c: {n: i for i, n in enumerate(CODES[c])} for c in CATS}           # 이름 → 번호
rule = {
    "MIN_SHAP": 0.05,     # 기본 이유: 이 값보다 크게 올린 것만
    "TOP_K": 3,           # 기본 이유 최대 개수
    "TODAY_MIN": 0.0,     # 오늘 상황: 조금이라도 올렸으면 보여줌
    "요일이름": {str(v): k[0] for k, v in info["weekdays"].items()},         # "0" → "월"
    "날씨_문장": {
        "기상상태": {str(code["기상상태"]["비"]): "비가 와요", str(code["기상상태"]["눈"]): "눈이 와요"},
        "노면상태": {str(code["노면상태"]["젖음/습기"]): "노면이 젖어 있어요",
                str(code["노면상태"]["적설"]): "눈이 쌓여 있어요",
                str(code["노면상태"]["서리/결빙"]): "노면이 얼었을 수 있어요"},
    },
    # 묶음: 변수(SHAP을 더할 변수), 종류(기본/오늘), 조건(문장을 보여줄 때), 문장({변수이름} 자리에 값이 들어감)
    "묶음": {
        "과거 사고": {"변수": ["칸_작년까지사고"], "종류": "기본",
                "조건": {"type": "notna", "col": "칸_작년까지사고"},
                "문장": "최근 3년간 사고가 많았던 곳이에요 (1년 평균 {칸_작년까지사고:.0f}건)"},
        "시간대 사고": {"변수": ["칸_작년까지같은시간대사고", "발생시"], "종류": "기본",
                "조건": {"type": "always"},
                "문장": "{발생시:.0f}시 무렵에 사고가 잦은 곳이에요"},
        "사람 많음": {"변수": ["생활인구_합계"], "종류": "기본",
                "조건": {"type": "notna", "col": "생활인구_합계"},
                "문장": "지금 사람이 많은 곳이에요 (생활인구 약 {생활인구_합계:,.0f}명)"},
        "큰 도로": {"변수": ["칸_도로연장_m", "칸_최대차로수", "칸_평균제한속도_kmh", "칸_간선도로비율"], "종류": "기본",
                "조건": {"type": "notna", "col": "칸_도로연장_m"},
                "문장": "차가 많이 다니는 큰 도로예요"},
        "교차로": {"변수": ["도로_작은분류"], "종류": "기본",
                "조건": {"type": "in", "col": "도로_작은분류",
                    "values": [code["도로_작은분류"][n] for n in ["교차로안", "교차로부근", "교차로횡단보도내"]]},
                "문장": "교차로 근처예요"},
        "요일": {"변수": ["요일"], "종류": "기본",
            "조건": {"type": "always"},
            "문장": "{요일이름}요일은 사고가 잦은 요일이에요"},
        "날씨·노면": {"변수": ["기상상태", "노면상태"], "종류": "오늘",
                "조건": {"type": "weather"},
                "문장": "(날씨_문장 사용)"},
        "막힘": {"변수": ["칸_직전1시간_속도비율"], "종류": "오늘",
            "조건": {"type": "lt", "col": "칸_직전1시간_속도비율", "value": 0.95},
            "문장": "평소보다 차가 막혀요 (속도가 평소의 {칸_직전1시간_속도비율:.0%})"},
    },
}
with open(f"{MODEL_DIR}/이유문장_규칙.json", "w", encoding="utf-8") as f:
    json.dump(rule, f, ensure_ascii=False, indent=2)

#  5. 모델 정보 
test_score = None
if os.path.exists(TEST_SCORE_PATH):
    s = pd.read_csv(TEST_SCORE_PATH, encoding="utf-8-sig").iloc[-1]          # 마지막 줄 = ② 최종 모델
    test_score = {k: round(float(s[k]), 4) for k in ["PR-AUC", "ROC-AUC", "상위10%_포착률", "로그손실"]}
model_info = {"features": FEATURES, "categorical": CATS,
            "category_codes": {c: CODES[c] for c in CATS},                   # 번호 = 리스트 위치
            "weekdays": info["weekdays"], "target": TARGET,
            "params": tuned["params"], "n_estimators": tuned["n_estimators"],
            "학습_데이터": "train + valid", "학습_줄수": len(both), "학습_날짜": str(date.today()),
            "test_점수": test_score,
            "주의": "위험점수는 진짜 사고 확률이 아니라 상대 순위용 (사고 안 난 데이터 1:3으로 만듦)"}
with open(f"{MODEL_DIR}/모델_정보.json", "w", encoding="utf-8") as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print(f"저장 완료: {sorted(os.listdir(MODEL_DIR))}")

#  6. 확인 ① 다시 불러온 모델 = 학습 직후 모델? 
p_now = model.predict_proba(test[FEATURES])[:, 1]
p_load = lgb.Booster(model_file=f"{MODEL_DIR}/사고위험_모델.txt").predict(test[FEATURES])
diff = np.abs(p_now - p_load).max()
print(f"\n[확인 ①] 학습 직후 vs 다시 불러온 모델, test 예측 최대 차이: {diff:.2e}  "
    f"→ {'저장 성공 (똑같음)' if diff < 1e-9 else '차이 있음 (확인 필요)'}")

# ── 7. 확인 ② predictor.py(Flask용)로 test 일부 예측 ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import RiskModel                                              # noqa: E402

rm = RiskModel(MODEL_DIR)
sample = test.sample(N_CHECK, random_state=42).reset_index(drop=True)
res = rm.predict(sample)
grade = pd.Series([r["등급"] for r in res])
n_reason = pd.Series([len(r["이유"]) for r in res])
rain = sample["기상상태"] == code["기상상태"]["비"]
rain_today = np.mean(["[오늘] 비가 와요" in r["오늘"] for r, is_rain in zip(res, rain) if is_rain])
score_same = np.abs(np.array([r["위험점수"] for r in res]) - np.round(
    rm.booster.predict(sample[FEATURES]), 3)).max()
print(f"\n[확인 ②] predictor.py로 test {N_CHECK:,}줄 예측")
print(f"등급 비율: {grade.value_counts(normalize=True).reindex(['높음', '보통', '낮음']).round(3).to_dict()}  "
    f"(5-③ test: 높음 10%, 보통 20%, 낮음 70% 근처면 정상)")
print(f"기본 이유 개수 비율: {n_reason.value_counts(normalize=True).sort_index().round(3).to_dict()}")
print(f"높음 등급 중 기본 이유 3개: {(n_reason[grade == '높음'] == 3).mean():.1%}")
print(f"비 오는 날 {rain.sum()}줄 중 '[오늘] 비가 와요' 비율: {rain_today:.1%}  (4-③ valid에서는 67%)")
print(f"위험점수 = 모델 예측? 최대 차이 {score_same:.1e}")

# ── 8. 확인 ③ 예시 3줄 ────────────────────────────────────
prob = np.array([r["위험점수"] for r in res])
picks = {"위험 가장 높은 줄": int(np.argmax(prob)),
        "비 오는 날 중 위험 가장 높은 줄": int(np.where(rain)[0][np.argmax(prob[rain.values])]) if rain.any() else None,
        "낮음 등급 예시": int(np.where(grade == "낮음")[0][0])}
print("\n[확인 ③] 예시")
for title, i in picks.items():
    if i is None:
        continue
    r, s = res[i], sample.iloc[i]
    print(f"\n- {title}: 칸 {s['grid_id']}, {s['date']} {int(s['발생시'])}시 (실제 사고 {int(s[TARGET])})")
    print(f"  위험점수 {r['위험점수']:.3f} → 등급 {r['등급']}")
    for k, t in enumerate(r["이유"], start=1):
        print(f"  {k}. {t}")
    for t in r["오늘"]:
        print(f"  + {t}")

print(f"\n저장 위치: {MODEL_DIR}")