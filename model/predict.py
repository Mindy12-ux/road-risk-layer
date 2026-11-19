"""
5단계 - ④ 저장된 모델로 예측하기 (Flask에서 불러다 쓰는 파일)
목적: models/ 폴더의 파일 4개만 불러와서, 줄(칸·시간)마다 위험 등급 + 이유 문장을 만들기
    → 다시 학습하지 않음 (학습·저장은 model/save_model.py)

사용법 (다른 파일에서):
    from predictor import RiskModel
    rm = RiskModel()                       # models/ 폴더 불러오기 (한 번만)
    rm.predict_one({"월": 7, "요일": 2, "발생시": 18, ...})   # 변수 19개를 넣은 dict
    → {"위험점수": 0.62, "등급": "높음",
    "이유": ["최근 3년간 사고가 많았던 곳이에요 (1년 평균 12건)", ...],
    "오늘": ["[오늘] 비가 와요"]}
    rm.predict(df)                         # 여러 줄(DataFrame)을 한 번에 → 위 dict의 리스트

필요한 파일 (models/):
    사고위험_모델.txt       LightGBM 모델
    위험등급_기준.json      높음/보통 경계 확률값
    이유문장_규칙.json      이유 묶음, 조건, 문장 틀
    모델_정보.json          변수 순서, 범주 목록
※ 위험점수는 '진짜 사고 확률'이 아니라 상대 위험 순위용 점수임 (사고 안 난 데이터를 1:3으로 만들어 넣었기 때문)
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class RiskModel:
    def __init__(self, model_dir=MODEL_DIR):
        self.booster = lgb.Booster(model_file=os.path.join(model_dir, "사고위험_모델.txt"))
        self.level = _load(os.path.join(model_dir, "위험등급_기준.json"))
        self.rule = _load(os.path.join(model_dir, "이유문장_규칙.json"))
        self.info = _load(os.path.join(model_dir, "모델_정보.json"))
        self.features = self.info["features"]
        self.weekday = self.rule["요일이름"]                 # "0" → "월"

    # ── 등급 ───────────────────────────────────────────
    def _grade(self, p):
        if p >= self.level["cut_high"]:
            return "높음"
        return "보통" if p >= self.level["cut_mid"] else "낮음"

    # ── 이유 문장 ─────────────────────────────────────────
    def _cond_ok(self, cond, r):
        kind = cond["type"]
        if kind == "always":
            return True
        if kind == "notna":
            return pd.notna(r[cond["col"]])
        if kind == "lt":
            return pd.notna(r[cond["col"]]) and r[cond["col"]] < cond["value"]
        if kind == "in":
            return pd.notna(r[cond["col"]]) and int(r[cond["col"]]) in cond["values"]
        if kind == "weather":
            return self._weather_text(r) is not None
        raise ValueError(f"모르는 조건: {kind}")

    def _weather_text(self, r):
        for col in ["기상상태", "노면상태"]:                  # 비·눈이 먼저, 없으면 노면
            if pd.notna(r[col]):
                text = self.rule["날씨_문장"][col].get(str(int(r[col])))
                if text:
                    return text
        return None

    def _text(self, group, r):
        g = self.rule["묶음"][group]
        if g["조건"]["type"] == "weather":
            return self._weather_text(r)
        values = {k: v for k, v in r.items()}
        values["요일이름"] = self.weekday[str(int(r["요일"]))]
        return g["문장"].format(**values)

    def _reasons(self, r, shap_row):
        base, today = [], []
        for name, g in self.rule["묶음"].items():
            s = sum(shap_row[c] for c in g["변수"])               # 묶음 SHAP = 들어간 변수 SHAP의 합
            if not self._cond_ok(g["조건"], r):
                continue
            if g["종류"] == "기본" and s > self.rule["MIN_SHAP"]:
                base.append((s, name))
            elif g["종류"] == "오늘" and s > self.rule["TODAY_MIN"]:
                today.append((s, name))
        base = sorted(base, reverse=True)[: self.rule["TOP_K"]]
        today = sorted(today, reverse=True)
        return ([self._text(n, r) for _, n in base],
                ["[오늘] " + self._text(n, r) for _, n in today])

    # ── 예측 ───────────────────────────────────────────
    def predict(self, df):
        X = df[self.features].astype(float).reset_index(drop=True)
        prob = self.booster.predict(X)
        contrib = self.booster.predict(X, pred_contrib=True)[:, :-1]     # 마지막 열 = 기본값(평균)
        out = []
        for i in range(len(X)):
            r = X.iloc[i].to_dict()
            shap_row = dict(zip(self.features, contrib[i]))
            reasons, today = self._reasons(r, shap_row)
            out.append({"위험점수": round(float(prob[i]), 3), "등급": self._grade(prob[i]),
                        "이유": reasons, "오늘": today})
        return out

    def predict_one(self, row):
        missing = [c for c in self.features if c not in row]
        assert not missing, f"빠진 변수: {missing}"
        return self.predict(pd.DataFrame([row]))[0]

# 모델 저장 결과: 서비스에 필요한 파일 4개(모델, 등급 기준, 이유 규칙, 모델 정보)를 
# models/에 저장했고, 다시 불러와 예측해도 결과가 완전히 똑같았음. 
# predictor.py로 test 5,000줄을 돌려 보니 등급 비율은 높음 10.6% / 보통 20.9% / 낮음 68.5%로 정상이었고, 
# 높음 줄의 99.8%가 이유 3개를 받았으며 비 오는 날의 86%에 "[오늘] 비가 와요"가 붙어서, 
# Flask에서 RiskModel().predict_one()만 부르면 바로 쓸 수 있음