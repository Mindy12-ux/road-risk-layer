"""
4단계: 경로 3개 받아서 경로마다 위험 등급 매기기 (Flask에서 불러다 쓰는 파일)

사용법 (다른 파일에서):
    from route_risk import RouteRisk
    rr = RouteRisk()                                            # 모델·표·키 불러오기 (한 번만)
    out = rr.routes((126.9707, 37.5547), (127.0276, 37.4979))   # (경도, 위도) 출발 → 도착, 지금 시각
    out = rr.routes(출발, 도착, when="2026-11-18 08:00", weather="비", road_state="조금 막힘")  # 미래 시각
    → {"경로": [{"이름", "거리_km", "시간_분", "등급", "높음_비율", "보통이상_비율", "선", "위험_칸"}, ...],
    "날씨_출처", "속도_출처", "안내"}

확인용 실행 (road-risk-layer 폴더에서):
    python model/route_risk.py

경로 등급 규칙 (팀 회의 결정, 거리 기준)
경로가 지나는 칸마다 '그 칸 안에서 달린 거리'를 잼
높음 칸 거리 ÷ 전체 거리 ≥ 35.2%          → 경로 높음
(높음 + 보통) 칸 거리 ÷ 전체 거리 ≥ 35.2%   → 경로 보통
그 밖                                     → 경로 낮음
(서울 2,118칸 밖을 지나는 거리는 계산에서 뺌)

경로 3개 만드는 법
카카오모빌리티 자동차 길찾기 (추천 + 대안 경로) → 3개가 안 되면 최단거리·최소시간 경로를 더 받아서 채움
거의 같은 경로(지나는 칸이 90% 이상 같음)는 하나로 침

시간
지금 시각   : 실시간 속도비율(3단계) + Open-Meteo 날씨
미래 시각   : 사용자가 고른 도로 상황·날씨 (걸리는 시간은 카카오가 알려준 지금 기준 시간)

필요한 패키지: pandas numpy requests pyproj  (+ 같은 폴더의 predict.py, make_features.py, speed_now.py)
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_features import MODEL_DIR, FeatureMaker  # noqa: E402
from predict import RiskModel  # noqa: E402
from speed_now import SpeedNow  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ENV_FILE = os.path.join(ROOT, ".env")
KAKAO_URL = "https://apis-navi.kakaomobility.com/v1/directions"

CELL = 500
STEP = 10                 # 경로 선을 10m 조각으로 잘라 어느 칸인지 봄 (road_geo.py와 같은 간격)
N_ROUTES = 3
SAME_ROUTE = 0.9          # 지나는 칸이 90% 이상 같으면 같은 경로로 봄
HIGH_SHARE = 0.352        # 경로 등급 기준: 평생 교통사고를 한 번 이상 겪을 확률 약 35.2% (한국교통안전공단)
LL_TO_5179 = Transformer.from_crs(4326, 5179, always_xy=True)
NOTICE = "위험 등급은 과거 사고 기록으로 만든 참고 정보예요. 실제 운전은 도로 상황과 교통 법규를 따라 주세요."


def read_env(name):
    v = os.environ.get(name)
    if not v and os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith(name + "="):
                    v = line.split("=", 1)[1].strip()
    assert v, f"{name}가 없어요 (.env 확인)"
    return v


def cell_distances(lonlat):
    """경로 선 [(경도, 위도), ...] → {grid_id: 그 칸 안에서 달린 거리(m)} (지나는 순서대로)"""
    ll = np.asarray(lonlat, float)
    xs, ys = LL_TO_5179.transform(ll[:, 0], ll[:, 1])
    dx, dy = np.diff(xs), np.diff(ys)
    seg = np.hypot(dx, dy)
    ok = seg > 0
    n = np.maximum(1, np.ceil(seg[ok] / STEP).astype(int))
    i = np.repeat(np.arange(ok.sum()), n)
    start = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(n.sum()) - start + 0.5) / n[i]
    x = np.asarray(xs)[:-1][ok][i] + t * dx[ok][i]
    y = np.asarray(ys)[:-1][ok][i] + t * dy[ok][i]
    piece = (seg[ok] / n)[i]                                   # 조각 하나의 길이
    cells = pd.Series(piece).groupby(
        pd.Series([f"{a}_{b}" for a, b in zip((x // CELL).astype(int), (y // CELL).astype(int))]), sort=False).sum()
    return cells.to_dict()


class RouteRisk:
    def __init__(self):
        self.key = read_env("KAKAO_REST_KEY")
        self.fm = FeatureMaker()
        self.rm = RiskModel(MODEL_DIR)
        self.sn = SpeedNow()
        self.seoul = set(self.fm.cells)

    #  카카오 길찾기 
    def _kakao(self, origin, dest, priority, alternatives):
        params = {"origin": f"{origin[0]},{origin[1]}", "destination": f"{dest[0]},{dest[1]}",
                "priority": priority, "alternatives": str(alternatives).lower(), "summary": "false"}
        r = requests.get(KAKAO_URL, params=params, headers={"Authorization": f"KakaoAK {self.key}"}, timeout=10)
        r.raise_for_status()
        out = []
        for k, route in enumerate(r.json().get("routes", [])):
            if route.get("result_code") != 0:                   # 0 = 길찾기 성공
                continue
            line = []
            for sec in route["sections"]:
                for road in sec["roads"]:
                    v = road["vertexes"]                         # [경도, 위도, 경도, 위도, ...]
                    line += list(zip(v[0::2], v[1::2]))
            name = {"RECOMMEND": "추천", "TIME": "최소시간", "DISTANCE": "최단거리"}[priority]
            out.append({"이름": name if k == 0 else f"{name} 대안 {k}",
                        "거리_km": round(route["summary"]["distance"] / 1000, 1),
                        "시간_분": round(route["summary"]["duration"] / 60),
                        "선": line})
        return out

    def find_routes(self, origin, dest):
        """서로 다른 경로 최대 3개"""
        found = []
        for priority, alt in [("RECOMMEND", True), ("DISTANCE", False), ("TIME", False)]:
            for r in self._kakao(origin, dest, priority, alt):
                r["칸거리"] = cell_distances(r["선"])
                cells = set(r["칸거리"])
                same = any(len(cells & set(f["칸거리"])) / max(len(cells | set(f["칸거리"])), 1) >= SAME_ROUTE
                        for f in found)
                if not same:
                    found.append(r)
                if len(found) == N_ROUTES:
                    return found
        return found

    #  경로 등급 
    @staticmethod
    def route_grade(dist, grade):
        """dist {칸: 거리}, grade {칸: 등급} → (경로 등급, 높음 비율, 보통이상 비율)"""
        total = sum(d for c, d in dist.items() if c in grade)
        if total == 0:
            return "정보 없음", 0.0, 0.0
        high = sum(d for c, d in dist.items() if grade.get(c) == "높음") / total
        mid_up = sum(d for c, d in dist.items() if grade.get(c) in ("높음", "보통")) / total
        level = "높음" if high >= HIGH_SHARE else ("보통" if mid_up >= HIGH_SHARE else "낮음")
        return level, round(high, 3), round(mid_up, 3)

    def routes(self, origin, dest, when=None, weather=None, road_state=None):
        """
        origin, dest : (경도, 위도)
        when         : 없으면 지금 (실시간 속도). 미래 시각이면 road_state 사용
        weather      : None = Open-Meteo / "맑음", "비", "눈"
        road_state   : "원활", "평소", "조금 막힘", "많이 막힘" (미래 시각용)
        """
        now = when is None
        found = self.find_routes(origin, dest)
        if not found:
            return {"경로": [], "안내": "길을 찾지 못했어요. 출발지와 도착지를 확인해 주세요."}

        # 경로 3개가 지나는 서울 칸을 한꺼번에 예측
        cells = sorted({c for r in found for c in r["칸거리"] if c in self.seoul})
        ratio = self.sn.ratios(cells=cells) if now else None
        X = self.fm.make(when=when, cells=cells, weather=weather, road_state=road_state, speed_ratio=ratio)
        pred = dict(zip(X["grid_id"], self.rm.predict(X)))
        grade = {c: p["등급"] for c, p in pred.items()}

        out = []
        for r in found:
            level, high, mid_up = self.route_grade(r["칸거리"], grade)
            risky = [{"grid_id": c, "등급": grade[c], "이유": pred[c]["이유"] + pred[c]["오늘"]}
                    for c in r["칸거리"] if grade.get(c) == "높음"]           # 지나는 순서대로
            seoul_km = sum(d for c, d in r["칸거리"].items() if c in self.seoul) / 1000
            out.append({"이름": r["이름"], "거리_km": r["거리_km"], "시간_분": r["시간_분"],
                        "등급": level, "높음_비율": high, "보통이상_비율": mid_up,
                        "서울안_거리_km": round(seoul_km, 1),
                        "선": [[round(y, 6), round(x, 6)] for x, y in r["선"]],      # [위도, 경도] (카카오 지도용)
                        "칸_등급": {c: grade[c] for c in r["칸거리"] if c in grade},
                        "위험_칸": risky})
        return {"경로": out, "시각": "지금" if now else str(pd.Timestamp(when)),
                "날씨_출처": self.fm.weather_source,
                "속도_출처": f"실시간 {self.sn.last}" if now else f"도로 상황 선택: {road_state or '평소'}",
                "안내": NOTICE}


# ---------------------------------------------------------------- 확인용 (python model/route_risk.py)
def check():
    import time
    rr = RouteRisk()
    tests = {"서울역 → 강남역": ((126.9707, 37.5547), (127.0276, 37.4979)),
            "광화문 → 잠실역": ((126.9769, 37.5759), (127.1001, 37.5133)),
            "신촌역 → 여의도": ((126.9368, 37.5552), (126.9246, 37.5216))}
    for name, (o, d) in tests.items():
        t = time.time()
        res = rr.routes(o, d)
        print(f"\n[{name}] {time.time() - t:.1f}초, 날씨 {res['날씨_출처']}")
        for r in res["경로"]:
            # 확인: 칸별 거리 합 = 카카오 거리
            total = sum(rr_d for rr_d in cell_distances([(x, y) for y, x in r["선"]]).values()) / 1000
            print(f"  {r['이름']:<10} {r['거리_km']:>5}km (칸 거리 합 {total:.1f}km) {r['시간_분']:>3}분  "
                f"등급 {r['등급']}  (높음 {r['높음_비율']:.0%}, 보통 이상 {r['보통이상_비율']:.0%})  "
                f"높음 칸 {len(r['위험_칸'])}개")
        if res["경로"] and res["경로"][0]["위험_칸"]:
            c = res["경로"][0]["위험_칸"][0]
            print(f"  첫 번째 높음 칸 {c['grid_id']}: {c['이유']}")
    t = time.time()
    res = rr.routes(*tests["서울역 → 강남역"], when=pd.Timestamp(dt.datetime.now()).floor("h") + pd.Timedelta(days=1, hours=-10),
                    weather="비", road_state="많이 막힘")
    print(f"\n[미래: 내일 {pd.Timestamp(res['시각']):%H시} + 비 + 많이 막힘] {time.time() - t:.1f}초, {res['속도_출처']}")
    for r in res["경로"]:
        print(f"  {r['이름']:<10} 등급 {r['등급']}  (높음 {r['높음_비율']:.0%}, 보통 이상 {r['보통이상_비율']:.0%})")


if __name__ == "__main__":
    check()

# 4단계 결과: 카카오 경로를 칸으로 나눈 거리가 실제 거리와 정확히 맞고, 경로마다 1~2.5초 만에 등급·이유가 계산됨
# 퇴근 시간이라 6개 경로가 모두 높음으로 나와서, 높음 비율이 가장 낮은 경로에 '가장 안전한 경로' 표시를 붙이는 게 필요함