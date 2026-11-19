"""서울 교통사고 위험 지도 + 경로 안내 (Flask가 HTML을 직접 보여줌, 지도는 카카오맵)

실행 (road-risk-layer 폴더에서):
    python web/app.py
    → 브라우저에서 http://localhost:5000

필요한 것
  - .env (road-risk-layer 폴더):  KAKAO_REST_KEY (길찾기), KAKAO_JS_KEY (지도 화면), KAKAO_MAP_REST_KEY (장소 검색),
                                 SEOUL_API_KEY (실시간 속도)   ※ 지도·검색 키는 카카오맵이 켜진 앱의 키여야 함
  - 카카오 디벨로퍼스 앱 → 플랫폼 키 → JavaScript 키 → JavaScript SDK 도메인에 http://localhost:5000 등록
  - 패키지: flask lightgbm pandas numpy requests pyproj

화면 기능 (팀 웹기능 체크리스트 결정)
  1 출발지·도착지 → 경로 3개, 경로마다 위험 등급·소요시간을 지도에 표시
  2 주소·장소 검색, 내 현재 위치(GPS)
  3 시간 슬라이더 (오늘 ~ 며칠 뒤)
  4 날씨: Open-Meteo 자동 / 사용자가 직접 선택
  5 지금 시각은 서울시 실시간 속도 API
  6 위험 지도, 이유 카드, 오늘의 위험 TOP10 칸, 경로 경고
  7 등급만 보여 줌 (점수 숫자 X)   8 낮음 = 색 없음, 클릭하면 "비교적 안전해요"
  9 높음 빨강 / 보통 노랑 / 낮음 없음   10 안내(면책) 문구
API (화면이 부르는 주소)
  GET /api/grid                       칸 2,118개 모양 (geojson)
  GET /api/risk?when=&weather=&road=  칸별 등급 (+ TOP10)
  GET /api/cell?lat=&lng=&when=&weather=&road=   클릭한 칸의 등급·이유
  GET /api/routes?o=경도,위도&d=경도,위도&when=&weather=&road=   경로 3개 + 등급
  GET /api/search?q=강남역             장소 검색 (카카오 로컬)
  GET /performance                    모델 성능 페이지 (① 고정 성능 ② 실시간 상태 ③ 나중에 채점)
  GET /api/performance/live           ② 실시간 상태 (등급 비율, 데이터 출처, 입력 드리프트, 응답 시간)
경로마다 '카메라' 목록도 붙음: 경로 선 25m 안의 단속카메라 (web/data/cameras_seoul.csv,
  공공데이터포털 전국무인교통단속카메라표준데이터 서울 4,491대: 과속·신호·구간단속·주정차)
경로마다 '보호구역' 목록도 붙음: 경로가 시설 200m 안을 지나는 어린이보호구역 (web/data/child_zones_seoul.csv,
  공공데이터포털 전국어린이보호구역표준데이터 서울 1,601곳)
"""
import datetime as dt
import json
import math
import os
import re
import sys
import threading
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from pyproj import Transformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

# LightGBM 이 Windows 에서 한글 경로(사고위험_모델.txt)를 못 여는 문제 우회: 파일을 읽어서 문자열로 넘김
_Booster = lgb.Booster


class _Booster2(_Booster):
    def __init__(self, *args, model_file=None, **kw):
        if model_file is not None and not str(model_file).isascii():
            with open(model_file, encoding="utf-8") as f:
                kw["model_str"] = f.read()
            model_file = None
        super().__init__(*args, model_file=model_file, **kw)


lgb.Booster = _Booster2

from make_features import MODEL_DIR, FeatureMaker  # noqa: E402
from predict import RiskModel  # noqa: E402
from route_risk import NOTICE, RouteRisk, read_env  # noqa: E402
from route_risk import cell_distances as rr_cell_distances  # noqa: E402
from speed_now import SpeedNow  # noqa: E402

app = Flask(__name__)
app.json.ensure_ascii = False

def env_or(name, default=""):
    try:
        return read_env(name)
    except AssertionError:
        return default


KAKAO_JS = env_or("KAKAO_JS_KEY")                                            # 지도 화면
KAKAO_SEARCH = env_or("KAKAO_MAP_REST_KEY") or read_env("KAKAO_REST_KEY")   # 장소 검색 (카카오맵 켜진 앱 키)
LL_TO_5179 = Transformer.from_crs(4326, 5179, always_xy=True)

fm = FeatureMaker()
rm = RiskModel(MODEL_DIR)
sn = SpeedNow()
rr = RouteRisk()

# ---- 길안내용: 카카오 응답의 안내 지점(guides)도 같이 받기 (팀 코드는 그대로 두고 감쌈) ----
_local = threading.local()
KAKAO_URL = "https://apis-navi.kakaomobility.com/v1/directions"
KAKAO_FUTURE_URL = "https://apis-navi.kakaomobility.com/v1/future/directions"   # 미래 운행 정보 길찾기 (출발 시각 기준 예상 교통)
KAKAO_ROUTE_KEY = read_env("KAKAO_REST_KEY")


def _kakao_with_guides(origin, dest, priority, alternatives, waypoint=None):
    """route_risk.RouteRisk._kakao 와 같은 결과 + 경로마다 '안내' (턴바이턴 안내 지점). waypoint=(경도, 위도) 면 거쳐 가기"""
    params = {"origin": f"{origin[0]},{origin[1]}", "destination": f"{dest[0]},{dest[1]}",
              "priority": priority, "alternatives": str(alternatives).lower(), "summary": "false"}
    if waypoint:
        params["waypoints"] = f"{waypoint[0]},{waypoint[1]}"
    depart = getattr(_local, "depart", None)          # 미래 출발이면 그 시각의 예상 교통으로 경로·시간을 받음
    url = KAKAO_FUTURE_URL if depart else KAKAO_URL
    if depart:
        params["departure_time"] = depart             # yyyyMMddHHmm
    r = requests.get(url, params=params, headers={"Authorization": f"KakaoAK {KAKAO_ROUTE_KEY}"}, timeout=10)
    r.raise_for_status()
    out = []
    for k, route in enumerate(r.json().get("routes", [])):
        if route.get("result_code") != 0:
            continue
        line, guides, names = [], [], {}
        for sec in route["sections"]:
            for road in sec["roads"]:
                v = road["vertexes"]
                line += list(zip(v[0::2], v[1::2]))
                for x, y in zip(v[0::2], v[1::2]):          # 칸마다 경로가 지나는 도로 이름 세기 (고속·고가 구간 판단용)
                    c = _gid(y, x)
                    names.setdefault(c, {}).setdefault(road.get("name", ""), 0)
                    names[c][road.get("name", "")] += 1
            for g in sec.get("guides", []):
                guides.append({"x": g["x"], "y": g["y"], "type": g["type"], "안내": g["guidance"],
                               "이름": g.get("name", "")})
        name = {"RECOMMEND": "추천", "TIME": "최소시간", "DISTANCE": "최단거리"}[priority]
        out.append({"이름": name if k == 0 else f"{name} 대안 {k}",
                    "거리_km": round(route["summary"]["distance"] / 1000, 1),
                    "시간_분": round(route["summary"]["duration"] / 60),
                    "선": line, "안내": guides,
                    "칸_도로명": {c: max(n, key=n.get) for c, n in names.items()}})
    return out


_find_routes = rr.find_routes


MAX_DETOUR = 1.5          # 우회 경로는 가장 빠른 경로 시간의 1.5배까지만


def _same(a, b):
    a, b = set(a["칸거리"]), set(b["칸거리"])
    return len(a & b) / max(len(a | b), 1) >= 0.9          # route_risk 와 같은 기준 (지나는 칸 90% 이상 같음)


def _detours(origin, dest, found):
    """카카오가 준 경로가 3개보다 적으면, 출발·도착 사이 옆쪽 지점을 거쳐 가는 우회 경로를 더 받음"""
    ox, oy = LL_TO_5179.transform(*origin)
    dx_, dy_ = LL_TO_5179.transform(*dest)
    vx, vy = dx_ - ox, dy_ - oy
    dist = max((vx * vx + vy * vy) ** 0.5, 1)
    px, py = -vy / dist, vx / dist                          # 출발→도착 방향에 수직인 방향
    off = min(max(0.2 * dist, 800), 3000)                   # 옆으로 0.8~3km
    base = min(r["시간_분"] for r in found) if found else None
    to_ll = Transformer.from_crs(5179, 4326, always_xy=True)
    cand = []
    for t in (0.5, 0.35, 0.65):                             # 가운데, 1/3, 2/3 지점의 양옆
        for side in (1, -1):
            wx, wy = ox + vx * t + px * off * side, oy + vy * t + py * off * side
            cand.append(to_ll.transform(wx, wy))
    for k, wp in enumerate(cand):
        if len(found) >= 3:
            break
        try:
            routes = _kakao_with_guides(origin, dest, "RECOMMEND", False, waypoint=wp)
        except requests.RequestException:
            continue
        for r in routes:
            r["칸거리"] = rr_cell_distances(r["선"])
            if base and r["시간_분"] > base * MAX_DETOUR:
                continue
            if any(_same(r, f) for f in found):
                continue
            r["이름"] = f"우회 {sum(1 for f in found if f['이름'].startswith('우회')) + 1}"
            found.append(r)
            break
    return found


def _find_routes_keep(origin, dest):
    found = _find_routes(origin, dest)
    if len(found) < 3:
        found = _detours(origin, dest, found)
    _local.found = found                      # routes() 결과에 안내를 붙이려고 기억
    return found


rr._kakao = _kakao_with_guides
rr.find_routes = _find_routes_keep
CELLS = set(fm.cells)
ROAD = pd.read_csv(os.path.join(ROOT, "data", "service_tables", "칸_도로명.csv"), encoding="utf-8-sig") \
    .groupby("grid_id")["도로명"].apply(lambda s: ", ".join(dict.fromkeys(s.dropna().astype(str)))[:40]).to_dict()
CENTER = fm.info[["중심_위도", "중심_경도"]].to_dict("index")

# 단속카메라 (공공데이터포털 전국무인교통단속카메라표준데이터, 서울만 → web/data/cameras_seoul.csv)
#   종류: 과속 / 신호 / 구간단속 시작 / 구간단속 끝 / 주정차
CAM = pd.read_csv(os.path.join(ROOT, "web", "data", "cameras_seoul.csv"), encoding="utf-8-sig")
CAM["x"], CAM["y"] = LL_TO_5179.transform(CAM["경도"].values, CAM["위도"].values)
CAM_NEAR_M = 25          # 경로 선에서 25m 안에 있는 카메라만 그 경로의 카메라로 봄


# 어린이보호구역 (공공데이터포털 전국어린이보호구역표준데이터, 서울 1,601곳 → web/data/child_zones_seoul.csv)
#   점 = 대상 시설(학교·유치원·어린이집) 위치. 보호구역은 보통 정문에서 300m 안이라, 경로가 시설 200m 안을 지나면 보호구역으로 봄
ZONE = pd.read_csv(os.path.join(ROOT, "web", "data", "child_zones_seoul.csv"), encoding="utf-8-sig")
ZONE["x"], ZONE["y"] = LL_TO_5179.transform(ZONE["경도"].values, ZONE["위도"].values)
ZONE_NEAR_M = 200
ZONE_LIMIT = 30          # 어린이보호구역 제한속도 (대부분 30km/h)


def near_route(points, line_latlng, max_m):
    """경로 선 [[위도, 경도], ...] 에서 max_m 안에 있는 점들 (점과 선분 사이 최단 거리)"""
    lat = np.array([p[0] for p in line_latlng])
    lng = np.array([p[1] for p in line_latlng])
    x, y = LL_TO_5179.transform(lng, lat)
    pad = max_m + 5
    c = points[(points.x >= x.min() - pad) & (points.x <= x.max() + pad) &
               (points.y >= y.min() - pad) & (points.y <= y.max() + pad)]
    if c.empty or len(x) < 2:
        return c.iloc[0:0]
    ax, ay, bx, by = x[:-1], y[:-1], x[1:], y[1:]
    dx, dy = bx - ax, by - ay
    seg2 = np.maximum(dx * dx + dy * dy, 1e-9)
    keep = []
    for idx, p in c.iterrows():
        t = np.clip(((p.x - ax) * dx + (p.y - ay) * dy) / seg2, 0, 1)
        if np.hypot(ax + t * dx - p.x, ay + t * dy - p.y).min() <= max_m:
            keep.append(idx)
    return c.loc[keep]


def cameras_on_route(line_latlng):
    """경로 선 근처(25m) 단속카메라"""
    return [{"종류": cam["종류"], "제한속도": None if pd.isna(cam["제한속도"]) else int(cam["제한속도"]),
             "장소": cam["설치장소"], "위도": cam["위도"], "경도": cam["경도"]}
            for _, cam in near_route(CAM, line_latlng, CAM_NEAR_M).iterrows()]


def route_limits(line_latlng):
    """경로가 지나는 칸 순서대로 [grid_id, 중심위도, 중심경도, 제한속도]
    제한속도 = 칸 안 도로들의 평균 제한속도(표준노드링크, 팀 road_link)를 10km/h 단위로 반올림 → 화면 표지판용"""
    out = []
    for c in rr_cell_distances([(lng, lat) for lat, lng in line_latlng]):   # 지나는 순서
        if c not in fm.info.index:
            continue
        v = fm.info.at[c, "칸_평균제한속도_kmh"]
        lim = int(round(v / 10) * 10) if pd.notna(v) and v > 0 else None
        out.append([c, CENTER[c]["중심_위도"], CENTER[c]["중심_경도"], lim])
    return out


# 도로 구간별 제한속도 (표준노드링크, web/make_road_limits.py 로 만든 표)
#   칸 평균은 골목(30)까지 섞여서 큰길에서도 30이 나올 수 있어서, 경로가 실제로 지나는 도로 구간의 제한속도를 씀
LINK_BIN = 100            # 선분 찾기용 격자 (m)
LINK_NEAR_M = 15          # 경로 점에서 15m 안의 도로 구간만
LINK_STEP_M = 20          # 경로를 20m 마다 확인
_LINK_PATH = os.path.join(ROOT, "web", "data", "road_links_seoul.csv")
LINK_SEG, LINK_IDX = None, {}
if os.path.exists(_LINK_PATH):
    _rows = []
    for spd, shape in pd.read_csv(_LINK_PATH, usecols=["제한속도", "모양"], encoding="utf-8-sig").itertuples(index=False):
        p = [tuple(map(float, t.split())) for t in shape.split(";")]
        for (x0, y0), (x1, y1) in zip(p, p[1:]):
            if x0 != x1 or y0 != y1:
                _rows.append((x0, y0, x1, y1, spd))
    LINK_SEG = np.array(_rows)
    _bx = (np.minimum(LINK_SEG[:, 0], LINK_SEG[:, 2]) // LINK_BIN).astype(int), (np.maximum(LINK_SEG[:, 0], LINK_SEG[:, 2]) // LINK_BIN).astype(int)
    _by = (np.minimum(LINK_SEG[:, 1], LINK_SEG[:, 3]) // LINK_BIN).astype(int), (np.maximum(LINK_SEG[:, 1], LINK_SEG[:, 3]) // LINK_BIN).astype(int)
    for i in range(len(LINK_SEG)):
        for bx in range(_bx[0][i], _bx[1][i] + 1):
            for by in range(_by[0][i], _by[1][i] + 1):
                LINK_IDX.setdefault((bx, by), []).append(i)
    LINK_IDX = {k: np.array(v) for k, v in LINK_IDX.items()}
    del _rows


def road_limits(line_latlng):
    """경로를 20m 마다 보면서 실제로 지나는 도로 구간의 제한속도를 찾음
    → 제한속도가 바뀌는 지점만 [위도, 경도, 제한속도] 로 돌려줌 (표가 없으면 빈 목록)
    같은 방향(45° 안)으로 놓인 가장 가까운 구간을 고름 → 위·아래로 겹친 다른 도로(골목, 고가 밑)를 덜 잡음"""
    if LINK_SEG is None or len(line_latlng) < 2:
        return []
    xs, ys = LL_TO_5179.transform([p[1] for p in line_latlng], [p[0] for p in line_latlng])
    to_ll = Transformer.from_crs(5179, 4326, always_xy=True)
    samples = []                                   # [x, y, 제한속도]
    for i in range(len(xs) - 1):
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
        L = float(np.hypot(x1 - x0, y1 - y0))
        if L == 0:
            continue
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        for t in np.arange(0, L, LINK_STEP_M):
            px, py = x0 + ux * t, y0 + uy * t
            cand = LINK_IDX.get((int(px // LINK_BIN), int(py // LINK_BIN)))
            if cand is None:
                continue
            s = LINK_SEG[cand]
            dx, dy = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
            sl = np.hypot(dx, dy)
            k = np.clip(((px - s[:, 0]) * dx + (py - s[:, 1]) * dy) / sl ** 2, 0, 1)
            d = np.hypot(s[:, 0] + k * dx - px, s[:, 1] + k * dy - py)
            cos = (dx * ux + dy * uy) / sl
            ok = (d <= LINK_NEAR_M) & (cos >= 0.7)
            if not ok.any():
                continue
            samples.append([px, py, int(s[ok][np.argmin(d[ok]), 4])])
    # 교차로·진입로에서 한두 점만 다른 값이 잡히면 표지판이 깜빡이므로, 60m(3점) 안 되는 구간은 앞 값으로 덮음
    runs = []                                      # [시작 index, 끝 index, 제한속도]
    for j, (_, _, lim) in enumerate(samples):
        if runs and runs[-1][2] == lim:
            runs[-1][1] = j
        else:
            runs.append([j, j, lim])
    kept = []
    for r in runs:
        if kept and (r[1] - r[0] + 1 < 3 or r[2] == kept[-1][2]):
            kept[-1][1] = r[1]
        else:
            kept.append(r)
    out = []
    for a, _, lim in kept:
        lng, lat = to_ll.transform(samples[a][0], samples[a][1])
        out.append([lat, lng, lim])
    return out


def zones_on_route(line_latlng):
    """경로가 지나가는 어린이보호구역 (시설 200m 안)"""
    return [{"이름": z["이름"], "시설": z["시설종류"], "제한속도": ZONE_LIMIT, "위도": z["위도"], "경도": z["경도"]}
            for _, z in near_route(ZONE, line_latlng, ZONE_NEAR_M).iterrows()]

_cache, _lock = {}, threading.Lock()


# ---- 이유 문장 다듬기 + 경로 위 위험 구간 ----


def fix_reason(t):
    """화면·음성에 보여 줄 이유 문장을 팀 요청대로 바꿈 (모델 규칙 파일은 그대로 둠)
    - 'n시 무렵에 사고가 잦은 곳이에요' → 'n시 무렵에 사고가 많은 곳이에요'
    - '지금 사람이 많은 곳이에요 (생활인구 약 N명)' → '유동인구가 많은 곳이에요'"""
    t = re.sub(r"(\d+)시 무렵에 사고가 잦은 곳이에요", r"\1시 무렵에 사고가 많은 곳이에요", t)
    if "사람이 많은 곳" in t:
        return ("[오늘] " if t.startswith("[오늘]") else "") + "유동인구가 많은 곳이에요"
    return t


def fix_reasons(lst):
    return [fix_reason(x) for x in (lst or [])]


def _gid(lat, lng):
    x, y = LL_TO_5179.transform(lng, lat)
    return f"{int(x // 500)}_{int(y // 500)}"


# 고속도로·도시고속도로·고가: 여기서는 '유동인구' 이유를 말하지 않음 (걷는 사람이 없는 길)
FAST_ROAD = re.compile(r"고속|고가|올림픽대로|강변북로|순환로|간선도로|분당수서|자동차전용|나들목|분기점|IC|JC")


def fast_cells(cell_names):
    return sorted(c for c, n in (cell_names or {}).items() if n and FAST_ROAD.search(n))


def mid_cells(grade_of, pred, fast=()):
    """경로 위 '보통' 칸 + 이유 (노란 구간 경고용)"""
    return [{"grid_id": c, "등급": "보통", "이유": pred[c]["이유"] + pred[c]["오늘"], "도로": ROAD.get(c, ""),
             "위도": CENTER[c]["중심_위도"], "경도": CENTER[c]["중심_경도"]}
            for c, g in grade_of.items() if g == "보통" and c in pred]


def risk_segments(line, grade_of, step_m=30):
    """경로 선을 30m 간격으로 잘게 나눠 칸 등급을 보고, 높음·보통이 이어지는 부분만 선 조각으로 돌려줌
    → 지도에서 500m 네모 대신 '도로 위 색 선'으로 위험을 보여 주려고"""
    pts = []
    for (a, b) in zip(line[:-1], line[1:]):
        d = math.hypot((b[0] - a[0]) * 111000, (b[1] - a[1]) * 88000)
        n = max(1, int(d // step_m))
        for k in range(n):
            t = k / n
            pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    if line:
        pts.append(tuple(line[-1]))
    segs, cur = [], None
    for la, ln in pts:
        gid = _gid(la, ln)
        g = grade_of.get(gid, "낮음")
        if cur and cur["등급"] == g and cur["grid_id"] == gid:
            cur["선"].append([la, ln])
            continue
        if cur:
            cur["선"].append([la, ln])          # 다음 조각과 끊기지 않게 이어 붙임
        cur = {"등급": g, "grid_id": gid, "선": [[la, ln]]} if g != "낮음" else None
        if cur:
            segs.append(cur)
    return [x for x in segs if len(x["선"]) > 1]


def params():
    """화면에서 온 시각·날씨·도로상황 → (when, weather, road, 실시간속도를쓰나)

    road (막힘 정도): "실시간" = 서울시 실시간 속도 API (지금 시각만)
                      "원활" / "평소" / "조금 막힘" / "많이 막힘" = 사용자가 직접 고른 값 (지금·미래 모두)
    실시간이 아니면 when 에 시각을 넣어 보냄 → 팀 코드가 실시간 대신 고른 도로 상황을 씀
    """
    when = request.args.get("when") or None
    weather = request.args.get("weather") or None          # None = Open-Meteo
    road = request.args.get("road") or "실시간"
    t = pd.Timestamp(when) if when else pd.Timestamp(dt.datetime.now())
    is_now = abs((t - pd.Timestamp(dt.datetime.now())).total_seconds()) < 3600
    live = is_now and road == "실시간"
    if not live and road == "실시간":                         # 미래 시각에는 실시간이 없음 → 평소
        road = "평소"
    when = None if live else t.floor("h")
    return when, (None if weather in (None, "auto") else weather), (None if live else road), live


def city_predict(when, weather, road, now):
    """서울 전체 칸 예측 (같은 조건은 10분 동안 재사용)"""
    hour = (pd.Timestamp(dt.datetime.now()) if now else pd.Timestamp(when)).floor("h")
    key = (str(hour), weather, road, now)
    with _lock:
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < 300:          # 5분 (화면이 5분마다 새로 받음)
            return hit[1]
    t0 = time.time()
    ratio = sn.ratios() if now else None
    X = fm.make(when=None if now else when, weather=weather, road_state=None if now else road, speed_ratio=ratio)
    pred = dict(zip(X["grid_id"], rm.predict(X)))
    for p in pred.values():
        p["이유"], p["오늘"] = fix_reasons(p["이유"]), fix_reasons(p["오늘"])
    meta = {"시각": str(hour), "날씨_출처": fm.weather_source,
            "속도_출처": f"실시간 속도 ({sn.last})" if now else f"막힘 정도 직접 선택: {road}"}
    with _lock:
        _cache[key] = (time.time(), (pred, meta))
    if now and weather is None:                          # 실시간 기본 조건일 때만 성능 페이지 상태로 기록
        record_live(hour, pred, ratio, time.time() - t0, fm.weather_source)
    return pred, meta


# ---------------------------------------------------------------- 모델 성능 페이지 (/performance)
# ① 고정 성능   web/data/performance_static.json  (python model/make_performance.py 로 만듦)
# ② 실시간 상태 GET /api/performance/live           (실시간 예측을 돌릴 때마다 아래 _live 에 기록)
# ③ 나중에 채점 web/data/performance_delayed.json + 매시 예측 저장 web/data/predictions/YYYY-MM-DD.csv
PERF_DIR = os.path.join(ROOT, "web", "data")
PRED_DIR = os.path.join(PERF_DIR, "predictions")
TRAIN_RATIO_MEDIAN = 0.97          # 학습 데이터 칸_직전1시간_속도비율 중앙값 (드리프트 비교 기준)
DRIFT_WARN, DRIFT_BAD = 0.05, 0.15  # 중앙값 차이가 이만큼 넘으면 주의 / 경고
MODEL_VERSION = str(rm.info.get("학습_날짜", "unknown"))
_live = {"updated_at": None, "latency_sec": None, "ratio_median": None, "weather": None,
         "speed": {}, "share": {}, "history": {}, "saved": set()}


def record_live(hour, pred, ratio, latency, weather_src):
    grades = pd.Series([p["등급"] for p in pred.values()])
    share = grades.value_counts(normalize=True).reindex(["높음", "보통", "낮음"], fill_value=0).round(4).to_dict()
    with _lock:
        _live.update(updated_at=dt.datetime.now().isoformat(timespec="minutes"), latency_sec=round(latency, 1),
                     ratio_median=round(float(np.median(list(ratio.values()))), 3) if ratio else None,
                     weather=weather_src, speed=dict(sn.last), share=share)
        _live["history"][str(hour)] = share["높음"]
        for k in sorted(_live["history"])[:-24]:         # 최근 24시간만
            del _live["history"][k]
    try:
        save_prediction(hour, pred)
    except OSError as e:                                  # 저장 실패해도 화면은 계속
        print(f"예측 저장 실패: {e}")


def save_prediction(hour, pred):
    """③ 나중에 채점용: 시각마다 서울 전체 칸의 점수·등급을 하루 1파일에 한 번씩 덧붙임"""
    stamp = f"{hour:%Y-%m-%d %H:%M}"
    if stamp in _live["saved"]:
        return
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f"{hour:%Y-%m-%d}.csv")
    if os.path.exists(path) and stamp in set(pd.read_csv(path, usecols=["datetime"], encoding="utf-8-sig")["datetime"]):
        _live["saved"].add(stamp)                          # 서버를 다시 켰을 때 같은 시각을 두 번 저장하지 않게
        return
    df = pd.DataFrame({"datetime": stamp, "grid_id": list(pred),
                       "score": [round(float(p["위험점수"]), 5) for p in pred.values()],
                       "grade": [p["등급"] for p in pred.values()], "model_version": MODEL_VERSION})
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False, encoding="utf-8-sig")
    _live["saved"].add(stamp)


def read_perf(name):
    path = os.path.join(PERF_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def drift_status(gap):
    return "bad" if gap >= DRIFT_BAD else "warn" if gap >= DRIFT_WARN else "ok"


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/app")
def index():
    return render_template("app.html", kakao_js_key=KAKAO_JS, notice=NOTICE)


@app.route("/performance")
def performance():
    return render_template("performance.html", static=read_perf("performance_static.json"),
                           delayed=read_perf("performance_delayed.json"))


@app.route("/api/performance/static")
def performance_static():
    return jsonify(read_perf("performance_static.json") or {})


@app.route("/api/performance/live")
def performance_live():
    city_predict(None, None, None, True)                  # 5분 안에 이미 돌렸으면 저장값 사용
    with _lock:
        live = {k: v for k, v in _live.items() if k != "saved"}
        live["history"] = dict(live["history"])
    sp = live.pop("speed") or {}
    weather = live.pop("weather")
    drift = []
    if live["ratio_median"] is not None:
        drift.append({"feature": "칸_직전1시간_속도비율", "stat": "중앙값", "live": live["ratio_median"],
                      "train": TRAIN_RATIO_MEDIAN,
                      "status": drift_status(abs(live["ratio_median"] - TRAIN_RATIO_MEDIAN))})
    days = sorted(f[:-4] for f in os.listdir(PRED_DIR) if f.endswith(".csv")) if os.path.isdir(PRED_DIR) else []
    return jsonify({**live, "n_cells": len(CELLS), "model_version": MODEL_VERSION,
                    "sources": {"weather": {"name": weather, "ok": str(weather).startswith("Open-Meteo")},
                                "speed": {k: sp.get(k) for k in ["구간", "도로명", "없음", "부른_구간",
                                                                  "속도_못받은_구간", "걸린_초"]}},
                    "drift": drift, "saved_days": len(days), "first_day": days[0] if days else None})


@app.route("/api/grid")
def grid():
    with open(os.path.join(ROOT, "data", "service_tables", "칸_모양.geojson"), encoding="utf-8") as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.route("/api/risk")
def risk():
    when, weather, road, now = params()
    pred, meta = city_predict(when, weather, road, now)
    grades = {c: p["등급"] for c, p in pred.items() if p["등급"] != "낮음"}
    top = sorted(pred.items(), key=lambda kv: -kv[1]["위험점수"])[:10]          # 순위는 점수로, 화면엔 등급만
    top10 = [{"grid_id": c, "등급": p["등급"], "도로": ROAD.get(c, ""), "이유": p["이유"][:1],
              "위도": CENTER[c]["중심_위도"], "경도": CENTER[c]["중심_경도"]} for c, p in top]
    count = pd.Series([p["등급"] for p in pred.values()]).value_counts().to_dict()
    return jsonify({"등급": grades, "top10": top10, "개수": count, **meta, "안내": NOTICE})


@app.route("/api/cell")
def cell():
    lat, lng = float(request.args["lat"]), float(request.args["lng"])
    x, y = LL_TO_5179.transform(lng, lat)
    gid = f"{int(x // 500)}_{int(y // 500)}"
    if gid not in CELLS:
        return jsonify({"grid_id": gid, "서울": False, "안내": "서울 밖이라 위험 정보가 없어요."})
    when, weather, road, now = params()
    pred, meta = city_predict(when, weather, road, now)
    p = pred[gid]
    msg = "비교적 안전해요" if p["등급"] == "낮음" else None
    return jsonify({"grid_id": gid, "서울": True, "등급": p["등급"], "이유": p["이유"], "오늘": p["오늘"],
                    "도로": ROAD.get(gid, ""), "한마디": msg, **meta})


@app.route("/api/routes")
def routes():
    o = tuple(map(float, request.args["o"].split(",")))
    d = tuple(map(float, request.args["d"].split(",")))
    when, weather, road, now = params()
    _local.found = []
    _local.depart = None
    dep = request.args.get("depart")                  # 화면에서 고른 출발 시각 (5분 단위)
    if dep:
        t = pd.Timestamp(dep)
        if t > pd.Timestamp(dt.datetime.now()) + pd.Timedelta(minutes=5):
            _local.depart = t.strftime("%Y%m%d%H%M")
    pred, _ = city_predict(when, weather, road, now)          # '보통' 칸 이유를 꺼내 쓰려고 (5분 저장값)
    try:
        res = rr.routes(o, d, when=when, weather=weather, road_state=road)   # when=None 이면 실시간 속도
    finally:
        depart, _local.depart = _local.depart, None
    res["길찾기_기준"] = (f"카카오 미래 길찾기 · {pd.Timestamp(depart).strftime('%m/%d %H:%M')} 출발 예상 교통"
                     if depart else "카카오 실시간 길찾기 · 지금 교통")
    for r, f in zip(res["경로"], getattr(_local, "found", [])):            # routes() 는 찾은 순서 그대로 돌려줌
        r["안내"] = f.get("안내", [])
        r["고속_칸"] = fast_cells(f.get("칸_도로명"))
    for r in res["경로"]:
        r["카메라"] = cameras_on_route(r["선"])
        r["보호구역"] = zones_on_route(r["선"])
        r["칸_제한속도"] = route_limits(r["선"])
        r["도로_제한속도"] = road_limits(r["선"])
        r["위험_구간"] = risk_segments(r["선"], r.get("칸_등급", {}))
        r["보통_칸"] = mid_cells(r.get("칸_등급", {}), pred)
        for c in r["보통_칸"]:
            c["이유"] = fix_reasons(c["이유"])
        for c in r["위험_칸"]:
            c["이유"] = fix_reasons(c.get("이유"))
    if res["경로"]:                               # 높음 비율이 가장 낮은 경로 = 가장 안전한 경로
        best = min(range(len(res["경로"])), key=lambda i: (res["경로"][i]["높음_비율"], res["경로"][i]["보통이상_비율"]))
        for i, r in enumerate(res["경로"]):
            r["가장안전"] = i == best and len(res["경로"]) > 1
            r["위험_칸"] = [dict(c, 도로=ROAD.get(c["grid_id"], ""), 위도=CENTER[c["grid_id"]]["중심_위도"],
                              경도=CENTER[c["grid_id"]]["중심_경도"]) for c in r["위험_칸"]]
    return jsonify(res)


@app.route("/api/route_rescore", methods=["POST"])
def route_rescore():
    """길안내 중: 지금 가는 경로 선은 그대로 두고, 바뀐 날씨·막힘·시각으로 경로 위 칸의 위험도만 다시 계산
    body: {"선": [[위도, 경도], ...]}   query: when, weather, road (다른 API와 같음)"""
    line = request.get_json(force=True)["선"]
    when, weather, road, now = params()
    pred, meta = city_predict(when, weather, road, now)              # 서울 전체 예측(5분 저장)에서 꺼내 씀
    dist = rr_cell_distances([(lng, lat) for lat, lng in line])
    grade = {c: pred[c]["등급"] for c in dist if c in pred}
    level, high, mid_up = rr.route_grade(dist, grade)
    risky = [{"grid_id": c, "등급": "높음", "이유": pred[c]["이유"] + pred[c]["오늘"], "도로": ROAD.get(c, ""),
              "위도": CENTER[c]["중심_위도"], "경도": CENTER[c]["중심_경도"]}
             for c in dist if grade.get(c) == "높음"]
    return jsonify({"등급": level, "높음_비율": high, "보통이상_비율": mid_up, "위험_칸": risky,
                    "위험_구간": risk_segments(line, grade), "보통_칸": mid_cells(grade, pred), **meta})


@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    r = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json",
                     params={"query": q, "size": 7, "x": 126.978, "y": 37.5665, "sort": "accuracy"},
                     headers={"Authorization": f"KakaoAK {KAKAO_SEARCH}"}, timeout=8)
    if not r.ok:
        return jsonify({"error": f"장소 검색 실패 ({r.status_code})"}), 502
    return jsonify([{"이름": p["place_name"], "주소": p.get("road_address_name") or p.get("address_name"),
                     "경도": float(p["x"]), "위도": float(p["y"])} for p in r.json().get("documents", [])])


if __name__ == "__main__":
    print("http://localhost:5000  (첫 위험 지도는 실시간 속도를 받느라 15초쯤 걸려요)")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
