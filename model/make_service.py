"""
1단계: 서비스용 표 만들기 (웹에서 빠르게 꺼내 쓸 값들을 미리 계산해서 저장)

사용법 (road-risk-layer 폴더에서):
    python model/make_service.py 경로/topis_speed
    (topis_speed = TOPIS "YYYY년 M월 서울시 차량통행속도.xlsx" 엑셀 폴더. 저장소에는 없어서 표를 다시 만들 때만 실행)
    팀원은 실행할 필요 없이 dvc pull data/service_tables.dvc 로 결과 표를 받으면 됨

왜 필요한가?
모델은 변수 19개를 넣어야 위험 점수를 줌. 그중 '칸마다 정해진 값'(도로 길이, 과거 사고 등)과
'평소 값'(평소 속도, 평소 사람 수)은 웹에서 매번 계산하면 느리니까 표로 미리 만들어 둠.
지금 날씨·지금 속도·지금 시각처럼 '그때그때 바뀌는 값'은 2단계(make_features)에서 붙임.

입력:
data/raw/taas_accidents.csv              과거 사고 기록 (TAAS)
(실행할 때 넣는 폴더)/YYYY년 M월 서울시 차량통행속도.xlsx  TOPIS 통행속도 (평소 속도용)
model/results/model_input/train.csv, test.csv                       도로 상황 비율 / 검증용
data/features/road_link.csv, pop.csv        팀원 변수 (도로 구조, 생활인구)
data/service/topis/TOPIS_구간위치.csv       TOPIS 구간 → 위치
data/service/moct/MOCT_LINK.shp             표준노드링크 (칸 → 도로명)

출력 (data/service_tables/):
칸_정보.csv          칸마다 고정 값 (도로 구조, 과거 사고, 시간대별 과거 사고, 도로 분류, 중심 좌표)
칸_생활인구.csv       칸 × 주말여부 × 시각 → 평소 생활인구
칸_평소속도.csv       칸 × 월 × 시각 → 평소 속도
구간_평소속도.csv     TOPIS 구간 × 월 × 시각 → 평소 속도 (3단계 실시간 속도 비율 계산용)
도로명_평소속도.csv   도로명 × 월 × 시각 → 평소 속도 (구간 위치를 못 찾은 칸의 예비용)
칸_구간.csv          칸 → 그 칸을 지나는 TOPIS 구간
칸_도로명.csv        칸 → 그 칸을 지나는 도로명
구간_목록.csv        TOPIS 구간 번호 · 도로명 · 위치를 찾았는지 (실시간 API 부를 목록)
칸_모양.geojson      칸 네모 모양 (지도 색칠용, 위도·경도)
도로상황.json        미래 시각용 '도로 상황' 선택지 → 속도비율 값
표_정보.json         어떤 기준으로 만들었는지 기록

필요한 패키지: pandas numpy pyproj geopandas python-calamine
"""

import json
import os
import re
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEAM = os.path.join(ROOT, "data")
RAW_ACC = os.path.join(TEAM, "raw", "taas_accidents.csv")
TOPIS_DIR = None   # 실행할 때 넣는 폴더 (main에서 정함)
TRAIN = os.path.join(ROOT, "model", "results", "model_input", "train.csv")
TEST = os.path.join(ROOT, "model", "results", "model_input", "test.csv")
ROAD_LINK = os.path.join(TEAM, "features", "road_link.csv")
POP = os.path.join(TEAM, "features", "pop.csv")
SEG_GEO = os.path.join(TEAM, "service", "topis", "TOPIS_구간위치.csv")
MOCT = os.path.join(TEAM, "service", "moct", "MOCT_LINK.shp")
OUT = os.path.join(ROOT, "data", "service_tables")

SERVICE_YEAR = 2026   # 서비스하는 해 → 과거 사고는 직전 3년(2023~2025)
SPEED_YEAR = 2025     # 평소 속도는 가장 최근 1년 TOPIS (같은 달·같은 시각 평균)
CELL = 500
HIST_YEARS = 3        # preprocess.py와 같음
SMOOTH = 5            # preprocess.py와 같음 (과거 사고 비율 스무딩)
POP_SMOOTH = 5        # 생활인구 평균 스무딩 (줄이 적은 칸은 더 넓은 평균 쪽으로 당김)
STEP = 10             # 선을 10m 간격으로 잘라 어느 칸을 지나는지 봄 (road_geo.py와 같음)
ROAD_TYPES = ["기타", "교차로안", "교차로부근", "교차로횡단보도내", "교량", "터널", "고가도로위", "지하차도(도로)내", "미분류"]

LL_TO_5179 = Transformer.from_crs(4326, 5179, always_xy=True)
TO_LL = Transformer.from_crs(5179, 4326, always_xy=True)
TM_TO_5179 = Transformer.from_crs(5186, 5179, always_xy=True)


def read(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False, encoding="utf-8-sig")
    print(f"   저장: {name}  ({len(df):,}줄)")


def norm_name(x):
    """도로명 비교용 (공백·괄호·점 제거) — road_geo.py와 같음"""
    return re.sub(r"[\s\(\)（）·.]", "", str(x or ""))


def line_cells(xs, ys):
    """선(5179 좌표)을 10m 조각으로 잘라 지나는 칸 번호 모음 — road_geo.py pieces()와 같은 방식"""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dx, dy = np.diff(xs), np.diff(ys)
    seg = np.hypot(dx, dy)
    ok = seg > 0
    if not ok.any():
        return set()
    n = np.maximum(1, np.ceil(seg[ok] / STEP).astype(int))
    i = np.repeat(np.arange(ok.sum()), n)
    start = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(n.sum()) - start + 0.5) / n[i]
    x = xs[:-1][ok][i] + t * dx[ok][i]
    y = ys[:-1][ok][i] + t * dy[ok][i]
    return {f"{a}_{b}" for a, b in zip((x // CELL).astype(int), (y // CELL).astype(int))}


# ---------------------------------------------------------------- 과거 사고 (preprocess.py add_history와 같은 계산)
def load_raw():
    raw = read(RAW_ACC, dtype=str)
    raw["연도"] = raw["연도"].astype(int)
    raw["시간대"] = raw["발생시"].astype(int) // 3
    raw["grid_id"] = ((raw["X_UTMK"].astype(float) // CELL).astype(int).astype(str) + "_" +
                      (raw["Y_UTMK"].astype(float) // CELL).astype(int).astype(str))
    two = ["이륜", "원동기"]
    raw["이륜차"] = raw["가해차종"].isin(two) | raw["피해차종"].isin(two)
    raw["보행자"] = raw["사고유형"].str.startswith("차대사람")
    raw["도로_작은분류"] = raw["도로형태"].str.split(" - ", n=1).str[1]
    return raw


def history(raw, cells, year):
    """그 해(year) 기준 직전 3년 사고 기록 → 칸별 값. 미래 기록은 안 봄."""
    past = raw[(raw["연도"] < year) & (raw["연도"] >= year - HIST_YEARS)]
    n_years = past["연도"].nunique()
    by = past.groupby("grid_id")
    count = by.size()
    h = pd.DataFrame(index=pd.Index(cells, name="grid_id"))
    h["칸_작년까지사고"] = (count / n_years).reindex(h.index).fillna(0)
    band = (past.groupby(["grid_id", "시간대"]).size() / n_years).unstack(fill_value=0)
    for b in range(8):   # 시간대 0 = 0~2시, 1 = 3~5시, ... 7 = 21~23시
        h[f"같은시간대사고_{b}"] = band.get(b, pd.Series(dtype=float)).reindex(h.index).fillna(0)
    for col, flag in [("칸_과거이륜차비율", "이륜차"), ("칸_과거보행자비율", "보행자")]:
        seoul = past[flag].mean()
        ratio = (by[flag].sum() + seoul * SMOOTH) / (count + SMOOTH)
        h[col] = ratio.reindex(h.index).fillna(seoul)
    return h


def road_type(raw, cells):
    """도로_작은분류: 학습 때 '사고 안 난 줄'은 그 칸에서 사고 났던 지점의 도로형태를 무작위로 골랐음.
    → 서비스에서는 그 칸 전체 기간 사고 지점들의 도로형태 비율과 가장 흔한 값을 저장."""
    r = raw[raw["도로_작은분류"].isin(ROAD_TYPES)]
    share = pd.crosstab(r["grid_id"], r["도로_작은분류"], normalize="index").reindex(
        index=cells, columns=ROAD_TYPES, fill_value=0)
    share = share.fillna(0)
    out = pd.DataFrame(index=share.index)
    out["도로_작은분류"] = share.idxmax(axis=1).map({n: i for i, n in enumerate(ROAD_TYPES)})
    for n in ROAD_TYPES:
        out[f"도로분류비율_{n}"] = share[n].round(4)
    return out


# ---------------------------------------------------------------- 생활인구
def pop_table(cells):
    """칸 × 주말여부 × 시각 평균. 줄이 적은 칸은 (칸 × 시각) 평균 쪽으로, 그것도 적으면 (칸 평균 × 서울 시간 모양) 쪽으로 당김."""
    p = read(POP, usecols=["date", "발생시", "grid_id", "생활인구_합계"]).dropna()
    p["주말"] = (pd.to_datetime(p["date"]).dt.day_of_week >= 5).astype(int)
    seoul_shape = p.groupby("발생시")["생활인구_합계"].mean() / p["생활인구_합계"].mean()
    cell_mean = p.groupby("grid_id")["생활인구_합계"].mean()

    full = pd.MultiIndex.from_product([cells, [0, 1], range(24)], names=["grid_id", "주말", "발생시"]).to_frame(index=False)
    prior1 = cell_mean.reindex(full["grid_id"]).values * seoul_shape.reindex(full["발생시"]).values
    ch = p.groupby(["grid_id", "발생시"])["생활인구_합계"].agg(["sum", "count"])
    k = pd.MultiIndex.from_frame(full[["grid_id", "발생시"]])
    s1, n1 = ch["sum"].reindex(k).fillna(0).values, ch["count"].reindex(k).fillna(0).values
    prior2 = (s1 + POP_SMOOTH * prior1) / (n1 + POP_SMOOTH)
    cwh = p.groupby(["grid_id", "주말", "발생시"])["생활인구_합계"].agg(["sum", "count"])
    k = pd.MultiIndex.from_frame(full[["grid_id", "주말", "발생시"]])
    s2, n2 = cwh["sum"].reindex(k).fillna(0).values, cwh["count"].reindex(k).fillna(0).values
    full["생활인구_합계"] = np.round((s2 + POP_SMOOTH * prior2) / (n2 + POP_SMOOTH), 2)
    full["모인_줄수"] = n2.astype(int)
    return full


# ---------------------------------------------------------------- 평소 속도 (road_speed.py와 같은 방식, 같은 달·같은 시각 평균)
def topis_usual(year):
    """TOPIS 1년치 → 구간 × 월 × 시각, 도로명 × 월 × 시각 평소 속도 + 구간 목록"""
    hours = [f"~{h + 1:02d}시" for h in range(24)]   # '~01시' = 0~1시 평균 → 0시
    seg_parts, road_parts, names = [], [], []
    for m in range(1, 13):
        path = os.path.join(TOPIS_DIR, f"{year}년 {m}월 서울시 차량통행속도.xlsx")
        d = pd.read_excel(path, engine="calamine", dtype={"링크아이디": str})
        v = d[hours].apply(pd.to_numeric, errors="coerce")
        v = v.where(v > 0)                               # 0 이하 = 빈 값 (road_speed.py fnum과 같음)
        v.columns = range(24)
        v["링크아이디"], v["일자"] = d["링크아이디"].astype(str), d["일자"].astype(str)
        v["도로명"] = d["도로명"].map(norm_name)
        # 구간: 그 달 날짜들의 평균
        seg_parts.append(v.groupby("링크아이디")[list(range(24))].mean().assign(월=m))
        # 도로명: 날짜마다 도로 구간들 평균 → 그 달 날짜들의 평균
        daily = v.groupby(["도로명", "일자"])[list(range(24))].mean()
        road_parts.append(daily.groupby("도로명").mean().assign(월=m))
        names.append(d[["링크아이디", "도로명", "시점명", "종점명", "방향"]].astype({"링크아이디": str}))
        print(f"   TOPIS {year}년 {m}월 읽음 ({len(d):,}줄)")
    seg = pd.concat(seg_parts).reset_index()
    road = pd.concat(road_parts).reset_index()
    links = pd.concat(names).drop_duplicates("링크아이디", keep="last")
    links["도로명_비교용"] = links["도로명"].map(norm_name)
    return seg, road, links


def cell_links(cells):
    """칸 → 지나는 TOPIS 구간 (TOPIS_구간위치.csv 경로)"""
    g = read(SEG_GEO, dtype={"링크아이디": str})
    rows = []
    for lid, path in zip(g["링크아이디"], g["경로(위도 경도;...)"]):
        if not isinstance(path, str) or not path:
            continue
        ll = [tuple(map(float, q.split())) for q in path.split(";")]
        xs, ys = LL_TO_5179.transform([b for _, b in ll], [a for a, _ in ll])
        rows += [(c, lid) for c in line_cells(xs, ys) if c in cells]
    return pd.DataFrame(rows, columns=["grid_id", "링크아이디"]).drop_duplicates().sort_values(["grid_id", "링크아이디"], ignore_index=True), g


def cell_roads(cells):
    """칸 → 지나는 도로명 (표준노드링크). road_geo.py read_links와 같이 선 가운데 점이 서울 근처인 것만."""
    links = gpd.read_file(MOCT, columns=["ROAD_NAME"], bbox=(170000, 520000, 230000, 570000), encoding="cp949")
    rows = []
    for name, geom in zip(links["ROAD_NAME"], links.geometry):
        name = norm_name(name)
        if not name or geom is None:
            continue
        lines = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        pts = [p for ln in lines for p in ln.coords]
        mx, my = pts[len(pts) // 2][:2]
        if not (170000 <= mx <= 230000 and 520000 <= my <= 570000):
            continue
        for ln in lines:
            c = np.asarray(ln.coords)
            xs, ys = TM_TO_5179.transform(c[:, 0], c[:, 1])
            rows += [(g, name) for g in line_cells(xs, ys) if g in cells]
    return pd.DataFrame(rows, columns=["grid_id", "도로명"]).drop_duplicates().sort_values(["grid_id", "도로명"], ignore_index=True)


def cell_usual(cells, c2seg, c2road, seg, road):
    """칸 × 월 × 시각 평소 속도: 지나는 구간 평균(출처 1) → 없으면 도로명 평균(출처 2) → 없으면 빈 값(0)"""
    H = list(range(24))
    a = c2seg.merge(seg, on="링크아이디").groupby(["grid_id", "월"])[H].mean()
    b = c2road.merge(road, on="도로명").groupby(["grid_id", "월"])[H].mean()
    full = pd.MultiIndex.from_product([cells, range(1, 13)], names=["grid_id", "월"])
    a, b = a.reindex(full), b.reindex(full)
    val = a.where(a.notna(), b)
    src = pd.DataFrame(np.where(a.notna(), 1, np.where(b.notna(), 2, 0)), index=full, columns=H)
    out = val.stack(future_stack=True).rename("칸_평소속도_kmh").round(1).to_frame()
    out["출처"] = src.stack()
    out.index.names = ["grid_id", "월", "발생시"]
    return out.reset_index()


def cell_shapes(cells):
    feats = []
    for g in cells:
        cx, cy = map(int, g.split("_"))
        xs = np.array([cx, cx + 1, cx + 1, cx, cx]) * CELL
        ys = np.array([cy, cy, cy + 1, cy + 1, cy]) * CELL
        lon, lat = TO_LL.transform(xs, ys)
        feats.append({"type": "Feature", "properties": {"grid_id": g},
                    "geometry": {"type": "Polygon", "coordinates": [[[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]]}})
    return {"type": "FeatureCollection", "features": feats}


def main():
    global TOPIS_DIR
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    TOPIS_DIR = sys.argv[1]
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rl = read(ROAD_LINK)
    cells = sorted(rl["grid_id"])
    cellset = set(cells)
    print(f"0) 서울 칸 {len(cells):,}개")

    # ---------------------------------------------------- 1) 칸_정보.csv
    raw = load_raw()
    print(f"1) 칸 고정 값 (과거 사고 = {SERVICE_YEAR - HIST_YEARS}~{SERVICE_YEAR - 1}년)")
    info = rl.set_index("grid_id")[["칸_도로연장_m", "칸_평균제한속도_kmh", "칸_최대차로수", "칸_간선도로비율"]]
    info = info.join(history(raw, cells, SERVICE_YEAR)).join(road_type(raw, cells))
    cx = info.index.str.split("_").str[0].astype(int)
    cy = info.index.str.split("_").str[1].astype(int)
    lon, lat = TO_LL.transform((cx + 0.5) * CELL, (cy + 0.5) * CELL)
    info.insert(0, "중심_경도", np.round(lon, 6))
    info.insert(0, "중심_위도", np.round(lat, 6))
    save(info.reset_index(), "칸_정보.csv")

    # 검증: 같은 함수로 2025년 기준을 만들어 test.csv 2025년 줄과 비교 → 학습 때와 같은 계산인지
    t = read(TEST, usecols=["grid_id", "연도", "발생시", "도로_작은분류", "칸_작년까지사고",
                            "칸_작년까지같은시간대사고", "칸_과거이륜차비율", "칸_과거보행자비율"])
    t = t[t["연도"] == 2025]
    chk = history(raw, cells, 2025).reindex(t["grid_id"])
    same = {"칸_작년까지사고": np.isclose(chk["칸_작년까지사고"].values, t["칸_작년까지사고"]).mean(),
            "칸_작년까지같은시간대사고": np.isclose(
                chk.values[np.arange(len(t)), [chk.columns.get_loc(f"같은시간대사고_{b}") for b in t["발생시"] // 3]],
                t["칸_작년까지같은시간대사고"]).mean(),
            "칸_과거이륜차비율": np.isclose(chk["칸_과거이륜차비율"].values, t["칸_과거이륜차비율"]).mean(),
            "칸_과거보행자비율": np.isclose(chk["칸_과거보행자비율"].values, t["칸_과거보행자비율"]).mean()}
    print("   [검증] 2025년 기준으로 다시 계산 → test 값과 같은 비율: " +
        ", ".join(f"{k} {v:.1%}" for k, v in same.items()))
    print(f"   [참고] 도로_작은분류 '가장 흔한 값' 사용 → 칸의 사고 지점 중 그 값 비율 평균 "
        f"{info[[f'도로분류비율_{n}' for n in ROAD_TYPES]].max(axis=1).mean():.0%}")

    # ---------------------------------------------------- 2) 칸_생활인구.csv
    print("2) 생활인구 (칸 × 주말여부 × 시각)")
    pop = pop_table(cells)
    save(pop, "칸_생활인구.csv")
    tp = read(TEST, usecols=["grid_id", "date", "발생시", "생활인구_합계"])
    tp["주말"] = (pd.to_datetime(tp["date"]).dt.day_of_week >= 5).astype(int)
    est = tp.merge(pop, on=["grid_id", "주말", "발생시"], suffixes=("", "_표"))
    r = np.corrcoef(np.log1p(est["생활인구_합계"]), np.log1p(est["생활인구_합계_표"]))[0, 1]
    err = (est["생활인구_합계_표"] / est["생활인구_합계"].clip(lower=1) - 1).abs().median()
    print(f"   [검증] test 실제 생활인구와 비교: 상관 {r:.3f}, 오차 중앙값 {err:.0%}")

    # ---------------------------------------------------- 3) 평소 속도
    print(f"3) 평소 속도 (TOPIS {SPEED_YEAR}년, 같은 달·같은 시각 평균)")
    seg, road, links = topis_usual(SPEED_YEAR)
    c2seg, geo = cell_links(cellset)
    c2road = cell_roads(cellset)
    c2road = c2road[c2road["도로명"].isin(set(road["도로명"]))]   # TOPIS에 있는 도로명만 (road_speed.py와 같음)
    usual = cell_usual(cells, c2seg, c2road, seg, road)
    found = set(geo.loc[geo["경로(위도 경도;...)"].notna(), "링크아이디"])
    links["위치찾음"] = links["링크아이디"].isin(found).astype(int)
    save(seg.round(1), "구간_평소속도.csv")
    save(road.round(1), "도로명_평소속도.csv")
    save(c2seg, "칸_구간.csv")
    save(c2road, "칸_도로명.csv")
    save(links, "구간_목록.csv")
    save(usual, "칸_평소속도.csv")
    src = usual.groupby("grid_id")["출처"].max().value_counts().to_dict()
    print(f"   칸별 출처: 구간 {src.get(1, 0)}칸, 도로명 평균 {src.get(2, 0)}칸, 없음 {src.get(0, 0)}칸")
    rs = read(os.path.join(TEAM, "features", "road_speed.csv"), usecols=["date", "발생시", "grid_id", "칸_평소속도_kmh"])
    rs = rs[rs["date"].str.startswith(str(SPEED_YEAR))].dropna()
    rs["월"] = rs["date"].str[5:7].astype(int)
    m = rs.merge(usual, on=["grid_id", "월", "발생시"], suffixes=("_팀", "")).dropna()
    diff = (m["칸_평소속도_kmh"] - m["칸_평소속도_kmh_팀"]).abs()
    print(f"   [검증] 팀 road_speed.csv {SPEED_YEAR}년 평소속도와 비교: 차이 0.5km/h 이하 {(diff <= 0.5).mean():.1%}, "
        f"차이 중앙값 {diff.median():.2f}km/h")

    # ---------------------------------------------------- 4) 칸 모양
    with open(os.path.join(OUT, "칸_모양.geojson"), "w", encoding="utf-8") as f:
        json.dump(cell_shapes(cells), f, ensure_ascii=False)
    print(f"4) 저장: 칸_모양.geojson ({len(cells):,}칸)")

    # ---------------------------------------------------- 5) 도로 상황 (미래 시각용)
    ratio = read(TRAIN, usecols=["칸_직전1시간_속도비율"])["칸_직전1시간_속도비율"].dropna()
    q = ratio.quantile([0.05, 0.20, 0.50, 0.80])
    road_state = {"원활": float(round(q[0.80], 2)), "평소": 1.0, "조금 막힘": float(round(q[0.20], 2)), "많이 막힘": float(round(q[0.05], 2))}
    with open(os.path.join(OUT, "도로상황.json"), "w", encoding="utf-8") as f:
        json.dump({"값": road_state, "기준": "학습 데이터 속도비율의 상위 20%(원활), 하위 20%(조금 막힘), 하위 5%(많이 막힘)",
                "학습_중앙값": float(round(q[0.50], 3))}, f, ensure_ascii=False, indent=2)
    print(f"5) 도로상황.json: {road_state}  (학습 데이터 속도비율 중앙값 {q[0.50]:.3f})")

    meta = {"서비스_연도": SERVICE_YEAR, "과거사고_기간": f"{SERVICE_YEAR - HIST_YEARS}~{SERVICE_YEAR - 1}",
            "평소속도_연도": SPEED_YEAR, "칸_수": len(cells), "만든_날짜": time.strftime("%Y-%m-%d"),
            "같은시간대사고": "같은시간대사고_0~7 (시간대 = 발생시 // 3)",
            "도로_작은분류": "칸 사고 지점 중 가장 흔한 값 (비율은 도로분류비율_*)",
            "생활인구": "pop.csv 평균, 칸×주말여부×시각, 스무딩 5",
            "평소속도": "TOPIS 같은 달·같은 시각 평균. 출처 1 구간, 2 도로명 평균, 0 없음"}
    with open(os.path.join(OUT, "표_정보.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"완료 ({time.time() - t0:.0f}초). 저장 위치: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()

# 서비스용 표 결과 요약: 서울 2,118칸의 고정값·생활인구·평소속도·칸 모양·도로상황 표를 data/service_tables/에 저장함
# 과거 사고 4개 변수는 test와 100% 같고, 평소속도는 팀 road_speed와 99.9% 일치함 → 학습 때와 같은 방식으로 계산됨
# 생활인구는 실제값과 상관 0.989(오차 중앙값 8%), 속도 없는 칸 247개는 학습 때처럼 빈 값으로 둠
# 도로상황 값은 데이터 기준으로 원활 1.04 / 평소 1.0 / 조금 막힘 0.94 / 많이 막힘 0.86으로 정함