"""
3단계: 실시간 속도 → 칸마다 '평소 대비 속도비율' 만들기 (Flask에서 불러다 쓰는 파일)

사용법 (다른 파일에서):
    from speed_now import SpeedNow
    sn = SpeedNow()                          # 표 불러오기 + API 키 읽기 (한 번만)
    ratio = sn.ratios()                      # 서울 전체 칸 {grid_id: 속도비율}  (처음 약 15초, 10분 동안은 저장값)
    ratio = sn.ratios(cells=["1910_3905"])   # 몇 칸만 (경로 칸) → 필요한 구간만 불러서 빠름
    X = fm.make(speed_ratio=ratio)           # 2단계 FeatureMaker에 그대로 넣기

확인용 실행 (road-risk-layer 폴더에서):
    python model/speed_now.py

어떻게 계산하나 (팀 road_speed.py와 같은 방식, '직전 1시간' 대신 '지금' 속도)
1) 칸을 지나는 TOPIS 구간이 있으면: 구간들의 지금 속도 평균 ÷ 같은 구간들의 평소 속도 평균 (출처 1)
2) 없으면 칸을 지나는 도로명: 그 도로 구간 몇 개(도로마다 최대 5개)의 지금 ÷ 평소 (출처 2)
3) 둘 다 없으면 비율 없음 → 2단계에서 1.0(평소) 또는 빈 값으로 처리
평소 속도 = data/service_tables/구간_평소속도.csv 의 같은 달·같은 시각 값 (2025년 TOPIS)
학습 데이터에서 거의 안 나온 값(0.5 미만, 1.6 초과)은 0.5~1.6으로 자름

실시간 API: 서울 열린데이터광장 TrafficInfo (구간 하나씩, xml만 됨)
http://openapi.seoul.go.kr:8088/{키}/xml/TrafficInfo/1/1/{링크아이디}
키는 환경변수 SEOUL_API_KEY 또는 .env 에서 읽음 (코드에 키를 적지 않음)

필요한 패키지: pandas numpy requests
"""

import datetime as dt
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SERVICE_DIR = os.path.join(ROOT, "data", "service_tables")
ENV_FILE = os.path.join(ROOT, ".env")
API = "http://openapi.seoul.go.kr:8088/{key}/xml/TrafficInfo/1/1/{link}"

CACHE_SEC = 10 * 60      # 한 번 받은 구간 속도는 10분 동안 다시 씀
WORKERS = 16             # 동시에 부르는 개수
ROAD_SAMPLE = 5          # 도로명 예비용: 도로마다 구간 최대 5개만 부름
CLIP = (0.5, 1.6)        # 학습 데이터 속도비율의 0.1% ~ 99.9% 범위


def read_key():
    key = os.environ.get("SEOUL_API_KEY")
    if not key and os.path.exists(ENV_FILE):
        with open(ENV_FILE, encoding="utf-8") as f:
            for line in f:
                if line.startswith("SEOUL_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    assert key, "SEOUL_API_KEY가 없어요 (.env 확인)"
    return key


class SpeedNow:
    def __init__(self, service_dir=SERVICE_DIR):
        def read(name):
            return pd.read_csv(os.path.join(service_dir, name), encoding="utf-8-sig",
                            dtype={"링크아이디": str})

        self.key = read_key()
        seg = read("칸_구간.csv")
        self.cell_seg = seg.groupby("grid_id")["링크아이디"].apply(list).to_dict()
        road = read("칸_도로명.csv")
        self.cell_road = road.groupby("grid_id")["도로명"].apply(list).to_dict()
        links = read("구간_목록.csv")
        usual = read("구간_평소속도.csv")
        self.usual = usual.set_index(["링크아이디", "월"])[[str(h) for h in range(24)]]
        # 도로명마다 대표 구간 최대 5개 (평소 속도가 있는 구간 중 고르게)
        has_usual = set(usual["링크아이디"])
        self.road_links = {}
        for name, g in links[links["링크아이디"].isin(has_usual)].groupby("도로명_비교용"):
            ids = sorted(g["링크아이디"])
            pick = np.linspace(0, len(ids) - 1, min(ROAD_SAMPLE, len(ids))).round().astype(int)
            self.road_links[name] = [ids[i] for i in sorted(set(pick))]
        self.cells = sorted(set(self.cell_seg) | set(self.cell_road))
        self.cache = {}                       # 링크아이디 → (받은 시각, 속도 또는 None)
        self.session = requests.Session()
        self.last = {}                        # 마지막 계산 정보 (출처별 칸 수, 실패 수, 걸린 시간)

    #  API 
    def _get(self, link):
        try:
            r = self.session.get(API.format(key=self.key, link=link), timeout=5)
            m = re.search(r"<prcs_spd>([\d.]+)</prcs_spd>", r.text)
            v = float(m.group(1)) if m else None
            return link, (v if v and v > 0 else None)
        except requests.RequestException:
            return link, None

    def fetch(self, links):
        """구간들의 지금 속도 {링크아이디: km/h 또는 None}. 10분 안에 받은 구간은 다시 안 부름."""
        now = time.time()
        need = [l for l in set(links) if l not in self.cache or now - self.cache[l][0] > CACHE_SEC]
        if need:
            with ThreadPoolExecutor(WORKERS) as ex:
                for link, v in ex.map(self._get, need):
                    self.cache[link] = (now, v)
        return {l: self.cache[l][1] for l in links}

    #  칸별 속도비율 
    def _usual(self, links, month, hour):
        idx = pd.MultiIndex.from_arrays([links, [month] * len(links)])
        return self.usual[str(hour)].reindex(idx).values

    def _ratio(self, links, speed, month, hour):
        """구간들의 지금 속도 평균 ÷ 같은 구간들의 평소 속도 평균 (둘 다 있는 구간만)"""
        if not links:
            return None
        now = np.array([speed.get(l) if speed.get(l) is not None else np.nan for l in links], float)
        usual = self._usual(links, month, hour)
        ok = ~np.isnan(now) & ~np.isnan(usual)
        if not ok.any():
            return None
        return float(np.clip(now[ok].mean() / usual[ok].mean(), *CLIP))

    def ratios(self, cells=None, when=None):
        """칸별 지금 속도비율 {grid_id: 비율}. 비율을 못 구한 칸은 빠짐 (2단계에서 1.0으로 처리)"""
        t0 = time.time()
        when = pd.Timestamp(when) if when is not None else pd.Timestamp(dt.datetime.now())
        month, hour = when.month, when.hour
        cells = list(cells) if cells is not None else self.cells
        seg_links = {c: self.cell_seg.get(c, []) for c in cells}
        road_links = {c: [l for r in self.cell_road.get(c, []) for l in self.road_links.get(r, [])]
                    for c in cells if not self.cell_seg.get(c)}
        need = {l for v in seg_links.values() for l in v} | {l for v in road_links.values() for l in v}
        speed = self.fetch(list(need))

        out, src = {}, {1: 0, 2: 0, 0: 0}
        for c in cells:
            r = self._ratio(seg_links[c], speed, month, hour)
            s = 1
            if r is None:                                # 구간이 없거나 속도를 못 받았으면 도로명 예비용
                links = road_links.get(c)
                if links is None:                        # 구간은 있었는데 속도를 못 받은 칸
                    links = [l for rd in self.cell_road.get(c, []) for l in self.road_links.get(rd, [])]
                    speed.update(self.fetch(links))
                r, s = self._ratio(links, speed, month, hour), 2
            if r is None:
                src[0] += 1
                continue
            out[c] = round(r, 3)
            src[s] += 1
        fail = sum(v is None for v in speed.values())
        self.last = {"시각": f"{when:%Y-%m-%d %H:%M}", "구간": src[1], "도로명": src[2], "없음": src[0],
                    "부른_구간": len(speed), "속도_못받은_구간": fail, "걸린_초": round(time.time() - t0, 1)}
        return out


# ---------------------------------------------------------------- 확인용 (python model/speed_now.py)
def check():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_features import MODEL_DIR, FeatureMaker
    from predict import RiskModel

    sn = SpeedNow()
    print(f"1) 칸 {len(sn.cells):,}개 (구간 있는 칸 {len(sn.cell_seg):,}, 도로명만 있는 칸 {len(sn.cells) - len(sn.cell_seg):,})")

    r = sn.ratios()
    print(f"2) 서울 전체 실시간 속도비율: {sn.last}")
    s = pd.Series(r)
    print(f"   비율 분포: 중앙값 {s.median():.3f}, 하위 10% {s.quantile(0.1):.3f}, 상위 10% {s.quantile(0.9):.3f}")
    print(f"   평소보다 막힘(0.95 미만) {(s < 0.95).mean():.0%}, 평소보다 원활(1.05 초과) {(s > 1.05).mean():.0%}")

    t = time.time()
    sn.ratios()
    print(f"3) 10분 안에 다시 부르면: {time.time() - t:.2f}초 (저장해 둔 속도 사용)")

    fm, rm = FeatureMaker(), RiskModel(MODEL_DIR)
    base = rm.predict(fm.make())
    live = rm.predict(fm.make(speed_ratio=r))
    g0 = pd.Series([x["등급"] for x in base]).value_counts(normalize=True)
    g1 = pd.Series([x["등급"] for x in live]).value_counts(normalize=True)
    print("4) 등급 비율 (속도 1.0 → 실시간): " +
        ", ".join(f"{g} {g0.get(g, 0):.1%} → {g1.get(g, 0):.1%}" for g in ["높음", "보통", "낮음"]))
    jam = sum("[오늘] 평소보다 차가 막혀요" in " ".join(x["오늘"]) for x in live)
    print(f"   '[오늘] 평소보다 차가 막혀요' 이유가 붙은 칸: {jam:,}개, 날씨 출처: {fm.weather_source}")


if __name__ == "__main__":
    check()

# 3단계 결과: 서울 1,871칸 중 1,868칸에 실시간 속도비율이 붙었고, 처음 13.5초·이후 10분간 0.7초로 계산됨
# 퇴근 시간이라 높음 칸이 18% → 24%로 늘었고, 비율 중앙값 0.88이 정상인지는 한가한 시간에 한 번 더 확인 필요함