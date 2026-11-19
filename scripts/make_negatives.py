"""
사고가 '안 난' 데이터(0)를 만듭니다. 사고 데이터(1)의 3배.

사용법 (프로젝트 폴더에서):
    python3 scripts/make_negatives.py

만드는 방법 (아주 쉽게):
  1) 장소 고르기: 서울 500m 네모 중 하나를 고르고, 그 네모 안에서 예전에 사고가 났던
     '도로 위 지점' 하나를 골라요. → 정확한 위도·경도와 도로형태가 생겨요.
     (산이나 강 한가운데 같은, 도로가 없는 곳은 뽑히지 않아요.)
  2) 시간 고르기: 2020~2025년 중 아무 달, 아무 요일, 아무 시각을 똑같은 확률로 골라요.
     그리고 그 달의 그 요일 중 하루를 실제 날짜로 정해요. (예: 2023년 3월 금요일 → 3월 17일)
  3) 확인하기: 그 네모, 그 달, 그 요일, 그 시각에 사고 기록이 있으면 버려요.
  4) 날씨 붙이기: 그 날짜, 그 시각의 서울 실제 날씨를 붙여요. (Open-Meteo 과거 날씨)

입력: data/raw/taas/taas_accidents_사망중상경상.csv
      data/raw/weather/seoul_hourly_openmeteo.json
      data/processed/grid_cells.csv   (build_dataset.py가 만든 서울 500m 네모 목록)
출력: data/processed/negatives.csv                     사고 안 난 데이터 (0)
      data/processed/accidents_with_negatives.csv      사고 난 데이터(1) + 사고 안 난 데이터(0)
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ACC_PATH = os.path.join(ROOT, "data", "raw", "taas", "taas_accidents_사망중상경상.csv")
WEATHER_PATH = os.path.join(ROOT, "data", "raw", "weather", "seoul_hourly_openmeteo.json")
CELLS_PATH = os.path.join(ROOT, "data", "processed", "grid_cells.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")

CELL = 500
RATIO = 3                 # 사고 1건당 사고 안 난 데이터 3개
YEARS = range(2020, 2026)
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
COLUMNS = ["사고여부", "연도", "발생년월", "기준날짜", "요일", "발생시", "사고내용",
           "기상상태", "노면상태", "도로형태", "위도", "경도"]
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 날씨: Open-Meteo 날씨 코드를 TAAS와 같은 이름(맑음/흐림/비/눈/안개)으로 바꿉니다.
# ---------------------------------------------------------------------------
def load_weather():
    with open(WEATHER_PATH, encoding="utf-8") as f:
        h = json.load(f)["hourly"]
    w = pd.DataFrame(h)
    w["time"] = pd.to_datetime(w["time"])

    # 경찰 사고 기록은 흐린 날도 대부분 '맑음'으로 적고, 이슬비는 '비'로 잘 안 적어요.
    # 그래서 기준을 경찰 기록에 맞춥니다.
    #   비: 1시간에 1mm 이상 내림 / 눈: 눈이 0.5mm 이상 내림 / 나머지: 맑음 (흐림 포함)
    sky = np.full(len(w), "맑음", dtype=object)
    sky[w["precipitation"] >= 1.0] = "비"
    sky[(w["snowfall"] > 0) & (w["precipitation"] >= 0.5)] = "눈"
    w["기상상태"] = sky

    # 노면상태: 최근 3시간 동안 비나 눈이 왔는지, 눈이 쌓였는지, 영하인지로 정합니다.
    recent_rain = w["precipitation"].rolling(3, min_periods=1).sum()
    recent_snow = w["snowfall"].rolling(3, min_periods=1).sum()
    surface = np.full(len(w), "건조", dtype=object)
    wet = recent_rain >= 1.0
    surface[wet] = "젖음/습기"
    # 최근 12시간 안에 비·눈이 왔는데 지금 영하면 → 얼었을 수 있음
    recent_12h = w["precipitation"].rolling(12, min_periods=1).sum()
    surface[(recent_12h >= 1.0) & (w["temperature_2m"] <= 0)] = "서리/결빙"
    # 쌓인 눈은 제설하면 금방 치워지므로, '최근 3시간 안에 눈이 내린 경우'만 적설로 봅니다.
    surface[recent_snow >= 0.5] = "적설"
    w["노면상태"] = surface
    return w.set_index("time")[["기상상태", "노면상태"]]


# ---------------------------------------------------------------------------
# 사고 데이터 읽기
# ---------------------------------------------------------------------------
def load_accidents():
    acc = pd.read_csv(ACC_PATH, encoding="utf-8-sig", dtype=str)
    acc["year"] = acc["연도"].astype(int)
    acc["month"] = acc["발생년월"].str.extract(r"(\d+)월")[0].astype(int)
    acc["weekday"] = acc["요일"].map({w: i for i, w in enumerate(WEEKDAYS)})
    acc["hour"] = acc["발생시"].astype(int)
    acc["cx"] = (acc["X_UTMK"].astype(float) // CELL).astype(int)
    acc["cy"] = (acc["Y_UTMK"].astype(float) // CELL).astype(int)
    return acc


def slot_key(cx, cy, year, month, weekday, hour):
    """네모 + 연월 + 요일 + 시각을 숫자 하나로 묶은 이름표."""
    return (((((np.asarray(cx) * 10_000 + np.asarray(cy)) * 10_000 + np.asarray(year)) * 100
              + np.asarray(month)) * 10 + np.asarray(weekday)) * 100 + np.asarray(hour))


def main():
    acc = load_accidents()
    weather = load_weather()
    cells = pd.read_csv(CELLS_PATH)

    # 장소 후보: 서울 네모 안에서 사고가 났던 도로 위 지점들
    seoul = acc.merge(cells[["cx", "cy"]], on=["cx", "cy"])
    by_cell = {k: g.index.values for k, g in seoul.groupby(["cx", "cy"])}
    cell_list = list(by_cell.keys())
    print(f"1) 장소 후보: 도로가 있는 네모 {len(cell_list)}개 (서울 전체 {len(cells)}개 중)")

    accident_keys = set(slot_key(acc["cx"], acc["cy"], acc["year"], acc["month"], acc["weekday"], acc["hour"]))

    # 시간 후보: 2020~2025년의 모든 (연, 월, 요일) 조합과 그에 해당하는 실제 날짜들
    days = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    dates_of = pd.Series(days).groupby([days.year, days.month, days.dayofweek]).agg(list).to_dict()
    combos = list(dates_of.keys())

    target = len(acc) * RATIO
    print(f"2) 사고 안 난 데이터 {target:,}개 만드는 중... (사고 {len(acc):,}건 × {RATIO})")

    rows, used = [], set()
    while len(rows) < target:
        n = (target - len(rows)) * 2
        cell_pick = RNG.integers(len(cell_list), size=n)
        combo_pick = RNG.integers(len(combos), size=n)
        hour_pick = RNG.integers(24, size=n)
        for ci, ti, hour in zip(cell_pick, combo_pick, hour_pick):
            cx, cy = cell_list[ci]
            year, month, wd = combos[ti]
            key = int(slot_key(cx, cy, year, month, wd, hour))
            if key in accident_keys or key in used:   # 그때 사고가 났거나, 이미 뽑은 줄이면 버림
                continue
            used.add(key)
            src = seoul.loc[RNG.choice(by_cell[(cx, cy)])]   # 이 네모 안의 도로 위 지점 하나
            date = dates_of[(year, month, wd)][RNG.integers(len(dates_of[(year, month, wd)]))]
            when = date + pd.Timedelta(hours=int(hour))
            sky, surface = weather.loc[when]
            rows.append({
                "사고여부": 0,
                "연도": year,
                "발생년월": f"{year}년 {month:>2}월",
                "기준날짜": date.strftime("%Y-%m-%d"),
                "요일": WEEKDAYS[wd],
                "발생시": f"{hour:02d}",
                "사고내용": "사고없음",
                "기상상태": sky,
                "노면상태": surface,
                "도로형태": src["도로형태"],
                "위도": src["위도"],
                "경도": src["경도"],
            })
            if len(rows) >= target:
                break

    neg = pd.DataFrame(rows, columns=COLUMNS)
    neg.to_csv(os.path.join(OUT_DIR, "negatives.csv"), index=False, encoding="utf-8-sig")

    # 사고 기록의 날씨 이름도 같은 기준으로 맞춥니다. (흐림, 안개 → 맑음)
    pos = acc.assign(사고여부=1, 기준날짜="")[COLUMNS]
    pos["기상상태"] = pos["기상상태"].replace({"흐림": "맑음", "안개": "맑음"})
    both = pd.concat([pos, neg], ignore_index=True)
    both.to_csv(os.path.join(OUT_DIR, "accidents_with_negatives.csv"), index=False, encoding="utf-8-sig")

    print(f"3) 저장 완료: 사고 난 데이터 {len(pos):,}줄 + 사고 안 난 데이터 {len(neg):,}줄 = {len(both):,}줄")
    print("\n[확인] 날씨 비교 (%)")
    print(pd.DataFrame({
        "사고 난 데이터": pos["기상상태"].value_counts(normalize=True).mul(100).round(1),
        "사고 안 난 데이터": neg["기상상태"].value_counts(normalize=True).mul(100).round(1),
    }).fillna(0).to_string())
    print("\n[확인] 노면 비교 (%)")
    print(pd.DataFrame({
        "사고 난 데이터": pos["노면상태"].value_counts(normalize=True).mul(100).round(1),
        "사고 안 난 데이터": neg["노면상태"].value_counts(normalize=True).mul(100).round(1),
    }).fillna(0).to_string())
    print(f"\n완료. 저장 위치: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()

#0) 불러옴: 761,090줄
#1) 정답 0/1로 바꿈: {0: 575445, 1: 185645}
#2) 서울 밖 줄 뺌: {0: 7287, 1: 325} → 남은 줄 753,478
#3) 월, 요일, 주말, 밤, 출퇴근, 도로 분류, 500m 칸 만듦
#4) 과거 기록 변수 11개 만듦 (2020년 126,646줄은 빈칸)
#5) road_link  붙임: ['칸_도로연장_m', '칸_평균제한속도_kmh', '칸_최대차로수','칸_간선도로비율', '칸_교량고가비율']  빈칸 비율 칸_도로연장_m 0.0%, 칸_평균제한속도_kmh 1.2%, 칸_최대차로수 1.2%, 칸_간선도로비율 1.2%, 칸_교량고가비율 1.2%
#5) pop        붙임: ['생활인구_합계']  빈칸 비율 생활인구_합계 0.0%
#5) road_speed 붙임: ['칸_평소속도_kmh', '칸_직전1시간_속도비율']  빈칸 비율 칸_평소속도_kmh 9.5%, 칸_직전1시간_속도비율 9.5%
#5) sun        붙임: ['태양고도', '박명비율']  빈칸 비율 태양고도 0.0%, 박명비율 0.0%
#6) 기상상태·노면상태·도로 분류에 번호 붙임
#7) 각 연도에서 60% / 20% / 20%로 나눔:
#   train 452,076줄   사고없음 340,892   사고 111,184
#   valid 150,695줄   사고없음 113,632   사고 37,063
#   test  150,707줄   사고없음 113,634   사고 37,073
#완료. 변수 31개. 저장 위치: /Users/woojin/ML project/data/model_input
