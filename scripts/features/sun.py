"""일출·일몰 피처 → data/features/sun.csv  (키: date, 발생시)

  python scripts/features/sun.py 경로/sun_hourly.csv
sun_hourly.csv 는 features/make_sun.py 로 만든 파일 (날짜·시간별 태양 위치).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FEAT_DIR, normalize_keys, read_table, write_csv  # noqa: E402

COLS = ["해진비율", "어두움비율", "박명비율", "실제밤", "태양고도", "일몰후분", "일출까지분"]

d = normalize_keys(read_table(sys.argv[1]))[["date", "발생시"] + COLS]
write_csv(d, FEAT_DIR / "sun.csv")
print(f"data/features/sun.csv  {len(d):,}줄, 변수 {COLS}")
