# 변수 파일 목록

새 변수를 올리면 여기에 한 줄 추가해 주세요. 규칙과 올리는 법은 [docs/DATA_GUIDE.md](../../docs/DATA_GUIDE.md) 5장.

| 이름 (파일) | 담당 | 키 | 변수 | 원자료 / 만드는 코드 | 상태 |
|---|---|---|---|---|---|
| `sun` | 민채 | date, 발생시 | 해진비율, 어두움비율, 박명비율, 실제밤, 태양고도, 일몰후분, 일출까지분 | 천문 계산 (`features/make_sun.py`의 `sun_hourly.csv`) → `scripts/features/sun.py` | 준비됨 |
| `zone` | woojin | grid_id | 어린이보호구역_수, 노인보호구역_수, 장애인보호구역_수, 보호구역CCTV_합계 | 공공데이터포털 전국어린이보호구역·전국노인장애인보호구역 표준데이터 (서울 구별 파일 합본) → `scripts/features/zone.py` | 준비됨 |
| `pop` | 민채 | date, 발생시, grid_id | 생활인구_내국인, 생활인구_장기외국인, 생활인구_단기외국인, 생활인구_합계 | 서울 생활인구 집계구 (`population/run_monthly.py`) → `scripts/features/pop.py` | 72개월 집계 중 |
