# 변수 파일 목록

새 변수를 올리면 여기에 한 줄 추가해 주세요. 규칙과 올리는 법은 [docs/DATA_GUIDE.md](../../docs/DATA_GUIDE.md) 5장.

| 이름 (파일) | 담당 | 키 | 변수 | 원자료 / 만드는 코드 | 상태 |
|---|---|---|---|---|---|
| `sun` | 민채 | date, 발생시 | 해진비율, 어두움비율, 박명비율, 실제밤, 태양고도, 일몰후분, 일출까지분 | 천문 계산 (`features/make_sun.py`의 `sun_hourly.csv`) → `scripts/features/sun.py` | 준비됨 |
| `zone` | woojin | grid_id | 어린이보호구역_수, 노인보호구역_수, 장애인보호구역_수, 보호구역CCTV_합계 | 공공데이터포털 전국어린이보호구역·전국노인장애인보호구역 표준데이터 (서울 구별 파일 합본) → `scripts/features/zone.py` | 준비됨 |
| `road_link` | cwoo0611 | grid_id | 칸_도로연장_m, 칸_평균제한속도_kmh, 칸_최대차로수, 칸_최고도로등급, 칸_간선도로비율, 칸_터널비율, 칸_교량고가비율 | 국가교통정보센터 전국표준노드링크 (MOCT_LINK.shp) → `scripts/features/road_link.py` | 준비됨 |
| `road_speed` | cwoo0611 | date, 발생시, grid_id | 칸_직전1시간_통행속도_kmh, 칸_평소속도_kmh, 칸_직전1시간_속도비율, 칸_속도출처 | 서울시 TOPIS 도로별 일자별 통행속도 2020~2025 + TOPIS 구간 위치 + 표준노드링크 → `scripts/features/road_speed.py` | 준비됨 |
| `pop` | 민채 | date, 발생시, grid_id | 생활인구_내국인, 생활인구_장기외국인, 생활인구_단기외국인, 생활인구_합계 | 서울 생활인구 집계구 (`population/run_monthly.py`) → `scripts/features/pop.py` | 72개월 집계 중 |
