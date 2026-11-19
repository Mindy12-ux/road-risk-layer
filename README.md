# road-risk-layer
서울 교통사고·날씨·도로 데이터로 500m 격자별 사고 위험도를 예측하는 LightGBM + Flask 서비스 (위험 지도 · 자율주행 시뮬레이션 · API)

## 데이터
데이터 파일은 깃이 아니라 DVC(+구글 드라이브 공유 폴더)로 버전관리합니다.
처음이면 **[docs/DATA_GUIDE.md](docs/DATA_GUIDE.md)** 를 따라 하세요.

```
git pull && dvc pull                 # 최신 데이터 받기
python scripts/merge.py sun pop      # 기준 데이터셋 + 고른 변수 → data/merged/
```

## 상관관계 · 모델 학습
```
python scripts/merge.py sun zone road_link road_speed pop --keep-keys
python scripts/make_model_data.py    # → data/model/ (연도별 분할, dataset_v2 규칙)
python model/check_correlation.py    # → results/corr/
python model/train_two_stage.py      # → results/model/ (2단계 LightGBM)
```
자세한 설명은 [docs/DATA_GUIDE.md](docs/DATA_GUIDE.md) 4장.
