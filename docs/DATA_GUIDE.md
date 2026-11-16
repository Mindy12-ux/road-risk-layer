# 데이터셋 버전관리 가이드 (DVC + 구글 드라이브)

> 처음 써보는 사람 기준으로 썼습니다. 명령어는 **저장소 맨 위 폴더(road-risk-layer)** 에서 실행합니다.
> Windows는 PowerShell, Mac은 터미널 기준입니다.

---

## 0. 한눈에 보기

### 왜 DVC를 쓰나
깃(GitHub)은 코드용이라 수십~수백 MB짜리 CSV를 올리면 느려지고 100MB가 넘으면 아예 안 올라갑니다.
**DVC**는 깃 옆에서 같이 쓰는 도구로, 이렇게 나눠서 관리합니다.

| 무엇 | 어디에 저장 | 예 |
|---|---|---|
| 코드, 가이드, **데이터의 "영수증"(.dvc 파일)** | GitHub | `data/features/sun.csv.dvc` (몇 줄짜리 텍스트) |
| **데이터 실제 파일** | 구글 드라이브 공유 폴더 `road-risk-dvc` | `sun.csv` 본체 |

`.dvc` 영수증에는 "이 버전의 sun.csv는 지문(해시)이 abc123…" 이라고만 적혀 있습니다.
그래서 **`git pull` 로 영수증을 받고 → `dvc pull` 로 영수증에 맞는 실제 파일을 받는** 두 단계가 기본입니다.
깃 커밋을 옛날로 돌리면 영수증도 옛날 것이 되니, 데이터도 그 시점 버전으로 돌아갈 수 있습니다.

### 데이터 구조 (충돌 안 나게 나눠 둠)

```
data/
├─ raw/                 원재료 (TAAS 사고, 무사고 샘플, 원본 train/valid/test)   ← 팀장만 수정
├─ base/                기준 데이터셋 = 원본 train/valid/test + 키 컬럼 2개       ← 팀장만 수정
│   ├─ train.csv  valid.csv  test.csv
│   └─ keys.csv         (기준 데이터셋에 나오는 날짜·시간·칸 조합 목록)
├─ features/            각자 맡은 변수 파일 (한 사람 = 한 파일 이상)
│   ├─ sun.csv          일출·일몰 (꾸꾸)
│   ├─ pop.csv          생활인구 (꾸꾸)
│   └─ ○○.csv           팀원 변수…
└─ merged/              merge.py 결과 (각자 컴퓨터에서 만듦, 공유 안 함)
```

**핵심 규칙: 큰 데이터셋을 통째로 고쳐서 돌리지 않는다.**
각자 **키 + 내 변수만** 담은 작은 파일을 `data/features/`에 올리고, 필요한 사람이 `merge.py`로 골라 붙입니다.

### 키 컬럼 (모든 파일의 공통 연결고리)

| 키 | 뜻 | 예 |
|---|---|---|
| `date` | 날짜 | `2021-07-18` |
| `발생시` | 시간 (0~23) | `14` |
| `grid_id` | 500m 칸 (`cx_cy`) | `1910_3905` |

기준 데이터셋(`data/base/*.csv`)에는 원본 컬럼 뒤에 `date`, `grid_id`가 붙어 있습니다.
원본에는 날짜가 없어서, 사고 행은 TAAS 사고번호로, 무사고 행은 무사고 샘플의 기준날짜로 복원했습니다.

---

## 1. 설치 (모두, 처음 한 번)

1. **Git** 설치: https://git-scm.com (설치 옵션은 기본값으로)
2. **Python 3.10 이상** 설치 (Windows는 설치 첫 화면에서 **"Add python.exe to PATH" 체크**)
3. **구글 드라이브 데스크톱 앱** 설치: https://www.google.com/drive/download/ → 구글 계정으로 로그인
   - 설치하면 Windows는 탐색기에 `G:` 드라이브(또는 다른 글자)가 생기고 그 안에 `내 드라이브`가 보입니다.
4. 패키지 설치 — 가상환경 안에서 하는 걸 추천 (다른 프로젝트와 버전이 안 꼬임)
   저장소를 clone 한 뒤(2-2, 3장) 저장소 폴더에서:
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1          # Mac: source .venv/bin/activate
   pip install dvc pandas pyarrow
   ```
   - 줄 앞에 `(.venv)`가 보이면 활성화된 것. **터미널을 새로 열 때마다 Activate 줄을 다시 실행**하세요.
   - "스크립트를 실행할 수 없습니다" 오류 → `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 한 번 실행 후 다시.
   - VS Code: 오른쪽 아래 Python 버전을 눌러 `.venv` 선택하면 터미널이 자동 활성화됩니다.
   - 아나콘다를 쓰면 conda 환경 안에서 같은 `pip install` 해도 됩니다. 이때 `conda env config vars set PYTHONUTF8=1` 도 한 번 해 두면 한글 경로·파일 오류를 막을 수 있습니다 (실행 후 `conda activate 환경이름` 다시). 가상환경 없이 설치해도 동작은 합니다.
5. 확인
   ```
   dvc --version
   ```
   숫자(예: `3.x.x`)가 나오면 성공. `dvc`를 못 찾는다고 나오면 앞으로 `dvc` 대신 **`python -m dvc`** 라고 쓰면 됩니다.

---

## 2. 팀장(꾸꾸) 처음 세팅 — 한 번만

### 2-1. 구글 드라이브에 데이터 창고 폴더 만들고 공유
1. 웹 구글 드라이브(drive.google.com) → **내 드라이브**에 새 폴더 `road-risk-dvc` 만들기 (이름 정확히)
2. 폴더 우클릭 → **공유** → 팀원 3명 구글 이메일 추가, 권한 **편집자**
3. 데스크톱 앱이 켜져 있으면 내 PC의 `G:\내 드라이브\road-risk-dvc` 에도 폴더가 생깁니다.

> 용량: 구글 무료 15GB 중 지금 데이터는 수백 MB 정도라 충분합니다. 팀원이 올린 파일도 폴더 주인(팀장) 용량을 씁니다.

### 2-2. 저장소 받기 + 창고 연결
```
git clone https://github.com/mindy12-ux/road-risk-layer.git
cd road-risk-layer
```
여기서 1장 4번(가상환경 + 패키지 설치)을 하고, 이어서
```
python scripts/set_storage.py
```
`설정 완료: 데이터 창고 = G:/내 드라이브/road-risk-dvc` 처럼 나오면 됩니다.
못 찾으면 탐색기에서 그 폴더 주소를 복사해서 `python scripts/set_storage.py "G:\내 드라이브\road-risk-dvc"`.

### 2-3. 원재료 넣기
`data/raw/` 폴더를 만들고 아래 파일을 넣습니다.
```
data/raw/taas_accidents.csv
data/raw/no_accident.csv
data/raw/original/train.csv      ← 팀원들과 같이 쓰는 원본 train/valid/test
data/raw/original/valid.csv
data/raw/original/test.csv
```

### 2-4. 기준 데이터셋 만들기
PowerShell은 줄바꿈 대신 한 줄로 쓰는 게 편합니다.
```
python scripts/make_base.py --split train=data/raw/original/train.csv --split valid=data/raw/original/valid.csv --split test=data/raw/original/test.csv --taas data/raw/taas_accidents.csv --noacc data/raw/no_accident.csv
```
`data/base/train.csv … keys.csv`가 생깁니다. (날짜를 못 찾은 행이 있으면 오류로 멈춥니다.)

### 2-5. 내 변수 파일 만들기
```
python scripts/features/sun.py 경로/sun_hourly.csv
python scripts/merge.py --check sun
```
생활인구는 72개월 집계(`pop_500m` 폴더)가 끝난 뒤에:
```
python scripts/features/pop.py 경로/pop_500m
python scripts/merge.py --check pop
```
(`pop.py`는 기준 데이터셋에 나오는 날짜·시간·칸만 골라서 수십 GB → 수십 MB로 줄입니다.)

### 2-6. DVC에 등록하고 올리기
```
dvc add data/raw data/base data/features/sun.csv
git add data .dvc
git commit -m "기준 데이터셋 + 일출·일몰 변수 등록"
git push
dvc push
```
- `dvc add` = 실제 파일은 DVC가 맡고, 영수증(`.dvc`)을 만듭니다.
- `git push` = 영수증을 GitHub에, `dvc push` = 실제 파일을 드라이브 폴더에.
- `dvc push` 후 **작업표시줄의 구글 드라이브 아이콘에서 동기화(업로드)가 끝날 때까지** 컴퓨터를 끄지 마세요. 끝나야 팀원이 받을 수 있습니다.

---

## 3. 팀원 처음 세팅 — 한 번만

1. 메일/드라이브 알림으로 온 `road-risk-dvc` 폴더 공유를 엽니다.
2. drive.google.com → **공유 문서함** → `road-risk-dvc` 우클릭 → **정리 → 바로가기 추가 → 내 드라이브**
   (이걸 해야 내 PC의 `G:\내 드라이브\` 안에 보입니다.)
3. PowerShell에서
   ```
   git clone https://github.com/mindy12-ux/road-risk-layer.git
   cd road-risk-layer
   ```
   1장 4번(가상환경 + 패키지 설치)을 한 뒤
   ```
   python scripts/set_storage.py
   dvc pull
   ```
4. `data/base/`, `data/features/` 안에 CSV가 생겼으면 끝!

> GitHub 저장소가 비공개라면 팀장이 GitHub 저장소 **Settings → Collaborators** 에서 팀원을 초대해야 clone/push가 됩니다.

---

## 4. 평소에 쓰는 법

### 최신 데이터 받기 (작업 시작할 때마다)
```
git pull
dvc pull
```

### 원하는 변수만 붙여서 학습용 데이터 만들기
```
python scripts/merge.py                  # 어떤 변수가 있는지 보기
python scripts/merge.py sun pop          # sun, pop 붙이기 → data/merged/train.csv, valid.csv, test.csv
python scripts/merge.py sun --out data/merged_sun    # 결과 폴더 이름 바꾸기
```
- 결과는 **원본과 같은 행·같은 순서·같은 컬럼 + 고른 변수**입니다. (`date`, `grid_id`는 기본으로 빼 줌. 남기려면 `--keep-keys`)
- 그래서 기존 학습 코드에서 읽는 파일 경로만 `data/merged/train.csv`로 바꾸면 됩니다.
- 붙인 변수 중 빈 값이 있으면 비율을 알려 줍니다. (예: 생활인구 데이터가 없는 달)

### 상관관계 보고 모델 학습하기
처음 한 번만 `pip install lightgbm scikit-learn scipy matplotlib`
```
python scripts/merge.py sun zone road_link road_speed pop --keep-keys   # ① 변수 붙이기 (--keep-keys 꼭)
python scripts/make_model_data.py                                       # ② 학습용 데이터 → data/model/
python model/check_correlation.py                                       # ③ 상관관계 → results/corr/
python model/train_two_stage.py                                         # ④ 모델 학습 → results/model/
```
- ②는 dataset_v2와 같은 규칙으로 바꿉니다: 사고 행 날씨·노면을 무사고 파일 기준으로 통일, 위도·경도·연도 제거, 2021~23 train / 2024 valid / 2025 test (2020 제외), 라벨 `사고심각도`(0~3).
- ③에서 `|r| ≥ 0.8`인 변수 쌍이 나오면 둘 중 하나를 빼고 학습해 보세요: `python model/train_two_stage.py --drop 일출까지분 실제밤`
- 팀원 변수 효과 비교: `python model/train_two_stage.py --only-base --out results/model_base` (원래 16개 변수만) 결과의 `metrics.json`과 비교

---

## 5. 내 변수 추가하기 (팀원이 제일 많이 할 일)

예: 도로 속도를 맡은 사람이 `speed`라는 변수를 추가하는 경우

### 5-1. 변수 파일 규칙
`data/features/speed.csv` (또는 `.parquet`) 하나에:

- **키 컬럼 1개 이상** + **내 변수 컬럼**
  | 내 변수가… | 키 | 예 |
  |---|---|---|
  | 날짜·시간마다 다름 (서울 전체 같은 값) | `date, 발생시` | 일출·일몰, 날씨 |
  | 칸마다 다름 (시간 무관) | `grid_id` | 도로 밀도, 신호등 수 |
  | 칸·날짜·시간마다 다름 | `date, 발생시, grid_id` | 생활인구, 속도 |
  | 칸·시간대마다 다름 (날짜 무관) | `발생시, grid_id` | 시간대별 평균 속도 |
- 같은 키 조합은 **한 줄만**
- 변수 이름은 **다른 사람 것과 안 겹치게** 구체적으로 (`속도` ✗ → `평균속도_kmh` ○)
- 파일이 아주 크면 `data/base/keys.csv`에 있는 조합만 남기세요 (기준 데이터셋에 필요한 건 그것뿐).
- `grid_id`는 `cx_cy` 형식 (`grid/grid.parquet`의 `grid_id`와 같음). 위경도만 있으면 그 칸을 계산해서 넣으세요.

예시 (`speed.csv`)
```
date,발생시,grid_id,평균속도_kmh
2021-07-18,14,1910_3905,23.4
2021-07-18,14,1911_3905,31.0
```

### 5-2. 만드는 코드도 같이 올리기
`scripts/features/speed.py` 처럼 **원자료 → `data/features/speed.csv`** 를 만드는 코드를 만들어 두세요. (`sun.py`, `pop.py` 참고)
나중에 데이터를 고칠 때 다시 돌릴 수 있고, 다른 사람이 어떻게 만들었는지 볼 수 있습니다.

### 5-3. 검사 → 등록 → 올리기
```
git pull
dvc pull
python scripts/merge.py --check speed          # [통과] 가 나와야 함
dvc add data/features/speed.csv
```
`data/features/README.md` 표에 내 변수 한 줄 추가한 뒤
```
git add data/features/speed.csv.dvc data/features/README.md scripts/features/speed.py
git commit -m "speed 변수 추가"
git push
dvc push
```
그리고 단톡에 "speed 올렸어요" 한마디 → 다른 사람은 `git pull` + `dvc pull` 하면 받습니다.

> 각자 다른 파일(`speed.csv.dvc`, `sun.csv.dvc`…)만 건드리니 동시에 올려도 충돌이 안 납니다.

---

## 6. 버전 바꾸기 / 되돌리기

### 내 변수를 고쳐서 새 버전 올리기
파일을 다시 만든 뒤 **같은 순서**로:
```
dvc add data/features/speed.csv
git commit -am "speed: 결측 처리 수정"
git push
dvc push
```
옛 버전도 드라이브 창고에 그대로 남아 있습니다.

### 예전 버전으로 돌아가기
```
git log --oneline -- data/features/speed.csv.dvc     # 버전 목록 (커밋 번호 확인)
git checkout 1a2b3c4 -- data/features/speed.csv.dvc  # 그 시점 영수증으로
dvc checkout data/features/speed.csv.dvc             # 실제 파일도 그 버전으로
```
다시 최신으로: `git checkout HEAD -- data/features/speed.csv.dvc` 후 `dvc checkout`.

### 실험 결과에 데이터 버전 남기기
모델 결과를 기록할 때 `git log -1 --oneline` 의 커밋 번호를 같이 적어 두면, 나중에 그 커밋으로 가서 `dvc pull` 하면 똑같은 데이터가 나옵니다.

---

## 7. 자주 나는 오류 (Windows 중심)

| 증상 | 원인 / 해결 |
|---|---|
| `'dvc'은(는) 내부 또는 외부 명령…이 아닙니다` | PATH 문제. `python -m dvc pull` 처럼 앞에 `python -m` 붙이기 |
| `'python'… 아닙니다` 또는 MS 스토어가 열림 | Python 설치 때 PATH 체크 안 함 → 재설치하며 "Add to PATH" 체크, 또는 `py` 로 실행 |
| `dvc pull` 에서 `missing cache files` / `failed to pull` | ① 올린 사람이 `dvc push`를 안 함 ② 올린 사람 PC의 구글 드라이브 동기화가 아직 안 끝남 ③ 내 PC의 드라이브 앱이 꺼져 있음. 올린 사람에게 확인 |
| `config file error` / `url` 관련 오류 | 창고 경로가 설정 안 됨 → `python scripts/set_storage.py` |
| `dvc remote modify --local …` 에서 `'ascii' codec can't encode` | DVC가 한글 경로를 설정 파일에 못 씀 → 그 명령 대신 `python scripts/set_storage.py "경로"` 사용 |
| `config file error: 'cp949' codec can't decode` | 설정 파일의 한글 경로를 못 읽음. `conda env config vars set PYTHONUTF8=1` → `conda activate 환경이름` (venv는 `setx PYTHONUTF8 1` 후 터미널 다시 열기) → `python scripts/set_storage.py` 다시 실행 |
| `set_storage.py` 가 폴더를 못 찾음 | 공유 폴더를 "내 드라이브에 바로가기 추가" 안 함, 또는 드라이브 앱 로그인 안 됨 |
| `git push` 가 `rejected` | 다른 사람이 먼저 올림 → `git pull` 후 다시 `git push` |
| `git pull` 에서 `.dvc` 파일 충돌 | 두 사람이 **같은 변수 파일**을 동시에 고침. 담당자 한 명만 고치기로 정하고, 담당자 버전으로 `git checkout --theirs 파일.dvc` 등 정리 (모르겠으면 팀장에게) |
| `[오류] 같은 [...] 조합이 N줄 중복` | 내 변수 파일에 같은 키가 여러 줄 → 평균/합계로 한 줄로 합치기 |
| `[오류] 컬럼 이름 [...] 이(가) 이미 있습니다` | 변수 이름이 기준 데이터셋이나 다른 변수와 겹침 → 이름 바꾸기 |
| CSV 한글이 엑셀에서 깨짐 | 스크립트는 `utf-8-sig`로 저장해서 엑셀에서 잘 보임. 직접 저장할 땐 `to_csv(..., encoding="utf-8-sig")` |
| 엑셀로 열었다 저장했더니 이상해짐 | **데이터 파일은 엑셀에서 저장하지 마세요.** 형식·인코딩이 바뀝니다. `dvc checkout`으로 원래대로 |
| `LF will be replaced by CRLF` 경고 | 무시해도 됩니다 |
| `dvc add` 했는데 `git status`에 CSV가 안 보임 | 정상. CSV는 깃에서 숨기고 `.dvc`만 올리는 게 맞습니다 |

---

## 8. 명령어 요약

| 하고 싶은 것 | 명령 |
|---|---|
| 최신 받기 | `git pull` → `dvc pull` |
| 변수 붙이기 | `python scripts/merge.py sun pop` |
| 내 변수 검사 | `python scripts/merge.py --check 이름` |
| 내 변수 등록 | `dvc add data/features/이름.csv` |
| 올리기 | `git add …` → `git commit -m "…"` → `git push` → `dvc push` |
| 지금 상태 | `git status`, `dvc status` |
| 창고 경로 설정 | `python scripts/set_storage.py` |

---

## 부록. 왜 구글 드라이브 "폴더" 방식인가

- DVC에는 구글 드라이브 직접 연결(`gdrive://`) 기능도 있지만, DVC 기본 로그인 앱이 구글에서 **"This app is blocked"** 로 막히는 문제가 있습니다 ([DVC 문서](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive)). 피하려면 팀장이 Google Cloud 프로젝트와 OAuth 클라이언트를 직접 만들어야 해서 초보자에게 번거롭습니다.
- **DagsHub**(GitHub 저장소 연결 + DVC 저장소 무료 제공)도 편하지만, 무료 플랜의 비공개 저장소는 협업자가 2명까지라 4인 팀에는 모자랍니다 ([요금제](https://dagshub.com/pricing/)).
- 그래서 데스크톱 앱이 동기화하는 **드라이브 폴더를 그냥 로컬 폴더처럼** 창고로 씁니다. 추가 계정·API 설정 없이 4명 모두 쓸 수 있습니다.
- 나중에 다른 창고로 옮기고 싶으면 `.dvc/config`의 `gdrive` 주소만 바꾸고 `dvc push` 하면 됩니다.
