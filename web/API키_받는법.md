# 네비게이션 웹 API 키 받는 법 (조원용)

각자 자기 계정으로 키를 받아서 `road-risk-layer/.env` 에 넣으면 돼요. `.env` 는 git에 안 올라가요.

## 1. 카카오 (길찾기 · 지도 · 장소검색) — 키 2개

1. https://developers.kakao.com/console/app 접속 → 카카오 계정 로그인
2. **앱 생성** → 앱 이름(아무거나), 회사명(이름이나 팀 이름), 카테고리 선택 → 저장
3. 왼쪽 **제품 설정 → 카카오맵 → 사용 설정 ON**
   - 계정에서 **처음 켜는 앱**이면 무료 쿼터가 나와요. (이미 다른 앱에서 켰다면 그 앱을 쓰세요. 새 앱은 결제 수단 연결을 요구해요)
4. 왼쪽 **앱 → 플랫폼 키** 에서 복사
   - **REST API 키** → `KAKAO_REST_KEY`, `KAKAO_MAP_REST_KEY` 에 같은 값
   - **JavaScript 키** → `KAKAO_JS_KEY`
5. 같은 화면에서 JavaScript 키 **수정** → **JavaScript SDK 도메인**에 아래 두 줄 추가 → 저장
   ```
   http://localhost:5000
   http://127.0.0.1:5000
   ```

길찾기(카카오모빌리티)는 키를 따로 받지 않고 이 **REST API 키**를 그대로 써요.
문서: https://developers.kakaomobility.com/docs/navi-api/directions/

## 2. 서울 열린데이터광장 (실시간 도로 속도) — 키 1개

1. https://data.seoul.go.kr/together/mypage/actkeyMain.do 접속 → 로그인 (서울시 통합회원)
2. **일반 인증키 신청** (실시간 지하철 말고 일반)
   - 서비스 환경: 연구(논문 등) / 사용 URL: `http://localhost` / 활용용도·내용: 교통사고 위험 예측 학습 프로젝트
   - 약관·개인정보 동의 체크 → 신청 → 팝업에서 **확인**
   - ※ Claude 앱 안의 브라우저에서는 확인 팝업이 안 떠서 신청이 안 돼요. 크롬·엣지에서 하세요.
3. 위 메뉴 **인증키 관리** 에서 키 복사 → `SEOUL_API_KEY`

## 3. `.env` 만들기

`road-risk-layer` 폴더(맨 위)에 `.env` 파일을 만들고:

```
KAKAO_REST_KEY=카카오_REST_API_키
KAKAO_MAP_REST_KEY=카카오_REST_API_키
KAKAO_JS_KEY=카카오_JavaScript_키
SEOUL_API_KEY=서울_인증키
```

## 4. 실행

```bash
pip install flask lightgbm pandas numpy requests pyproj   # 처음 한 번
cd road-risk-layer
dvc pull                        # 모델·서비스용 표 받기
python web/app.py               # → http://localhost:5000
```
