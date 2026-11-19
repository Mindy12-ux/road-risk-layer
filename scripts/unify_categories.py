"""
사고 난 데이터(1)와 사고 안 난 데이터(0)의 날씨·노면 이름을 똑같이 맞춥니다.

왜? 한쪽에만 있는 이름(예: '흐림')이 있으면 모델이
"흐림이면 무조건 사고"처럼 엉터리 규칙을 외워버립니다.

사용법 (프로젝트 폴더에서):
    python3 scripts/unify_categories.py

입력: data/raw/taas/taas_accidents_사망중상경상.csv   사고 난 데이터 (원본)
    data/processed/negatives_사고안난.csv           사고 안 난 데이터
출력: data/processed/사고 안 난 + 난_정리.csv          이름을 맞춘 합본
"""

import os

import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
POS_PATH = os.path.join(ROOT, "data", "raw", "taas", "taas_accidents_사망중상경상.csv")
NEG_PATH = os.path.join(ROOT, "data", "processed", "negatives_사고안난.csv")
OUT_PATH = os.path.join(ROOT, "data", "processed", "사고 안 난 + 난_정리.csv")

COLUMNS = ["사고여부", "연도", "발생년월", "기준날짜", "요일", "발생시", "사고내용",
        "기상상태", "노면상태", "도로형태", "위도", "경도"]

# 두 데이터 모두 이 이름만 쓰도록 바꿉니다.
WEATHER_MAP = {
    "맑음": "맑음",
    "흐림": "맑음",   # 날씨 데이터는 하늘이 흐린지 정확히 경찰 기록과 맞출 수 없어서 '비·눈 안 옴'으로 합침
    "안개": "맑음",   # 6년 동안 12건뿐이라 따로 두기 어려움
    "비": "비",
    "눈": "눈",
}
SURFACE_MAP = {
    "건조": "건조",
    "젖음/습기": "젖음/습기",
    "침수": "젖음/습기",   # 물에 잠김 → 젖은 노면 (18건)
    "해빙": "젖음/습기",   # 얼음이 녹는 중 → 젖은 노면 (11건)
    "서리/결빙": "서리/결빙",
    "적설": "적설",
}
# '기타'는 무슨 날씨·노면인지 알 수 없고, 사고 난 데이터에만 있어서 뺍니다.


def main():
    pos = pd.read_csv(POS_PATH, encoding="utf-8-sig", dtype=str)
    pos = pos.assign(사고여부="1", 기준날짜="")[COLUMNS]
    neg = pd.read_csv(NEG_PATH, encoding="utf-8-sig", dtype=str)[COLUMNS]
    both = pd.concat([pos, neg], ignore_index=True)

    before = both["사고여부"].value_counts()
    both["기상상태"] = both["기상상태"].map(WEATHER_MAP)   # 표에 없는 이름(기타)은 빈칸이 됨
    both["노면상태"] = both["노면상태"].map(SURFACE_MAP)
    both = both.dropna(subset=["기상상태", "노면상태"])
    after = both["사고여부"].value_counts()

    print("빠진 줄 ('기타'): 사고 난 데이터 "
        f"{before['1'] - after['1']:,}줄, 사고 안 난 데이터 {before['0'] - after['0']:,}줄")
    print(f"남은 줄: 사고 난 {after['1']:,} + 사고 안 난 {after['0']:,} = {len(both):,}")

    for col in ["기상상태", "노면상태", "도로형태"]:
        t = pd.crosstab(both[col], both["사고여부"]).rename(columns={"1": "사고 난", "0": "사고 안 난"})
        missing = t[(t == 0).any(axis=1)]
        print(f"\n[{col}] 한쪽에만 있는 이름: {list(missing.index) or '없음'}")
        print(t[["사고 난", "사고 안 난"]].to_string())

    both.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n완료. 저장 위치: {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
