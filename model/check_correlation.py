"""모델 학습 전 상관관계 점검 (data/model = dataset_v2 형식 + 팀원 변수).

확인하는 것
1. 피처끼리 상관 (Spearman) → 서로 너무 비슷한 변수(다중공선성) 찾기. |r| ≥ 0.8이면 표시.
2. 피처와 라벨의 관계
   - 사고 여부(0/1)와의 상관 : 1단계 모델용
   - 사고 행만 모았을 때 사고심각도(1~3)와의 상관 : 2단계 모델용
3. 범주형 변수(요일·날씨·노면·도로형태)는 상관계수가 의미 없어서
   범주별 사고 비율과 큰 사고(중상+사망) 비율, Cramér's V로 따로 봄.

4. 변수별 빈 값 비율 (붙인 변수가 일부 날짜·칸에 없을 때)

사용 (저장소 맨 위 폴더에서):
    pip install pandas numpy matplotlib scipy
    python model/check_correlation.py                       # data/model → results/corr
    python model/check_correlation.py --data dataset_v2폴더 --out results/corr_v2   # 다른 폴더
    (기본은 train.csv만 봄. 학습 데이터로만 판단해야 검증·테스트가 깨끗하게 남음)
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import chi2_contingency

TARGET = "사고심각도"
# 숫자로 적혀 있지만 크기 순서가 없는 범주형 변수 (있는 것만 씀)
CAT_ALL = ["요일", "기상상태", "노면상태", "도로_큰분류", "도로_작은분류", "칸_최고도로등급", "칸_속도출처"]
ROOT = Path(__file__).resolve().parents[1]


def find_csv(data_dir, k):
    """train_v2.csv처럼 _v2가 붙은 이름을 먼저 찾고, 없으면 train.csv를 씀."""
    for name in (f"{k}_v2.csv", f"{k}.csv"):
        p = Path(data_dir) / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{data_dir}에 {k}_v2.csv 또는 {k}.csv가 없어요. --data로 폴더를 지정하세요.")


def set_korean_font():
    names = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"]:
        if f in names:
            matplotlib.rcParams["font.family"] = f
            break
    else:
        print("※ 한글 폰트를 찾지 못해 그림의 한글이 깨질 수 있어요 (Windows: 맑은 고딕이면 자동 적용).")
    matplotlib.rcParams["axes.unicode_minus"] = False


def heatmap(c, title, path):
    n = len(c)
    fig, ax = plt.subplots(figsize=(max(11, n * 0.45), max(9, n * 0.4)))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.iat[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7 if n <= 25 else 5,
                    color="white" if abs(v) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cramers_v(x, y):
    t = pd.crosstab(x, y)
    chi2 = chi2_contingency(t, correction=False)[0]
    n = t.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(t.shape) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "model"), help="train/valid/test.csv가 있는 폴더 (기본: data/model)")
    ap.add_argument("--out", default=str(ROOT / "results" / "corr"))
    ap.add_argument("--split", default="train", help="볼 파일 (train / valid / test / all)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    set_korean_font()

    rd = lambda k: pd.read_csv(find_csv(a.data, k), encoding="utf-8-sig")
    df = pd.concat([rd(k) for k in ["train", "valid", "test"]]) if a.split == "all" else rd(a.split)
    df["사고여부"] = (df[TARGET] > 0).astype(int)
    feats = [c for c in df.columns if c not in (TARGET, "사고여부")]
    CAT = [c for c in CAT_ALL if c in feats] + [c for c in feats if df[c].dtype == object and c not in CAT_ALL]
    for c in CAT:
        if df[c].dtype == object:
            df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
    num = [c for c in feats if c not in CAT]
    print(f"{a.split}: {len(df):,}행, 피처 {len(feats)}개")

    # 0) 빈 값 / 값이 하나뿐인 변수
    na = df[feats].isna().mean()
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    if (na > 0).any():
        print("\n[0] 빈 값 비율 (높으면 그 변수는 효과가 약할 수 있음)")
        print(na[na > 0].sort_values(ascending=False).map("{:.1%}".format).to_string())
    if const:
        print(f"\n[0] 값이 하나뿐이라 쓸모없는 변수 → 빼세요: {const}")
    na.rename("빈값비율").round(4).to_csv(out / "0_missing.csv", encoding="utf-8-sig")

    # 1) 피처끼리 상관
    c = df[feats].corr(method="spearman")
    heatmap(c, "피처 간 Spearman 상관", out / "1_feature_corr.png")
    c.round(3).to_csv(out / "1_feature_corr.csv", encoding="utf-8-sig")
    pairs = (c.where(np.triu(np.ones(c.shape, bool), 1)).stack().rename("r").reset_index()
             .rename(columns={"level_0": "변수1", "level_1": "변수2"}))
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)
    print("\n[1] 피처끼리 상관 상위 15 (|r| ≥ 0.8이면 둘 중 하나 빼는 걸 검토)")
    print(pairs.head(15).round(3).to_string(index=False))
    pairs.round(3).to_csv(out / "1_feature_pairs.csv", index=False, encoding="utf-8-sig")

    # 2) 피처와 라벨
    acc = df[df["사고여부"] == 1]
    t = pd.DataFrame({
        "사고여부_spearman": df[num].corrwith(df["사고여부"], method="spearman"),
        "심각도(사고행)_spearman": acc[num].corrwith(acc[TARGET], method="spearman"),
    }).sort_values("사고여부_spearman", key=abs, ascending=False)
    print("\n[2] 수치형 피처와 라벨의 상관")
    print(t.round(3).to_string())
    t.round(4).to_csv(out / "2_target_corr.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(t) * 0.3)), sharey=True)
    for ax, col in zip(axes, t.columns):
        ax.barh(t.index, t[col], color=np.where(t[col] > 0, "#c0392b", "#2c7fb8"))
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(col)
    axes[0].invert_yaxis()
    fig.tight_layout(); fig.savefig(out / "2_target_corr.png", dpi=150); plt.close(fig)

    # 3) 범주형 변수: 범주별 비율 + Cramér's V
    rows = []
    print("\n[3] 범주형 변수 (사고 비율 = 그 범주에서 사고 행 비율, 큰 사고 비율 = 사고 중 중상+사망 비율)")
    for col in CAT:
        g = df.groupby(col).agg(행수=("사고여부", "size"), 사고비율=("사고여부", "mean"))
        g["큰사고비율(사고중)"] = acc.groupby(col)[TARGET].apply(lambda s: (s >= 2).mean())
        print(f"\n- {col}\n{g.round(3).to_string()}")
        g.round(4).to_csv(out / f"3_cat_{col}.csv", encoding="utf-8-sig")
        rows.append({"변수": col, "CramerV_사고여부": cramers_v(df[col], df["사고여부"]),
                     "CramerV_심각도(사고행)": cramers_v(acc[col], acc[TARGET])})
    v = pd.DataFrame(rows).set_index("변수").round(3)
    print("\nCramér's V (0~1, 0.1 미만이면 관계가 약함)\n", v.to_string())
    v.to_csv(out / "3_cramers_v.csv", encoding="utf-8-sig")

    # 4) 라벨 분포
    print("\n[4] 사고심각도 분포\n", df[TARGET].value_counts().sort_index().rename("행수").to_frame()
          .assign(비율=lambda x: (x.행수 / x.행수.sum()).round(4)).to_string())
    print(f"\n결과 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
