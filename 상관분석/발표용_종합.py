"""
발표용 종합 그래프 - 상관분석 1단계(정답과의 관계) + 2단계(변수끼리 겹침)를 슬라이드 2장으로 정리

먼저 아래 6개를 실행해서 결과 CSV를 만든 뒤 이 파일을 실행합니다 (저장소 맨 위 폴더에서):
    python 상관분석/target_sc/spearman.py
    python 상관분석/target_sc/phi.py
    python 상관분석/target_sc/V.py
    python 상관분석/target_sc/anova.py
    python 상관분석/multi/pearson.py
    python 상관분석/multi/VIF.py
    python 상관분석/발표용_종합.py

출력: reports/발표용/
    01_정답과의_관계_종합.png   변수 하나하나가 사고여부와 얼마나 관계있나 (기법 4개를 한 장에)
    02_다중공선성_정리.png      피어슨 높은 짝 + VIF 정리 전/후
    정답과의_관계_종합.csv

기법 고르는 규칙 (변수 종류 → 기법):
    숫자·순서 변수  → 스피어만 r (순위가 같이 오르내리나)  + ANOVA η² (사고 난/안 난 그룹 평균 차이)
    0/1 변수        → 파이 φ
    범주(이름) 변수 → 크라메르 V
    r, φ, V는 같은 Cohen 기준(0.1 약함 / 0.3 중간 / 0.5 강함)이라 한 축에 같이 그릴 수 있음.
    η²는 '제곱' 값이라 기준이 달라서(0.01 / 0.06 / 0.14) 오른쪽 칸에 따로 그림.
"""

import os
import platform

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = {"Windows": "Malgun Gothic", "Darwin": "AppleGothic"}.get(platform.system(), "NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # 저장소 맨 위 폴더
REP = f"{ROOT}/reports"
OUT_DIR = f"{REP}/발표용"
os.makedirs(OUT_DIR, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
COLOR = {"스피어만 r": "#2a78d6", "파이 φ": "#eb6834", "크라메르 V": "#1baf7a"}


def read(path):
    full = f"{REP}/{path}"
    if not os.path.exists(full):
        raise SystemExit(f"[없음] {full}\n  → 파일 맨 위 설명의 6개 스크립트를 먼저 실행하세요.")
    return pd.read_csv(full, encoding="utf-8-sig")


sp = read("target_anal_re/spearman/스피어만_결과.csv")
ph = read("target_phi/파이_결과.csv")
cv = read("target_V/크라메르V_결과.csv")
an = read("target_anova/ANOVA_결과.csv")
vif = read("multi_VIF/VIF_결과.csv").rename(columns={"Unnamed: 0": "칸"})
pairs = read("multi_pearson/피어슨_높은짝.csv")

# ── 1. 한 표로 합치기 ─────────────────────────────────────────
rows = []
for _, r in sp.iterrows():
    if r["변수"] == "요일":          # 요일은 범주로 보는 게 맞아서 크라메르 V 쪽 값을 씀
        continue
    rows.append({"변수": r["변수"], "기법": "스피어만 r", "값": r["스피어만_r"]})
for _, r in ph.iterrows():
    rows.append({"변수": r["변수"], "기법": "파이 φ", "값": r["파이_φ"]})
for _, r in cv.iterrows():
    rows.append({"변수": r["변수"], "기법": "크라메르 V", "값": r["크라메르_V"]})
df = pd.DataFrame(rows)
df["크기"] = df["값"].abs()
df = df.merge(an[["변수", "η²"]], on="변수", how="left")


def strength(a):
    return "강함" if a >= 0.5 else "중간" if a >= 0.3 else "약함" if a >= 0.1 else "거의 없음"


df["판단"] = df["크기"].map(strength)
df = df.sort_values("크기", ascending=False).reset_index(drop=True)
df.round(4).to_csv(f"{OUT_DIR}/정답과의_관계_종합.csv", index=False, encoding="utf-8-sig")
print(df.round(3).to_string(index=False))

# ── 2. 그래프 ① 정답과의 관계 (왼쪽 r·φ·V / 오른쪽 η²) ─────────────
plot = df.iloc[::-1].reset_index(drop=True)          # 큰 값이 위로
h = max(7, 0.28 * len(plot) + 1.8)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, h), sharey=True,
                            gridspec_kw={"width_ratios": [3, 1.3], "wspace": 0.12})

ax.barh(plot.index, plot["크기"], color=[COLOR[m] for m in plot["기법"]], height=0.7)
for y, (v, raw) in enumerate(zip(plot["크기"], plot["값"])):
    ax.text(v + 0.005, y, f"{raw:.2f}".replace("−", "-"), va="center", fontsize=7.5, color=INK)
lim = max(0.45, plot["크기"].max() * 1.15)
for t, name in [(0.1, "약함"), (0.3, "중간"), (0.5, "강함")]:
    if t < lim:
        ax.axvline(t, color=MUTED, ls="--", lw=0.8)
        ax.text(t, len(plot) - 0.3, f" {name} {t}", color=MUTED, fontsize=8, va="bottom")
ax.set_xlim(0, lim)
ax.set_yticks(plot.index)
ax.set_yticklabels(plot["변수"], fontsize=8)
ax.set_xlabel("관계 크기 |r|, |φ|, V   (숫자 = 부호 포함 값, 음수면 클수록 사고 감소)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR.values()]
ax.legend(handles, ["스피어만 r (숫자 변수)", "파이 φ (0/1 변수)", "크라메르 V (범주 변수)"],
        loc="lower right", frameon=False, fontsize=9)

e = plot["η²"]
ax2.barh(plot.index, e.fillna(0), color=COLOR["스피어만 r"], alpha=0.55, height=0.7)
for y, v in enumerate(e):
    if pd.notna(v):
        ax2.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=7, color=INK)
elim = max(0.16, e.max() * 1.25)
for t in (0.01, 0.06, 0.14):
    ax2.axvline(t, color=MUTED, ls="--", lw=0.8)
ax2.set_xlim(0, elim)
ax2.set_xlabel("ANOVA η²  (점선 0.01 / 0.06 / 0.14)")
ax2.tick_params(axis="y", length=0)

for a in (ax, ax2):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="x", color=GRID, lw=0.6)
    a.set_axisbelow(True)
fig.suptitle("상관분석 1단계: 변수별로 사고여부와 얼마나 관계있나 (train)", x=0.01, y=0.995, ha="left", fontsize=13)
fig.subplots_adjust(top=1 - 0.9 / h)
plt.savefig(f"{OUT_DIR}/01_정답과의_관계_종합.png", dpi=200, bbox_inches="tight")
plt.close()

# ── 3. 그래프 ② 다중공선성: 높은 짝 + VIF 정리 전/후 ───────────────
LV = {"심각": "#d93025", "주의": "#eb6834", "괜찮음": "#2a78d6"}
rcol = [c for c in pairs.columns if "피어슨" in c or c == "r"][0]
top = pairs.assign(a=pairs[rcol].abs()).sort_values("a", ascending=False).head(9).iloc[::-1]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.8, 1]})
lvl = top["a"].map(lambda a: "심각" if a >= 0.9 else "주의" if a >= 0.7 else "괜찮음")
ax.barh(range(len(top)), top["a"], color=[LV[l] for l in lvl], height=0.65)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{a} ↔ {b}" for a, b in zip(top["변수1"], top["변수2"])], fontsize=8)
for y, v in enumerate(top[rcol]):
    ax.text(abs(v) + 0.01, y, f"{v:.2f}".replace("−", "-"), va="center", fontsize=8)
for t in (0.7, 0.9):
    ax.axvline(t, color=MUTED, ls="--", lw=0.8)
ax.set_xlim(0, 1.1)
ax.set_xlabel("|피어슨 r|  (점선: 0.7 주의, 0.9 심각)")
ax.set_title("① 두 개씩 비교 (피어슨): 많이 겹치는 짝", loc="left", fontsize=11)

cnt = pd.DataFrame({
    "정리 전": vif["판단_정리전"].value_counts(),
    "정리 후": vif["판단_정리후"].value_counts(),
}).reindex(["심각", "주의", "괜찮음"]).fillna(0).astype(int)
x = range(2)
bottom = [0, 0]
for lv in ["괜찮음", "주의", "심각"]:
    vals = cnt.loc[lv].tolist()
    ax2.bar(x, vals, bottom=bottom, color=LV[lv], width=0.55, edgecolor="white", linewidth=2, label=lv)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v:
            ax2.text(i, b + v / 2, f"{lv} {v}", ha="center", va="center", fontsize=9, color="white")
    bottom = [b + v for b, v in zip(bottom, vals)]
ax2.set_xticks(list(x))
ax2.set_xticklabels(["정리 전", f"정리 후\n(겹치는 칸 {int((vif['판단_정리후'] == '삭제').sum())}개 뺌)"])
ax2.set_ylabel("칸 수")
ax2.set_title("② 하나 vs 나머지 전부 (VIF ≥10 심각)", loc="left", fontsize=11)

for a in (ax, ax2):
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle("상관분석 2단계: 변수끼리 정보가 겹치나 (다중공선성)", x=0.01, ha="left", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_다중공선성_정리.png", dpi=200, bbox_inches="tight")
plt.close()

print(f"\n저장 위치: {OUT_DIR}")
