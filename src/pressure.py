"""
STEP 5-6 - Pressure Index (leverage) + high-pressure flag + validation.
Reads  outputs/wp.pkl
Writes outputs/pressure.pkl, outputs/pressure_by_over.png, outputs/pressure_validation.csv
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
b = pd.read_pickle(OUT / "wp.pkl")
b["is_t20i"] = (b.competition == "T20I").astype(int)
b["rrr_f"] = b.rrr.fillna(36)

# ---- same feature sets / settings as 02_win_prob.py (keep them in sync) ----
FEATS = {1: (["balls_left", "wkts_left", "score_before", "is_t20i"], [0, 1, 1, 0]),
         2: (["balls_left", "wkts_left", "runs_needed", "rrr_f", "is_t20i"], [1, 1, -1, -1, 0])}
PARAMS = {1: dict(depth=3, iters=150, leaf=1000), 2: dict(depth=4, iters=300, leaf=200)}

# ---------- 1. fit one model per innings on ALL data (used only to ask "what if?") ----------
models = {}
for inn, (cols, mono) in FEATS.items():
    d = b[b.innings == inn]
    p = PARAMS[inn]
    models[inn] = HistGradientBoostingClassifier(
        max_depth=p["depth"], learning_rate=0.05, max_iter=p["iters"], min_samples_leaf=p["leaf"],
        l2_regularization=5.0, monotonic_cst=mono, random_state=42).fit(d[cols], d.bat_win)

# ---------- 2. how likely is each outcome of the next ball, in each kind of state? ----------
OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
def outcome_label(df):
    r = df.total_runs.clip(upper=7).replace({5: 4, 7: 6})
    return np.where(df.team_wkt, "W", r.astype(int).astype(str))
b["wkt_bucket"] = pd.cut(b.wkts_left, [0, 3, 6, 10], labels=["1-3", "4-6", "7-10"])
legal = b[b.legal].copy()
legal["outcome"] = outcome_label(legal)
probs = (legal.groupby(["innings", "phase", "wkt_bucket"], observed=True).outcome
         .value_counts(normalize=True).unstack(fill_value=0).reindex(columns=OUTCOMES, fill_value=0))
P = b[["innings", "phase", "wkt_bucket"]].join(probs, on=["innings", "phase", "wkt_bucket"])[OUTCOMES].fillna(0).values

# ---------- 3. leverage: expected |change in win prob| on the NEXT ball ----------
def wp_after_outcome(d, o):
    """Win probability (batting team) if outcome o happens on a legal ball in state d."""
    r = 0 if o == "W" else int(o)
    wk = d.wkts_left - (1 if o == "W" else 0)
    bl = (d.balls_left - 1).clip(lower=0)
    inn = d.innings.iloc[0]
    if inn == 1:
        sc = d.score_before + r
        X = pd.DataFrame({"balls_left": bl, "wkts_left": wk, "score_before": sc,
                          "is_t20i": d.is_t20i})
        return models[1].predict_proba(X[FEATS[1][0]])[:, 1]
    need = d.runs_needed - r
    rrr = np.where(bl > 0, need * 6 / bl.replace(0, np.nan), 36)
    rrr = pd.Series(rrr, index=d.index).clip(upper=36).fillna(36)
    sc = d.score_before + r
    X = pd.DataFrame({"balls_left": bl, "wkts_left": wk, "runs_needed": need, "rrr_f": rrr,
                      "is_t20i": d.is_t20i})
    wp = models[2].predict_proba(X[FEATS[2][0]])[:, 1]
    wp = np.where(need <= 0, 1.0, wp)                      # target reached -> won
    wp = np.where((need > 0) & ((wk <= 0) | (bl <= 0)), 0.0, wp)   # all out / balls gone -> lost
    return wp

b["li_raw"] = np.nan
for inn in (1, 2):
    d = b[b.innings == inn]
    swing = np.zeros(len(d))
    for j, o in enumerate(OUTCOMES):
        swing += P[b.index.get_indexer(d.index), j] * np.abs(wp_after_outcome(d, o) - d.wp_before.values)
    b.loc[d.index, "li_raw"] = swing

# Pressure Index = leverage relative to the average ball in the whole dataset (1.0 = typical ball,
# 2.0 = the next ball can move the game twice as much as a typical one). ONE scale for both innings.
b["pi"] = b.li_raw / b.li_raw.mean()
b["pi_pct"] = b.pi.rank(pct=True)                             # percentile (used for sensitivity tests later)
HIGH_Q = 0.85                                                 # top 15% of balls = high pressure
PI_THRESHOLD = b.pi.quantile(HIGH_Q)
b["high_pressure"] = b.pi > PI_THRESHOLD
print(f"High-pressure threshold: PI > {PI_THRESHOLD:.2f}  ({b.high_pressure.mean():.1%} of balls)")
print("High-pressure balls by innings/phase (share of that group):")
print(b.groupby(["innings", "phase"], observed=True).high_pressure.mean().round(3).unstack())

# ---------- 4. validation: does it separate pressure from ordinary? ----------
bat = b[b.batter_faced].copy()
def summarise(d):
    n_leg = d.legal.sum()
    return pd.Series({
        "balls": len(d),
        "bat_SR": 100 * d.bat_runs.sum() / max(d.batter_faced.sum(), 1),
        "dot%": 100 * ((d.total_runs == 0) & d.legal).sum() / max(n_leg, 1),
        "boundary%": 100 * (d.is_four | d.is_six).sum() / max(n_leg, 1),
        "wkts/100 balls": 100 * d.team_wkt.sum() / max(n_leg, 1),
        "economy": 6 * d.bowler_runs.sum() / max(n_leg, 1),
        "mean|WPA|": d.wpa_bat.abs().mean(),
        "mean PI": d.pi.mean()})
val = b.groupby(["innings", "high_pressure"]).apply(summarise, include_groups=False).round(3)
print("=== High vs normal pressure (all phases) ===\n", val.to_string())
valp = b.groupby(["innings", "phase", "high_pressure"], observed=True).apply(summarise, include_groups=False).round(3)
print("\n=== Same, split by phase (controls for stage of innings) ===\n", valp[["balls", "bat_SR", "dot%", "wkts/100 balls", "mean PI"]].to_string())
val.to_csv(OUT / "pressure_validation.csv")

print("\nShare of high-pressure balls by phase (innings 2):")
print(b[b.innings == 2].groupby("phase", observed=True).high_pressure.mean().round(3).to_dict())
print("Mean PI by chase situation (innings 2):")
c = b[b.innings == 2].copy()
c["gap"] = pd.cut(c.rrr - c.crr, [-100, -2, 0, 2, 4, 100], labels=["RRR<<CRR", "RRR<CRR", "RRR~CRR", "RRR>CRR", "RRR>>CRR"])
print(c.groupby("gap", observed=True).pi.mean().round(2).to_dict())
print("Mean PI by wickets left:", b.groupby("wkts_left").pi.mean().round(2).to_dict())
print("Mean PI knockout vs league:", b.groupby(b.knock_out.astype(bool)).pi.mean().round(3).to_dict())

# ---------- 5. chart: pressure by over ----------
b["over"] = (b.legal_before // 6 + 1).clip(upper=20)
fig, ax = plt.subplots(figsize=(8, 4.5))
for inn, lab in ((1, "Innings 1 (setting)"), (2, "Innings 2 (chasing)")):
    t = b[b.innings == inn].groupby("over").pi.mean()
    ax.plot(t.index, t.values, marker="o", label=lab)
ax.axhline(1, color="k", ls="--", lw=1)
ax.set(xlabel="Over", ylabel="Mean Pressure Index (1 = typical ball)", title="How pressure builds through an innings")
ax.legend(); plt.tight_layout(); plt.savefig(OUT / "pressure_by_over.png", dpi=120)

b.to_pickle(OUT / "pressure.pkl")
print("\nSaved outputs/pressure.pkl, pressure_by_over.png, pressure_validation.csv")