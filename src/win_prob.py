"""
STEP 4 - Win-probability model.
Reads  outputs/state.pkl   (made by prepare_data.py)
Writes outputs/wp.pkl      (same table + wp_before, wp_after, wpa_bat)
       outputs/calibration.png
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
b = pd.read_pickle(OUT / "state.pkl")
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# ---------- 1. features (ONLY what is known before the ball) ----------
b["is_t20i"] = (b.competition == "T20I").astype(int)
b["rrr_f"] = b.rrr.fillna(36)              # balls_left == 0 -> maximum required rate

# Innings 1 never uses inn1_total (that would leak the final score).
FEATS = {
    1: (["balls_left", "wkts_left", "score_before", "is_t20i"], [0, 1, 1, 0]),
    2: (["balls_left", "wkts_left", "runs_needed", "rrr_f", "is_t20i"], [1, 1, -1, -1, 0]),
}
# Innings 1 is a weak predictor (the game is far from over) -> keep that model small and smooth.
PARAMS = {1: dict(depth=3, iters=150, leaf=1000), 2: dict(depth=4, iters=300, leaf=200)}

# State AFTER the ball, used only for the final ball of innings 1 (see section 3)
a = pd.DataFrame(index=b.index)
a["balls_left"]   = (b.balls_left - b.legal.astype(int)).clip(lower=0)
a["wkts_left"]    = b.wkts_left - b.team_wkt.astype(int)
a["score_before"] = b.score_before + b.total_runs
a["is_t20i"]      = b.is_t20i
is_last = ~b.duplicated(["match_id", "innings"], keep="last")      # last ball of each innings

# ---------- 2. out-of-fold predictions, split BY MATCH ----------
b["wp_before"] = np.nan
b["wp_after"] = np.nan
gkf = GroupKFold(n_splits=5)
for inn, (cols, mono) in FEATS.items():
    idx = b.index[b.innings == inn]
    X, y, grp = b.loc[idx, cols], b.loc[idx, "bat_win"], b.loc[idx, "match_id"]
    oof = pd.Series(np.nan, index=idx)
    for tr, te in gkf.split(X, y, grp):
        p = PARAMS[inn]
        mdl = HistGradientBoostingClassifier(
            max_depth=p["depth"], learning_rate=0.05, max_iter=p["iters"], min_samples_leaf=p["leaf"],
            l2_regularization=5.0, monotonic_cst=mono, random_state=42)
        mdl.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        if inn == 1:   # last ball of innings 1: score the END state with the SAME model
            li = X.index[te][is_last.loc[X.index[te]].values]
            b.loc[li, "wp_after"] = mdl.predict_proba(a.loc[li, cols])[:, 1]
    b.loc[idx, "wp_before"] = oof
    print(f"Innings {inn}: logloss={log_loss(y, oof):.3f} (coin-flip 0.693) | "
          f"brier={brier_score_loss(y, oof):.3f} | AUC={roc_auc_score(y, oof):.3f}")

# ---------- 3. win prob AFTER each ball -> WPA ----------
# Every ball except the last of an innings: after = next ball's "before".
nxt = b.groupby(["match_id", "innings"]).wp_before.shift(-1)
b["wp_after"] = b.wp_after.fillna(nxt)
# Last ball of innings 2: the match is over, so the outcome is known.
m2 = is_last & (b.innings == 2)
b.loc[m2, "wp_after"] = b.loc[m2, "bat_win"]
# (The jump from "end of innings 1" to "start of innings 2" belongs to no ball, so nobody gets credit for it.)
b["wpa_bat"] = b.wp_after - b.wp_before        # positive = good for the batting team
b = b.dropna(subset=["wp_before", "wp_after"])

# ---------- 4. sanity checks ----------
print("\nMean wp_before at ball 1  - innings 1:", round(b[(b.innings==1)&(b.legal_before==0)].wp_before.mean(), 3),
      "| innings 2:", round(b[(b.innings==2)&(b.legal_before==0)].wp_before.mean(), 3))
print("Mean |WPA| per ball by phase:")
print(b.groupby(["innings", "phase"], observed=True).wpa_bat.apply(lambda s: s.abs().mean()).round(4).unstack())
print("Mean WPA on a wicket ball:", round(b[b.team_wkt].wpa_bat.mean(), 4),
      "| on a six:", round(b[b.is_six].wpa_bat.mean(), 4))
lb = b[~b.duplicated(["match_id", "innings"], keep="last")]
print("Last-ball mean |WPA|  innings 1:", round(lb[lb.innings==1].wpa_bat.abs().mean(), 3),
      "| innings 2:", round(lb[lb.innings==2].wpa_bat.abs().mean(), 3))

# ---------- 5. calibration plot ----------
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
for axis, inn in zip(ax, (1, 2)):
    d = b[b.innings == inn]
    bins = pd.cut(d.wp_before, np.linspace(0, 1, 11))
    t = d.groupby(bins, observed=True).agg(pred=("wp_before", "mean"), actual=("bat_win", "mean"))
    axis.plot([0, 1], [0, 1], "k--", lw=1)
    axis.plot(t.pred, t.actual, "o-")
    axis.set(title=f"Innings {inn}", xlabel="Predicted win prob", xlim=(0, 1), ylim=(0, 1))
ax[0].set_ylabel("Actual win rate")
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=120)

b.to_pickle(OUT / "wp.pkl")
print("\nSaved outputs/wp.pkl and outputs/calibration.png")