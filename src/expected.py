"""
STEP 7 - Context-adjusted expectations.
For every ball: what would an AVERAGE player be expected to do here, and how far was the actual result from that?
Reads  outputs/pressure.pkl
Writes outputs/residuals.pkl
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
b = pd.read_pickle(OUT / "pressure.pkl")

# ---------- 1. clean attribution of wickets ----------
b["spin"] = (b.pace_or_spin == "Spin").astype(int)
b["striker_out"] = b.batter_out & (b.dismissed_player_id == b.striker_id)       # THIS batter was dismissed
b["nonstriker_ro"] = b.batter_out & (b.dismissed_player_id != b.striker_id)     # non-striker run out: not the striker's failure

# Context features: the SITUATION only. Nothing about who the player is, so the baseline is "an average player".
# 'pi' is included on purpose: a player is compared with peers facing the SAME level of pressure.
CTX = ["legal_before", "wkts_left", "innings", "is_t20i", "spin", "pi"]

def oof_predict(df, target, kind):
    """Out-of-fold prediction of `target` from CTX, split BY MATCH (no leakage)."""
    pred = pd.Series(np.nan, index=df.index)
    Model = HistGradientBoostingRegressor if kind == "reg" else HistGradientBoostingClassifier
    for tr, te in GroupKFold(n_splits=5).split(df, groups=df.match_id):
        m = Model(max_depth=4, learning_rate=0.05, max_iter=200, min_samples_leaf=500,
                  l2_regularization=5.0, random_state=42)
        m.fit(df.iloc[tr][CTX], df.iloc[tr][target])
        pred.iloc[te] = m.predict(df.iloc[te][CTX]) if kind == "reg" else m.predict_proba(df.iloc[te][CTX])[:, 1]
    return pred

# ---------- 2. expectations ----------
bf = b[b.batter_faced]                                     # balls the batter actually faced (no wides)
b["x_bat_runs"] = oof_predict(bf, "bat_runs", "reg").reindex(b.index)
b["x_striker_out"] = oof_predict(b, "striker_out", "clf")
b["x_bowl_runs"] = oof_predict(b, "bowler_runs", "reg")
b["x_bowl_wkt"] = oof_predict(b, "bowler_wkt", "clf")

# ---------- 3. residuals: actual minus expected ----------
b["bat_rae"]  = b.bat_runs - b.x_bat_runs                  # batter: runs above expected (NaN on wides)
b["bat_out_ae"] = b.striker_out.astype(float) - b.x_striker_out   # batter: dismissal above expected (+ = got out more than expected)
b["bowl_rae"] = b.x_bowl_runs - b.bowler_runs              # bowler: runs SAVED vs expected (+ = good)
b["bowl_wae"] = b.bowler_wkt.astype(float) - b.x_bowl_wkt  # bowler: wickets above expected (+ = good)
# Value to the bowler's side is the mirror image of the batting side's win-probability change
b["wpa_bowl"] = -b.wpa_bat
b.loc[b.nonstriker_ro, "bat_out_ae"] = np.nan              # not the striker's fault -> ignore for batter metrics

# ---------- 4. sanity checks ----------
print("Mean residuals (should be ~0):")
print(b[["bat_rae", "bat_out_ae", "bowl_rae", "bowl_wae"]].mean().round(4).to_dict())
print("\nMean residual by pressure group (still ~0 => baseline is fair at every pressure level):")
print(b.groupby(["innings", "high_pressure"])[["bat_rae", "bat_out_ae", "bowl_rae", "bowl_wae"]].mean().round(4))

print("\nExpected vs actual runs per ball faced, by expected-runs quintile:")
q = pd.qcut(b.x_bat_runs, 5, duplicates="drop")
print(b[b.batter_faced].groupby(q, observed=True).agg(expected=("x_bat_runs", "mean"), actual=("bat_runs", "mean")).round(3))
print("\nExpected vs actual dismissal rate (per 100 balls) by quintile:")
q2 = pd.qcut(b.x_striker_out, 5, duplicates="drop")
print((100 * b.groupby(q2, observed=True).agg(expected=("x_striker_out", "mean"), actual=("striker_out", "mean"))).round(2))

print("\nAverage expected runs per ball by phase and pressure:")
print(b[b.batter_faced].groupby(["phase", "high_pressure"], observed=True).x_bat_runs.mean().round(3).unstack())

# ---------- 5. save ----------
keep = ["match_id", "innings", "innings_ball_num", "match_date", "season", "competition", "knock_out", "stage",
        "phase", "batting_team_id", "bowling_team_id", "striker_id", "striker_name", "bowler_id", "bowler_name",
        "pace_or_spin", "legal", "batter_faced", "bat_runs", "bowler_runs", "striker_out", "bowler_wkt",
        "is_four", "is_six", "total_runs", "wp_before", "wpa_bat", "wpa_bowl", "pi", "high_pressure",
        "x_bat_runs", "x_striker_out", "x_bowl_runs", "x_bowl_wkt",
        "bat_rae", "bat_out_ae", "bowl_rae", "bowl_wae", "nonstriker_ro", "bat_win"]
b[keep].to_pickle(OUT / "residuals.pkl")
print("\nSaved outputs/residuals.pkl", b[keep].shape)