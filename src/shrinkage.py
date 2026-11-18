"""
STEP 9 - Fair comparison: opposition adjustment, eligibility, empirical-Bayes shrinkage.
Reads  outputs/residuals.pkl
Writes outputs/batter_scores.csv, outputs/bowler_scores.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
r = pd.read_pickle(OUT / "residuals.pkl")

# ---- eligibility rules (change here, re-run, and see how rankings move: this is your sensitivity test) ----
MIN_HP_BALLS = 100        # pressure balls faced / bowled
MIN_HP_MATCHES = 15       # ...spread over at least this many matches (no 2-innings wonders)
MIN_NORM_BALLS = 200      # needed for a stable personal baseline (the delta)
OPP_SHRINK_K = 1500       # opposition effect is shrunk by n/(n+K): small-sample teams get less adjustment

# =====================================================================================================
# A. OPPOSITION ADJUSTMENT: a run against a weak attack is worth less than a run against a strong one
# =====================================================================================================
def opp_effect(df, team_col, metric):
    g = df.groupby(team_col)[metric]
    return (g.mean() * g.size() / (g.size() + OPP_SHRINK_K))

bat = r[r.batter_faced & ~r.nonstriker_ro].copy()
bat["bat_good_out"] = -bat.bat_out_ae
for m in ("wpa_bat", "bat_rae", "bat_good_out"):                      # batter vs the bowling team he faced
    bat[m + "_adj"] = bat[m] - bat.bowling_team_id.map(opp_effect(bat, "bowling_team_id", m))
bw = r.copy()
for m in ("wpa_bowl", "bowl_rae", "bowl_wae"):                        # bowler vs the batting team he bowled to
    bw[m + "_adj"] = bw[m] - bw.batting_team_id.map(opp_effect(bw, "batting_team_id", m))

# Save the adjusted ball-level data: Step 10 (consistency) uses it
bat[["striker_id", "match_id", "match_date", "competition", "high_pressure", "wpa_bat_adj", "bat_rae_adj", "bat_good_out_adj"]].to_pickle(OUT / "bat_adj.pkl")
bw[["bowler_id", "match_id", "match_date", "competition", "high_pressure", "wpa_bowl_adj", "bowl_rae_adj", "bowl_wae_adj"]].to_pickle(OUT / "bowl_adj.pkl")

# =====================================================================================================
# B. PLAYER TABLES (same layout as Step 8, but on adjusted values and with a standard error for each metric)
# =====================================================================================================
def table(df, pid, name, metrics):
    latest = df.sort_values("match_date").groupby(pid)[name].last().rename("player")
    out = [latest]
    for flag, tag in ((True, "hp"), (False, "norm")):
        g = df[df.high_pressure == flag].groupby(pid)
        out.append(g.size().rename(f"balls_{tag}"))
        out.append(g.match_id.nunique().rename(f"matches_{tag}"))
        for key, col in metrics.items():
            out.append((100 * g[col].mean()).rename(f"{key}_{tag}"))
            out.append((100 * g[col].std() / np.sqrt(g.size())).rename(f"se_{key}_{tag}"))
    t = pd.concat(out, axis=1)
    for key in metrics:                                              # delta = pressure minus own normal level
        t[f"{key}_delta"] = t[f"{key}_hp"] - t[f"{key}_norm"]
        t[f"se_{key}_delta"] = np.sqrt(t[f"se_{key}_hp"] ** 2 + t[f"se_{key}_norm"] ** 2)
    t["ipl_share_hp"] = df[df.high_pressure].groupby(pid).competition.apply(lambda s: (s == "IPL").mean())
    return t.reset_index().rename(columns={pid: "player_id"})

bt = table(bat, "striker_id", "striker_name", {"wpa": "wpa_bat_adj", "rae": "bat_rae_adj", "outav": "bat_good_out_adj"})
wt = table(bw, "bowler_id", "bowler_name", {"wpa": "wpa_bowl_adj", "rae": "bowl_rae_adj", "wkt": "bowl_wae_adj"})

# =====================================================================================================
# C. ELIGIBILITY
# =====================================================================================================
def eligible(t):
    return t[(t.balls_hp >= MIN_HP_BALLS) & (t.matches_hp >= MIN_HP_MATCHES) & (t.balls_norm >= MIN_NORM_BALLS)].copy()
be, we = eligible(bt), eligible(wt)
print(f"Eligible batters: {len(be)} of {len(bt)} | eligible bowlers: {len(we)} of {len(wt)}")

# =====================================================================================================
# D. EMPIRICAL-BAYES SHRINKAGE (DerSimonian-Laird)
#    observed_i ~ N(true_i, se_i^2),  true_i ~ N(mu, tau^2)   ->   shrunk_i = mu + B_i*(observed_i - mu),
#    B_i = tau^2 / (tau^2 + se_i^2).  Noisy players (big se) are pulled hard towards the league mean mu.
# =====================================================================================================
def eb(x, se):
    x, se = np.asarray(x, float), np.asarray(se, float)
    w = 1 / se**2
    mu0 = (w * x).sum() / w.sum()
    Q = (w * (x - mu0) ** 2).sum()
    tau2 = max(0.0, (Q - (len(x) - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    for _ in range(50):                                              # refine mu with the random-effects weights
        w2 = 1 / (tau2 + se**2)
        mu = (w2 * x).sum() / w2.sum()
    B = tau2 / (tau2 + se**2)
    return mu, np.sqrt(tau2), B, mu + B * (x - mu), np.sqrt(B * se**2)

def add_shrinkage(elig, keys, label):
    """tau (the real spread between players) is estimated on the SAME eligible group we rank.
    Estimating it on a wider pool (e.g. tail-enders with 50 balls) overstates the spread and under-shrinks."""
    rows = []
    for k in keys:
        for suffix in ("hp", "delta"):
            col, sec = f"{k}_{suffix}", f"se_{k}_{suffix}"
            mu, tau, _, _, _ = eb(elig[col], elig[sec])
            B = tau**2 / (tau**2 + elig[sec] ** 2)
            elig[f"{col}_shr"] = mu + B * (elig[col] - mu)
            elig[f"{col}_B"] = B
            elig[f"{col}_psd"] = np.sqrt(B * elig[sec] ** 2)
            rows.append((label, col, round(mu, 2), round(tau, 2), round(elig[sec].median(), 2), round(B.median(), 2)))
    summ = pd.DataFrame(rows, columns=["role", "metric", "league_mu", "tau(true spread)", "median_se", "median_B(weight on own data)"])
    return elig, summ

be, s1 = add_shrinkage(be, ["wpa", "rae", "outav"], "batter")
we, s2 = add_shrinkage(we, ["wpa", "rae", "wkt"], "bowler")
print("\nShrinkage summary (B = how much of the player's own number survives; 1 = all, 0 = none):")
print(pd.concat([s1, s2]).to_string(index=False))

# =====================================================================================================
# E. PREVIEW and save
# =====================================================================================================
be.round(4).to_csv(OUT / "batter_scores.csv", index=False)
we.round(4).to_csv(OUT / "bowler_scores.csv", index=False)

def preview(e, cols, title):
    e = e.copy()
    e["rank_raw"] = e.wpa_hp.rank(ascending=False).astype(int)
    e["rank_shrunk"] = e.wpa_hp_shr.rank(ascending=False).astype(int)
    print(f"\n--- {title}: top 10 by SHRUNK pressure WPA per 100 balls (units = wins added) ---")
    print(e.sort_values("wpa_hp_shr", ascending=False).head(10)[["player", "balls_hp", "wpa_hp", "wpa_hp_shr", "rank_raw", "rank_shrunk"] + cols].round(2).to_string(index=False))
preview(be, ["rae_hp_shr", "outav_hp_shr"], "Batters")
preview(we, ["rae_hp_shr", "wkt_hp_shr"], "Bowlers")
print("\nSaved outputs/batter_scores.csv and bowler_scores.csv")