"""
STEP 8 - Player metrics: pressure vs normal, per batter and per bowler.
Reads  outputs/residuals.pkl
Writes outputs/batter_metrics.csv, bowler_metrics.csv, batter_match_hp.pkl, bowler_match_hp.pkl
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
r = pd.read_pickle(OUT / "residuals.pkl")

def build_table(df, pid, name, cols):
    """df: one row per ball for this role. cols: dict role-specific column names."""
    wpa, rae, out_or_wkt, runs = cols["wpa"], cols["rae"], cols["wkt_ae"], cols["runs"]
    latest = df.sort_values("match_date").groupby(pid)[name].last().rename("player")
    parts = []
    for flag, tag in ((True, "hp"), (False, "norm")):
        d = df[df.high_pressure == flag]
        g = d.groupby(pid)
        t = pd.DataFrame({
            f"balls_{tag}":   g.size(),
            f"matches_{tag}": g.match_id.nunique(),
            f"wpa100_{tag}":  100 * g[wpa].mean(),                       # win-probability added per 100 balls
            f"se_wpa100_{tag}": 100 * g[wpa].std() / np.sqrt(g.size()),   # uncertainty of that estimate
            f"rae100_{tag}":  100 * g[rae].mean(),                       # runs above (batter) / saved vs (bowler) expected per 100
            f"wkt_ae100_{tag}": 100 * g[out_or_wkt].mean(),              # dismissal / wicket rate vs expected per 100
            f"runs_per_ball_{tag}": g[runs].mean(),
        })
        parts.append(t)
    t = pd.concat(parts, axis=1).join(latest)
    t["ipl_share_hp"] = df[df.high_pressure].groupby(pid).competition.apply(lambda s: (s == "IPL").mean())
    # DELTA: how a player's performance (vs peers at the same pressure level) changes when pressure rises
    t["delta_rae100"] = t.rae100_hp - t.rae100_norm
    t["delta_wkt_ae100"] = t.wkt_ae100_hp - t.wkt_ae100_norm
    t[["balls_hp", "balls_norm", "matches_hp", "matches_norm"]] = t[["balls_hp", "balls_norm", "matches_hp", "matches_norm"]].fillna(0).astype(int)
    return t.reset_index().rename(columns={pid: "player_id"})

# ---------------- BATTERS ----------------
bat = r[r.batter_faced & ~r.nonstriker_ro].copy()
bat["bat_good_out"] = -bat.bat_out_ae          # flip sign so that, for every metric, higher = better
bt = build_table(bat, "striker_id", "striker_name",
                 dict(wpa="wpa_bat", rae="bat_rae", wkt_ae="bat_good_out", runs="bat_runs"))
bt = bt.rename(columns={"wkt_ae100_hp": "out_avoided100_hp", "wkt_ae100_norm": "out_avoided100_norm",
                        "delta_wkt_ae100": "delta_out_avoided100"})
hp = bat[bat.high_pressure]
bt["sr_hp"] = bt.player_id.map(100 * hp.groupby("striker_id").bat_runs.mean())
bt["dismiss_per_100_hp"] = bt.player_id.map(100 * hp.groupby("striker_id").striker_out.mean())

# ---------------- BOWLERS ----------------
bw = r.copy()
bw["wkt_f"] = bw.bowler_wkt.astype(float)
wt = build_table(bw, "bowler_id", "bowler_name",
                 dict(wpa="wpa_bowl", rae="bowl_rae", wkt_ae="bowl_wae", runs="bowler_runs"))
wt = wt.rename(columns={"runs_per_ball_hp": "conceded_per_ball_hp", "runs_per_ball_norm": "conceded_per_ball_norm"})
hpb = bw[bw.high_pressure]
wt["economy_hp"] = wt.player_id.map(6 * hpb.groupby("bowler_id").bowler_runs.mean())
wt["wkts_per_100_hp"] = wt.player_id.map(100 * hpb.groupby("bowler_id").wkt_f.mean())

# ---------------- PER-MATCH HIGH-PRESSURE TABLES (used for consistency in Step 10) ----------------
bm = bat[bat.high_pressure].groupby(["striker_id", "match_id"]).agg(
    n=("wpa_bat", "size"), wpa=("wpa_bat", "sum"), rae=("bat_rae", "sum"), date=("match_date", "first")).reset_index()
wm = bw[bw.high_pressure].groupby(["bowler_id", "match_id"]).agg(
    n=("wpa_bowl", "size"), wpa=("wpa_bowl", "sum"), rae=("bowl_rae", "sum"), date=("match_date", "first")).reset_index()
bm.to_pickle(OUT / "batter_match_hp.pkl"); wm.to_pickle(OUT / "bowler_match_hp.pkl")

bt.round(4).to_csv(OUT / "batter_metrics.csv", index=False)
wt.round(4).to_csv(OUT / "bowler_metrics.csv", index=False)

# ---------------- sanity checks / preview ----------------
print("Batters:", len(bt), "| with >=100 pressure balls:", (bt.balls_hp >= 100).sum(), "| >=150:", (bt.balls_hp >= 150).sum())
print("Bowlers:", len(wt), "| with >=100 pressure balls:", (wt.balls_hp >= 100).sum(), "| >=150:", (wt.balls_hp >= 150).sum())

show = ["player", "balls_hp", "wpa100_hp", "se_wpa100_hp", "rae100_hp", "rae100_norm", "delta_rae100"]
print("\n--- RAW top 8 batters by pressure WPA, ANY sample size (why shrinkage is needed) ---")
print(bt[bt.balls_hp >= 20].sort_values("wpa100_hp", ascending=False).head(8)[show].round(2).to_string(index=False))
print("\n--- Same, but only players with >=100 pressure balls ---")
el = bt[bt.balls_hp >= 100]
print(el.sort_values("wpa100_hp", ascending=False).head(8)[show].round(2).to_string(index=False))
print("\n--- Bowlers with >=100 pressure balls, top 8 by pressure WPA ---")
elw = wt[wt.balls_hp >= 100]
print(elw.sort_values("wpa100_hp", ascending=False).head(8)[["player", "balls_hp", "wpa100_hp", "se_wpa100_hp", "rae100_hp", "rae100_norm", "economy_hp", "wkts_per_100_hp"]].round(2).to_string(index=False))

print("\nDoes a player's normal-pressure skill predict his pressure skill? (corr of rae100, eligible players)")
print("  batters:", round(el.rae100_hp.corr(el.rae100_norm), 2), "| bowlers:", round(elw.rae100_hp.corr(elw.rae100_norm), 2))
print("Typical noise (median s.e. of pressure WPA/100): batters", round(el.se_wpa100_hp.median(), 2),
      "| spread (std) of pressure WPA/100 across batters", round(el.wpa100_hp.std(), 2))
print("\nSaved batter_metrics.csv, bowler_metrics.csv, batter_match_hp.pkl, bowler_match_hp.pkl")