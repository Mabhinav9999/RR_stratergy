"""
STEP 10 - Consistency and reliability.
Reads  outputs/bat_adj.pkl, bowl_adj.pkl, batter_scores.csv, bowler_scores.csv
Writes outputs/batter_consistency.csv, bowler_consistency.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MIN_HP_IN_MATCH = 4        # a match "counts" for a player only if he had at least this many pressure balls in it
PRIOR_M = 10               # strength (in matches) of the pull towards the league hit rate
MIN_HP_IN_SEASON = 20      # a calendar year "counts" if he had at least this many pressure balls
N_SPLITS = 300             # repeated random split-half tests
rng = np.random.default_rng(42)

ROLES = {
    "batter": dict(df=pd.read_pickle(OUT / "bat_adj.pkl"), pid="striker_id", scores=pd.read_csv(OUT / "batter_scores.csv"),
                   m=dict(wpa="wpa_bat_adj", rae="bat_rae_adj", wk="bat_good_out_adj")),
    "bowler": dict(df=pd.read_pickle(OUT / "bowl_adj.pkl"), pid="bowler_id", scores=pd.read_csv(OUT / "bowler_scores.csv"),
                   m=dict(wpa="wpa_bowl_adj", rae="bowl_rae_adj", wk="bowl_wae_adj")),
}

def analyse(role, cfg):
    pid, m = cfg["pid"], cfg["m"]
    elig = cfg["scores"][["player_id", "player", "balls_hp"]]
    d = cfg["df"][cfg["df"].high_pressure & cfg["df"][pid].isin(elig.player_id)].copy()
    d["year"] = pd.to_datetime(d.match_date).dt.year

    # ---------- 1. match-by-match hit rate ----------
    pm = d.groupby([pid, "match_id"]).agg(n=("match_id", "size"), wpa=(m["wpa"], "sum"), rae=(m["rae"], "sum"),
                                          wk=(m["wk"], "sum"), year=("year", "first")).reset_index()
    q = pm[pm.n >= MIN_HP_IN_MATCH].copy()
    q["pos_rae"] = q.rae > 0
    q["pos_wpa"] = q.wpa > 0
    h = q.groupby(pid).agg(q_matches=("n", "size"), pos_rae=("pos_rae", "sum"), pos_wpa=("pos_wpa", "sum"))
    for k in ("rae", "wpa"):
        p0 = h[f"pos_{k}"].sum() / h.q_matches.sum()                      # league-wide hit rate
        h[f"hit_{k}_raw"] = h[f"pos_{k}"] / h.q_matches
        h[f"hit_{k}"] = (h[f"pos_{k}"] + PRIOR_M * p0) / (h.q_matches + PRIOR_M)   # shrunk towards league rate
        print(f"  {role}: league share of pressure-matches with positive {k.upper()} = {p0:.3f}")
    h["consistency"] = (h.hit_rae + h.hit_wpa) / 2

    # ---------- 2. season-by-season stability ----------
    ys = d.groupby([pid, "year"]).agg(n=("match_id", "size"), rae=(m["rae"], "mean")).reset_index()
    ys = ys[ys.n >= MIN_HP_IN_SEASON]
    ys["pos"] = ys.rae > 0
    s = ys.groupby(pid).agg(seasons_q=("pos", "size"), seasons_pos=("pos", "sum"))
    s["season_share_pos"] = s.seasons_pos / s.seasons_q

    # ---------- 3. split-half reliability (random halves of MATCHES, repeated) ----------
    pmm = pm.copy()
    match_ids = pmm.match_id.unique()
    rel = {}
    for k in ("wpa", "rae", "wk"):
        rs = []
        for _ in range(N_SPLITS):
            half = pd.Series(rng.integers(0, 2, len(match_ids)), index=match_ids)
            pmm["half"] = pmm.match_id.map(half)
            g = pmm.groupby([pid, "half"]).agg(n=("n", "sum"), v=(k, "sum")).unstack("half")
            g = g.dropna()
            g = g[(g[("n", 0)] >= 20) & (g[("n", 1)] >= 20)]
            a, b_ = g[("v", 0)] / g[("n", 0)], g[("v", 1)] / g[("n", 1)]
            rs.append(np.corrcoef(a, b_)[0, 1])
        r = np.nanmean(rs)
        rel[k] = (r, 2 * r / (1 + r))                                       # Spearman-Brown: reliability of the FULL sample
    out = elig.set_index("player_id").join(h).join(s)
    return out.reset_index(), rel

for role, cfg in ROLES.items():
    print(f"\n=== {role.upper()}S ===")
    tbl, rel = analyse(role, cfg)
    print(f"  Split-half reliability of the pressure measure (random match halves, {N_SPLITS} repeats):")
    for k, label in (("wpa", "win-prob added"), ("rae", "runs above/saved vs expected"), ("wk", "dismissal / wicket vs expected")):
        r, sb = rel[k]
        print(f"     {label:<34} half-vs-half r = {r:5.2f}   -> full-sample reliability = {sb:4.2f}")
    tbl.round(3).to_csv(OUT / f"{role}_consistency.csv", index=False)
    show = ["player", "balls_hp", "q_matches", "pos_rae", "hit_rae", "hit_wpa", "consistency", "seasons_pos", "seasons_q"]
    print(f"\n  Most consistent {role}s (shrunk hit rate, both measures averaged):")
    print(tbl.sort_values("consistency", ascending=False).head(8)[show].round(2).to_string(index=False))
    sc = cfg["scores"].merge(tbl[["player_id", "consistency"]], on="player_id")
    print(f"  Corr(consistency, shrunk pressure RAE) = {sc.consistency.corr(sc.rae_hp_shr):.2f} | "
          f"Corr(consistency, shrunk pressure WPA) = {sc.consistency.corr(sc.wpa_hp_shr):.2f}")
print("\nSaved batter_consistency.csv and bowler_consistency.csv")