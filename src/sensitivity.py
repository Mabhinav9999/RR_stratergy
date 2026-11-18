from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WICKET_RUN_VALUE = 15.0
N_BOOT = 500
SEED = 42


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def eb_shrink(vals: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Empirical-Bayes shrinkage toward the sample-weighted grand mean."""
    mu = np.average(vals, weights=n)
    s2_within = np.average((vals - mu) ** 2, weights=n)
    s2_between = max(np.var(vals, ddof=1) - np.mean(s2_within / np.maximum(n, 1)), 1e-9)
    k = float(np.clip(s2_within / s2_between, 25.0, 400.0))
    return (n * vals + k * mu) / (n + k)


def weight_and_cutoff_sensitivity(role: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"all_{role}_rankings.csv")
    base_top10 = df.head(10)[["rank", "player_id", "player", "balls_hp"]].copy()
    base_top10 = base_top10.rename(columns={"rank": "base_rank"})

    weight_schemes = {
        "rk_wpa_heavy": (0.60, 0.20, 0.10, 0.10),
        "rk_skill_heavy": (0.25, 0.45, 0.15, 0.15),
        "rk_delta_heavy": (0.30, 0.20, 0.35, 0.15),
        "rk_consist_heavy": (0.30, 0.20, 0.15, 0.35),
    }

    for col_name, (w1, w2, w3, w4) in weight_schemes.items():
        score = (
            w1 * df["z_wpa_hp"]
            + w2 * df["z_perf_hp"]
            + w3 * df["z_delta"]
            + w4 * df["z_consistency"]
        )
        ranks = score.rank(ascending=False, method="min").astype(int)
        rank_map = dict(zip(df["player_id"], ranks))
        base_top10[col_name] = base_top10["player_id"].map(rank_map)

    # Sample cutoff sensitivity (1.25x and 1.5x current minimum balls_hp)
    min_b = int(df["balls_hp"].min())
    for mult, label in [(1.25, "rk_cut_1.25x"), (1.50, "rk_cut_1.5x")]:
        cut = int(np.ceil(min_b * mult))
        sub = df[df["balls_hp"] >= cut].copy()
        sub["sub_rk"] = sub["composite_z"].rank(ascending=False, method="min").astype(int)
        rank_map = dict(zip(sub["player_id"], sub["sub_rk"]))
        base_top10[label] = base_top10["player_id"].map(rank_map).fillna(-1).astype(int)

    return base_top10


def threshold_sensitivity(res: pd.DataFrame, role: str, min_balls_base: int) -> pd.DataFrame:
    if role == "batter":
        d = res[res["batter_faced"].astype(bool)].copy()
        d["pid"] = d["striker_id"]
        d["wpa"] = d["wpa_bat"]
        d["perf"] = d["bat_rae"] + WICKET_RUN_VALUE * d["bat_out_ae"]
    else:
        d = res[res["legal"].astype(bool)].copy()
        d["pid"] = d["bowler_id"]
        d["wpa"] = d["wpa_bowl"]
        d["perf"] = d["bowl_rae"] + WICKET_RUN_VALUE * d["bowl_wae"]

    results = {}
    for pct_label, q, scale in [("rk_pi_top20", 0.80, 0.80), ("rk_pi_top25", 0.75, 1.00), ("rk_pi_top33", 0.67, 1.32)]:
        thresh = d["pi"].quantile(q)
        d["hp"] = d["pi"] >= thresh
        min_b = max(50, int(round(min_balls_base * scale)))

        hp_grp = d[d["hp"]].groupby("pid").agg(
            n_hp=("wpa", "count"),
            wpa_hp=("wpa", "mean"),
            perf_hp=("perf", "mean"),
        )
        norm_grp = d[~d["hp"]].groupby("pid").agg(
            n_norm=("wpa", "count"),
            wpa_norm=("wpa", "mean"),
            perf_norm=("perf", "mean"),
        )
        m_grp = (
            d[d["hp"]]
            .groupby(["pid", "match_id"], as_index=False)
            .agg(m_wpa=("wpa", "sum"), m_perf=("perf", "sum"))
        )
        m_grp["pos"] = ((m_grp["m_wpa"] > 0) & (m_grp["m_perf"] > 0)).astype(float)
        cons_grp = m_grp.groupby("pid").agg(cons=("pos", "mean"))

        agg = hp_grp.join(norm_grp, how="inner").join(cons_grp, how="inner")
        agg = agg[(agg["n_hp"] >= min_b) & (agg["n_norm"] >= min_b)].copy()

        agg["wpa_hp_shr"] = eb_shrink(agg["wpa_hp"].values, agg["n_hp"].values)
        agg["perf_hp_shr"] = eb_shrink(agg["perf_hp"].values, agg["n_hp"].values)
        agg["wpa_delta_shr"] = eb_shrink((agg["wpa_hp"] - agg["wpa_norm"]).values, agg["n_hp"].values)
        agg["perf_delta_shr"] = eb_shrink((agg["perf_hp"] - agg["perf_norm"]).values, agg["n_hp"].values)

        comp = (
            0.45 * zscore(agg["wpa_hp_shr"])
            + 0.25 * zscore(agg["perf_hp_shr"])
            + 0.15 * (0.5 * zscore(agg["wpa_delta_shr"]) + 0.5 * zscore(agg["perf_delta_shr"]))
            + 0.15 * zscore(agg["cons"])
        )
        ranks = comp.rank(ascending=False, method="min").astype(int)
        results[pct_label] = ranks.to_dict()

    return pd.DataFrame(results)


def bootstrap_stability(role: str, min_balls: int) -> pd.DataFrame:
    if role == "batter":
        adj = pd.read_pickle(OUT / "bat_adj.pkl")
        adj = adj.rename(columns={"striker_id": "pid", "wpa_bat_adj": "wpa", "bat_rae_adj": "rae", "bat_good_out_adj": "wkt"})
    else:
        adj = pd.read_pickle(OUT / "bowl_adj.pkl")
        adj = adj.rename(columns={"bowler_id": "pid", "wpa_bowl_adj": "wpa", "bowl_rae_adj": "rae", "bowl_wae_adj": "wkt"})

    adj["perf"] = adj["rae"] + WICKET_RUN_VALUE * adj["wkt"]
    adj["hp"] = adj["high_pressure"].astype(bool)

    # Pre-aggregate to (pid, match_id, hp) so 500 bootstraps run in ~1 second
    pm = adj.groupby(["pid", "match_id", "hp"], as_index=False).agg(
        n=("wpa", "count"),
        sum_wpa=("wpa", "sum"),
        sum_perf=("perf", "sum"),
    )
    pm["pos_match"] = ((pm["sum_wpa"] > 0) & (pm["sum_perf"] > 0)).astype(float)

    matches = adj["match_id"].unique()
    n_matches = len(matches)
    rng = np.random.default_rng(SEED)

    rank_records: dict[int, list[int]] = {}

    for _ in range(N_BOOT):
        sampled = rng.choice(matches, size=n_matches, replace=True)
        m_ids, m_counts = np.unique(sampled, return_counts=True)
        mult_df = pd.DataFrame({"match_id": m_ids, "mult": m_counts})

        b = pm.merge(mult_df, on="match_id", how="inner")
        b["wn"] = b["n"] * b["mult"]
        b["w_wpa"] = b["sum_wpa"] * b["mult"]
        b["w_perf"] = b["sum_perf"] * b["mult"]
        b["w_pos"] = b["pos_match"] * b["mult"]

        hp = b[b["hp"]].groupby("pid").agg(
            n_hp=("wn", "sum"),
            wpa_hp=("w_wpa", "sum"),
            perf_hp=("w_perf", "sum"),
            pos_m=("w_pos", "sum"),
            tot_m=("mult", "sum"),
        )
        hp = hp[hp["n_hp"] >= min_balls].copy()
        hp["wpa_hp"] /= hp["n_hp"]
        hp["perf_hp"] /= hp["n_hp"]
        hp["cons"] = (hp["pos_m"] + 1.0) / (hp["tot_m"] + 2.0)

        nm = b[~b["hp"]].groupby("pid").agg(
            n_nm=("wn", "sum"),
            wpa_nm=("w_wpa", "sum"),
            perf_nm=("w_perf", "sum"),
        )
        nm["wpa_nm"] /= np.maximum(nm["n_nm"], 1)
        nm["perf_nm"] /= np.maximum(nm["n_nm"], 1)

        g = hp.join(nm, how="inner")
        g["wpa_hp_shr"] = eb_shrink(g["wpa_hp"].values, g["n_hp"].values)
        g["perf_hp_shr"] = eb_shrink(g["perf_hp"].values, g["n_hp"].values)
        g["wpa_d_shr"] = eb_shrink((g["wpa_hp"] - g["wpa_nm"]).values, g["n_hp"].values)
        g["perf_d_shr"] = eb_shrink((g["perf_hp"] - g["perf_nm"]).values, g["n_hp"].values)

        comp = (
            0.45 * zscore(g["wpa_hp_shr"])
            + 0.25 * zscore(g["perf_hp_shr"])
            + 0.15 * (0.5 * zscore(g["wpa_d_shr"]) + 0.5 * zscore(g["perf_d_shr"]))
            + 0.15 * zscore(g["cons"])
        )
        ranks = comp.rank(ascending=False, method="min").astype(int)
        for pid, rk in ranks.items():
            rank_records.setdefault(pid, []).append(int(rk))

    rows = []
    for pid, rks in rank_records.items():
        arr = np.array(rks)
        rows.append({
            "player_id": pid,
            "boot_med_rk": int(np.median(arr)),
            "boot_ci90": f"[{int(np.percentile(arr, 5))}-{int(np.percentile(arr, 95))}]",
            "p_top10_%": round(float(np.mean(arr <= 10) * 100), 1),
        })
    return pd.DataFrame(rows)


def run_role(res: pd.DataFrame, role: str) -> pd.DataFrame:
    all_rk = pd.read_csv(OUT / f"all_{role}_rankings.csv")
    min_b = int(all_rk["balls_hp"].min())

    wc = weight_and_cutoff_sensitivity(role)
    th = threshold_sensitivity(res, role, min_b)
    bt = bootstrap_stability(role, min_b)

    out = wc.merge(th, left_on="player_id", right_index=True, how="left")
    out = out.merge(bt, on="player_id", how="left")

    # Robustness flag: stays <= 12 across all weight schemes & >= 60% bootstrap Top-10 probability
    w_cols = ["rk_wpa_heavy", "rk_skill_heavy", "rk_delta_heavy", "rk_consist_heavy"]
    max_w_rk = out[w_cols].max(axis=1)
    out["verdict"] = np.where((max_w_rk <= 12) & (out["p_top10_%"] >= 60.0), "ROBUST", "BORDERLINE")
    return out


def main():
    print("Loading residuals.pkl for threshold sensitivity...")
    res = pd.read_pickle(OUT / "residuals.pkl")

    bat_sens = run_role(res, "batter")
    bowl_sens = run_role(res, "bowler")

    bat_sens.to_csv(OUT / "batter_sensitivity.csv", index=False)
    bowl_sens.to_csv(OUT / "bowler_sensitivity.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)

    print("\n=== BATTER TOP-10 SENSITIVITY & BOOTSTRAP STABILITY (-1 = below raised cutoff) ===")
    print(bat_sens.drop(columns=["player_id"]).to_string(index=False))

    print("\n=== BOWLER TOP-10 SENSITIVITY & BOOTSTRAP STABILITY (-1 = below raised cutoff) ===")
    print(bowl_sens.drop(columns=["player_id"]).to_string(index=False))
    print("\nSaved outputs/batter_sensitivity.csv and outputs/bowler_sensitivity.csv")


if __name__ == "__main__":
    main()