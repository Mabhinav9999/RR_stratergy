from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# Composite weights (must sum to 1.0)
W_WPA_HP = 0.45      # Shrunk high-pressure Win Probability Added per ball
W_PERF_HP = 0.25     # Shrunk high-pressure Runs + Wicket Value Above Expected per ball
W_DELTA = 0.15       # Shrunk pressure lift vs. player's own normal baseline
W_CONSIST = 0.15     # Match-level & multi-season consistency

# Run value of a wicket in T20 (~15 runs) to combine RAE and wicket/dismissal impact
WICKET_RUN_VALUE = 15.0


def zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def build_rankings(role: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(OUT / f"{role}_scores.csv")
    cons = pd.read_csv(OUT / f"{role}_consistency.csv")

    # Inner join keeps players who met the eligibility thresholds in consistency.py
    keep_cons_cols = [
        "player_id", "q_matches", "hit_wpa", "hit_rae",
        "seasons_q", "season_share_pos", "consistency"
    ]
    df = scores.merge(cons[keep_cons_cols], on="player_id", how="inner")

    # Combined runs-equivalent performance above expected (runs + wicket impact)
    if role == "batter":
        df["perf_hp_shr"] = df["rae_hp_shr"] + WICKET_RUN_VALUE * df["outav_hp_shr"]
        df["perf_delta_shr"] = df["rae_delta_shr"] + WICKET_RUN_VALUE * df["outav_delta_shr"]
        extra_col = "outav_hp_shr"
        extra_label = "outs_saved_per_100b"
    else:
        df["perf_hp_shr"] = df["rae_hp_shr"] + WICKET_RUN_VALUE * df["wkt_hp_shr"]
        df["perf_delta_shr"] = df["rae_delta_shr"] + WICKET_RUN_VALUE * df["wkt_delta_shr"]
        extra_col = "wkt_hp_shr"
        extra_label = "wkts_ae_per_100b"

    # Standardize the 4 pillars across the eligible player pool
    df["z_wpa_hp"] = zscore(df["wpa_hp_shr"])
    df["z_perf_hp"] = zscore(df["perf_hp_shr"])
    df["z_delta"] = 0.5 * zscore(df["wpa_delta_shr"]) + 0.5 * zscore(df["perf_delta_shr"])
    df["z_consistency"] = zscore(df["consistency"])

    # Weighted composite score & intuitive 0-100 rating (mean = 50, SD = 15)
    df["composite_z"] = (
        W_WPA_HP * df["z_wpa_hp"]
        + W_PERF_HP * df["z_perf_hp"]
        + W_DELTA * df["z_delta"]
        + W_CONSIST * df["z_consistency"]
    )
    df["pressure_rating"] = (50.0 + 15.0 * df["composite_z"]).round(1)

    df = df.sort_values("composite_z", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    # Clean presentation table (per-100-ball scaling for readability)
    clean = pd.DataFrame({
        "rank": df["rank"],
        "player_id": df["player_id"],
        "player": df["player"],
        "balls_hp": df["balls_hp"].astype(int),
        "matches_hp": df["matches_hp"].astype(int),
        "ipl_share_%": (df["ipl_share_hp"] * 100).round(1),
        "wpa_hp_per_100b": (df["wpa_hp_shr"] * 100).round(3),
        "rae_hp_per_100b": (df["rae_hp_shr"] * 100).round(2),
        extra_label: (df[extra_col] * 100).round(2),
        "wpa_delta_per_100b": (df["wpa_delta_shr"] * 100).round(3),
        "consistency": df["consistency"].round(3),
        "composite_z": df["composite_z"].round(3),
        "pressure_rating": df["pressure_rating"],
    })

    return df, clean


def main():
    full_bat, clean_bat = build_rankings("batter")
    full_bowl, clean_bowl = build_rankings("bowler")

    full_bat.to_csv(OUT / "all_batter_rankings.csv", index=False)
    clean_bat.head(10).to_csv(OUT / "top10_batters.csv", index=False)

    full_bowl.to_csv(OUT / "all_bowler_rankings.csv", index=False)
    clean_bowl.head(10).to_csv(OUT / "top10_bowlers.csv", index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"\n=== TOP 10 CLUTCH BATTERS (out of {len(full_bat)} eligible) ===")
    print(clean_bat.head(10).drop(columns=["player_id"]).to_string(index=False))

    print(f"\n=== TOP 10 CLUTCH BOWLERS (out of {len(full_bowl)} eligible) ===")
    print(clean_bowl.head(10).drop(columns=["player_id"]).to_string(index=False))
    print("\nSaved top10_batters.csv, top10_bowlers.csv, all_batter_rankings.csv, all_bowler_rankings.csv")


if __name__ == "__main__":
    main()