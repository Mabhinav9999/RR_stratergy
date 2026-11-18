from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

# Final safety net for any banned/retired names not yet re-run in auctionfilter.py
EXTRA_EXCLUSIONS = {"Harry Brook", "Tim Southee"}


def load_filtered_full_stats(role: str) -> pd.DataFrame:
    filt_path = OUT / f"rr_auction_targets_{role}s.csv"
    all_path = OUT / f"all_{role}_rankings.csv"

    filt_df = pd.read_csv(filt_path)
    all_df = pd.read_csv(all_path)

    valid_names = set(filt_df["player"]) - EXTRA_EXCLUSIONS
    merged = all_df[all_df["player"].isin(valid_names)].copy()
    merged = merged.sort_values("rank", ascending=True).reset_index(drop=True)
    merged["auction_rank"] = np.arange(1, len(merged) + 1)

    merged["ipl_share_%"] = (merged["ipl_share_hp"] * 100.0).round(1)
    merged["wpa_hp_100b"] = (merged["wpa_hp_shr"] * 100.0).round(2)
    merged["wpa_norm_100b"] = (merged["wpa_norm"] * 100.0).round(2)
    merged["wpa_delta_100b"] = (merged["wpa_delta_shr"] * 100.0).round(2)
    merged["rae_hp_100b"] = (merged["rae_hp_shr"] * 100.0).round(1)
    merged["consistency"] = merged["consistency"].round(3)
    merged["pressure_rating"] = merged["pressure_rating"].round(1)

    if role == "batter" and "outav_hp_shr" in merged.columns:
        merged["outs_saved_100b"] = (merged["outav_hp_shr"] * 100.0).round(2)
    elif role == "bowler" and "wkt_hp_shr" in merged.columns:
        merged["wkts_ae_100b"] = (merged["wkt_hp_shr"] * 100.0).round(2)

    return merged


def make_rr_chart1_top15(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))

    panels = [
        (axes[0], bat_df.head(15), "Top 15 Auction-Available Clutch Batters"),
        (axes[1], bowl_df.head(15), "Top 15 Auction-Available Clutch Bowlers"),
    ]

    for ax, df, title in panels:
        d = df.sort_values("auction_rank", ascending=False).reset_index(drop=True)
        labels = [f"A#{ar} (#{r}) {p}" for ar, r, p in zip(d["auction_rank"], d["rank"], d["player"])]
        bar_colors = ["#1B4F72" if s >= 25.0 else "#D68910" for s in d["ipl_share_%"]]

        bars = ax.barh(labels, d["pressure_rating"], color=bar_colors, height=0.64)
        min_x = max(35.0, d["pressure_rating"].min() - 5.0)
        max_x = d["pressure_rating"].max() + 7.5
        ax.set_xlim(min_x, max_x)
        ax.set_xlabel("Pressure Rating (League Mean = 50, SD = 15)", fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        for bar, rating, balls, ipl_pct in zip(bars, d["pressure_rating"], d["balls_hp"], d["ipl_share_%"]):
            ax.text(
                rating + 0.35,
                bar.get_y() + bar.get_height() / 2,
                f"{rating:.1f} ({int(balls)}b | {ipl_pct:.0f}% IPL)",
                va="center",
                ha="left",
                fontsize=8.0,
                fontweight="bold",
            )

    fig.suptitle(
        "RR IPL 2027 Auction Shortlist: Navy = Proven IPL Pressure Volume (≥25% IPL) | Gold = T20I-Heavy",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fname = "rr_chart1_auction_top15.png"
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()
    return fname


def make_rr_chart2_baseline_split(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    panels = [
        (axes[0], bat_df.head(12), "Top 12 Auction Batters: High-Pressure vs. Normal WPA / 100b"),
        (axes[1], bowl_df.head(12), "Top 12 Auction Bowlers: High-Pressure vs. Normal WPA / 100b"),
    ]

    for ax, df, title in panels:
        d = df.sort_values("auction_rank", ascending=False).reset_index(drop=True)
        labels = [f"A#{ar} {p}" for ar, p in zip(d["auction_rank"], d["player"])]
        y = np.arange(len(d))

        norm_vals = d["wpa_norm_100b"]
        hp_vals = d["wpa_hp_100b"]

        for i in range(len(d)):
            ax.plot([norm_vals.iloc[i], hp_vals.iloc[i]], [y[i], y[i]], color="#95A5A6", linewidth=2, zorder=1)

        ax.scatter(norm_vals, y, color="#7F8C8D", s=65, label="Normal Situations (Raw)", zorder=2)
        ax.scatter(hp_vals, y, color="#C0392B", s=75, label="High Pressure (EB-Shrunk)", zorder=3)
        ax.axvline(0, color="black", linewidth=0.9, alpha=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Win Probability Added per 100 Balls (%)", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    fname = "rr_chart2_baseline_vs_hp.png"
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()
    return fname


def make_rr_chart3_quadrant(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    panels = [
        (axes[0], bat_df, "Auction Batters: High-Pressure WPA vs. Match Consistency"),
        (axes[1], bowl_df, "Auction Bowlers: High-Pressure WPA vs. Match Consistency"),
    ]

    for ax, df, title in panels:
        x = df["wpa_hp_100b"]
        y = df["consistency"]
        sizes = np.clip(df["balls_hp"] * 1.1, 45, 340)
        colors = ["#1B4F72" if s >= 25.0 else "#D68910" for s in df["ipl_share_%"]]

        ax.scatter(x, y, s=sizes, c=colors, alpha=0.82, edgecolors="white", linewidth=0.8)
        ax.axvline(x.median(), color="#7F8C8D", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axhline(y.median(), color="#7F8C8D", linestyle="--", linewidth=1.0, alpha=0.7)

        # Label Top-10 auction targets
        top_sub = df.head(10)
        for idx, row in top_sub.iterrows():
            dy = 0.006 if idx % 2 == 0 else -0.010
            ax.text(
                row["wpa_hp_100b"],
                row["consistency"] + dy,
                f"A#{int(row['auction_rank'])} {row['player']}",
                fontsize=7.8,
                fontweight="bold",
                ha="center",
            )

        ax.set_xlabel("High-Pressure WPA per 100 Balls (EB-Shrunk)", fontsize=9.5)
        ax.set_ylabel("Match-Level Clutch Consistency", fontsize=9.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)

    fig.suptitle(
        "RR Auction Target Quadrant (Bubble Size = High-Pressure Deliveries | Dashed Lines = Pool Medians)",
        fontsize=11.5,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fname = "rr_chart3_scout_quadrant.png"
    plt.savefig(OUT / fname, bbox_inches="tight")
    plt.close()
    return fname


def build_rr_markdown_report(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> str:
    bat_cols = [
        "auction_rank", "rank", "player", "balls_hp", "matches_hp", "ipl_share_%",
        "wpa_hp_100b", "wpa_norm_100b", "wpa_delta_100b", "rae_hp_100b",
    ]
    if "outs_saved_100b" in bat_df.columns:
        bat_cols.append("outs_saved_100b")
    bat_cols += ["consistency", "pressure_rating"]

    bowl_cols = [
        "auction_rank", "rank", "player", "balls_hp", "matches_hp", "ipl_share_%",
        "wpa_hp_100b", "wpa_norm_100b", "wpa_delta_100b", "rae_hp_100b",
    ]
    if "wkts_ae_100b" in bowl_df.columns:
        bowl_cols.append("wkts_ae_100b")
    bowl_cols += ["consistency", "pressure_rating"]

    bat_top15_md = bat_df[bat_cols].head(15).to_markdown(index=False)
    bowl_top15_md = bowl_df[bowl_cols].head(15).to_markdown(index=False)

    all_bat_md = bat_df[bat_cols].to_markdown(index=False)
    all_bowl_md = bowl_df[bowl_cols].to_markdown(index=False)

    return f"""# Rajasthan Royals (RR) IPL 2027 Auction Shortlist — Clutch Performance Analysis

## Executive Summary
Applying our Ball-Level Pressure Index ($PI \\ge 75\\text{{th}}$ percentile) and Empirical-Bayes shrinkage model to the **IPL 2027 auction pool** isolates active, eligible players who directly address Rajasthan Royals' post-retention squad gaps.

- **RR Retention & Purse Context:** **19 players retained** (14 Indian, 5 overseas: Jofra Archer, Nandre Burger, Donovan Ferreira, Dasun Shanaka, Lhuan-dre Pretorius; plus trade signings Mohammed Shami and Manimaran Siddharth). RR enters the auction with **₹27.05 crore** and **6 open squad slots** (up to **3 overseas**).
- **Strict Eligibility Filtering:** All IPL 2027 retained/traded players, retired veterans (including MS Dhoni, Andre Russell, Ajinkya Rahane, Tim Southee, Glenn Maxwell, Moeen Ali, Ravichandran Ashwin), auction opt-outs/unavailable players (Mitchell Starc, Travis Head, Faf du Plessis, Harry Brook), and Pakistani players were removed.
- **Filtered Auction Pool Size:** **{len(bat_df)} eligible batters** and **{len(bowl_df)} eligible bowlers** remain available for RR to target.

---

## 1. Top 15 Auction-Available Clutch Batters (Full Stats)
`auction_rank` (`A#`) denotes rank among active IPL 2027 auction-available players; `rank` denotes overall model rank across all 94 eligible batters.

{bat_top15_md}

---

## 2. Top 15 Auction-Available Clutch Bowlers (Full Stats)
`auction_rank` (`A#`) denotes rank among active IPL 2027 auction-available players; `rank` denotes overall model rank across all 89 eligible bowlers.

{bowl_top15_md}

![RR Top 15 Auction Targets](rr_chart1_auction_top15.png)

---

## 3. High-Pressure vs. Normal Baseline & Consistency Quadrant
The dumbbell and quadrant views below separate players who sustain positive leverage in high-pressure moments from those whose output drops sharply under scoreboard pressure.

![RR Baseline vs High Pressure](rr_chart2_baseline_vs_hp.png)

![RR Scout Quadrant](rr_chart3_scout_quadrant.png)

---

## 4. Tactical Auction Blueprint for RR (₹27.05 Cr Purse | 6 Slots)

1. **Priority 1 — Finishing All-Rounder (No. 7/8, ~50–55% of Purse):**
   Releasing Shimron Hetmyer (₹11 Cr) and trading Sam Curran leaves RR needing a lower-order hitter who bowls 2–3 overs so Ravi Bishnoi can enter cleanly as Impact Sub without exposing the tail at #8. Target spin-bowling all-rounders (**Sikandar Raza**, **George Linde**, **Kamindu Mendis**, or domestic left-arm options like **Prashant Veer** / **Mahipal Lomror**).
2. **Priority 2 — Indian Middle-Order Depth (No. 4–6, ~25–30% of Purse):**
   Behind Jaiswal, Sooryavanshi, Parag, and Jurel, adding proven Indian middle-order batters from the released pool (**Nehal Wadhera**, **Nitish Rana**, **Rahul Tripathi**, or **Deepak Hooda**) preserves overseas slots for pace and all-rounders.
3. **Priority 3 — High-Leverage Pace & Death Cover (~20% of Purse):**
   With Jofra Archer and Mohammed Shami requiring workload protection, **Trent Boult** (`A#1` bowler, `70.2` rating, `87.3%` IPL share) is the premier overseas spearhead, while **Avesh Khan** (`381` HP balls, `72.8%` IPL share) and **Deepak Chahar** provide domestic pace insurance without consuming an overseas slot.

---

## Appendix: Complete Filtered Auction Tables

### All Filtered Auction-Available Batters ({len(bat_df)})
{all_bat_md}

### All Filtered Auction-Available Bowlers ({len(bowl_df)})
{all_bowl_md}
"""


def build_html_report(md_text: str) -> str:
    try:
        import markdown
        body = markdown.markdown(md_text, extensions=["tables"])
    except ImportError:
        lines = md_text.splitlines()
        html_lines = []
        in_table = False
        for line in lines:
            if line.startswith("### "):
                html_lines.append(f"<h3>{line[4:]}</h3>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("# "):
                html_lines.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("![") and "](" in line:
                alt = line.split("![")[1].split("](")[0]
                src = line.split("](")[1].rstrip(")")
                html_lines.append(f'<div class="chart"><img src="{src}" alt="{alt}"></div>')
            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(set(c) <= set("-:") for c in cells):
                    continue
                if not in_table:
                    html_lines.append("<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr></thead><tbody>")
                    in_table = True
                else:
                    html_lines.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            else:
                if in_table:
                    html_lines.append("</tbody></table>")
                    in_table = False
                if line.startswith("- "):
                    html_lines.append(f"<ul><li>{line[2:]}</li></ul>")
                elif line.strip() == "---":
                    html_lines.append("<hr>")
                elif line.strip():
                    html_lines.append(f"<p>{line}</p>")
        if in_table:
            html_lines.append("</tbody></table>")
        body = "\n".join(html_lines)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RR IPL 2027 Auction Clutch Targets Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
         max-width: 1120px; margin: 28px auto; padding: 0 22px; color: #1f2937; line-height: 1.5; }}
  h1 {{ color: #1b263b; border-bottom: 2px solid #e91e63; padding-bottom: 8px; }}
  h2 {{ color: #1b4f72; margin-top: 26px; border-bottom: 1px solid #d5dbdb; padding-bottom: 4px; }}
  h3 {{ color: #2c3e50; margin-top: 18px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 12px; }}
  th, td {{ border: 1px solid #d0d7de; padding: 5px 7px; text-align: right; }}
  th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
  th {{ background-color: #f0f4f8; font-weight: 600; }}
  tr:nth-child(even) {{ background-color: #fafbfc; }}
  .chart {{ text-align: center; margin: 18px 0; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #e5e7eb; border-radius: 4px; }}
  @media print {{ body {{ max-width: 100%; margin: 0; }} h2 {{ page-break-after: avoid; }} }}
</style>
</head>
<body>
{body}
</body>
</html>"""


def main():
    bat_df = load_filtered_full_stats("batter")
    bowl_df = load_filtered_full_stats("bowler")

    c1 = make_rr_chart1_top15(bat_df, bowl_df)
    c2 = make_rr_chart2_baseline_split(bat_df, bowl_df)
    c3 = make_rr_chart3_quadrant(bat_df, bowl_df)

    md_report = build_rr_markdown_report(bat_df, bowl_df)
    (OUT / "rr_auction_report.md").write_text(md_report, encoding="utf-8")

    html_report = build_html_report(md_report)
    (OUT / "rr_auction_report.html").write_text(html_report, encoding="utf-8")

    print("Generated RR auction charts:")
    for c in [c1, c2, c3]:
        print(f"  - outputs/{c}")
    print("Generated RR auction reports:")
    print("  - outputs/rr_auction_report.md")
    print("  - outputs/rr_auction_report.html (open in Chrome/Edge and press Ctrl+P -> Save as PDF)")


if __name__ == "__main__":
    main()