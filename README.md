# Quantifying Clutch Performance in T20 Cricket

A Data-Driven Framework for Identifying High-Pressure Performers in IPL and T20I Cricket (2022-2026)

---

## Overview

How do we identify players who perform when the pressure is highest?

This project develops a data-driven framework to quantify and rank clutch performance in T20 cricket, covering the Indian Premier League (IPL) and T20 Internationals (T20Is) from 2022 to 2026.

Rather than relying on conventional batting and bowling statistics, the framework evaluates player contributions in high-pressure match situations using:

- Ball-level match leverage
- Calibrated win probability modelling
- Context-adjusted player impact
- Empirical-Bayes shrinkage
- Performance consistency analysis
- Sensitivity analysis and match-level bootstrapping

The objective is to distinguish players who consistently deliver under pressure from those whose statistics are influenced by favourable match situations.

---

## What This Project Produces

| Output | Description |
|---|---|
| Batter Rankings | Top-performing batters in high-pressure situations |
| Bowler Rankings | Top-performing bowlers under pressure |
| Pressure Index (PI) | Quantifies match pressure using changes in win probability |
| WPA | Win Probability Added during high-pressure situations |
| Expected Performance | Runs and wickets above context-adjusted expectations |
| Pressure Validation | Validates whether identified situations represent genuine pressure |
| Sensitivity Analysis | Tests ranking robustness across different assumptions |
| Bootstrap Stability | Evaluates ranking stability across 500 match-level resamples |

### Final Report

The complete analysis is available in Markdown and HTML formats.

- outputs/final_report.html: Browser-ready report
- outputs/final_report.md: Markdown report
- outputs/top10_batters.csv: Top 10 batter rankings
- outputs/top10_bowlers.csv: Top 10 bowler rankings

Open outputs/final_report.html in your browser to explore the results.

---

## IPL 2027 Auction Filter: Rajasthan Royals (RR)

The project extends its player-ranking framework into an IPL 2027 auction availability analysis, generating a shortlist of potential Rajasthan Royals (RR) targets.

Starting with the eligible player rankings, the auction filter removes players who are unlikely to be available:

- Retained or traded IPL 2027 players
- Retired players
- Players who opted out or withdrew
- Pakistani players

The remaining batters and bowlers are re-ranked to create an auction-specific player pool.

### Auction Outputs

| File | Description |
|---|---|
| outputs/rr_auction_targets_batters.csv | Available batter shortlist |
| outputs/rr_auction_targets_bowlers.csv | Available bowler shortlist |
| outputs/rr_auction_report.html | Complete RR auction analysis |

### Run the Auction Analysis

```powershell
python src\auctionfilter.py
python src\rrauctionreport.py
```

Note: The auction filter is an availability and release-screening mechanism, not a salary or tactical prediction model. It does not estimate auction prices, purse balancing, or squad-role constraints. The RR report should be interpreted alongside the base rankings and its stated purse and squad assumptions.

---

## Project Structure

```text
T20-Clutch-Performance/
│
├── data/
│   └── Source match and ball-by-ball CSV files
│
├── src/
│   └── Data preparation, modelling, scoring,
│       analysis, and reporting scripts
│
├── outputs/
│   └── Models, intermediate tables, charts,
│       rankings, and final reports
│
├── notebook/
│   └── Exploratory and presentation notebooks
│
├── requirements.txt
└── README.md
```

---

## Installation and Requirements

### Prerequisites

- Python 3.10+
- Windows, macOS, or Linux
- Required match and ball-by-ball datasets in the data folder

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

## Running the Project

Execute the following commands sequentially from the project root directory.

### Step 1: Data Preparation

```powershell
python src\prepare_data.py
```

### Step 2: Win Probability Modelling

```powershell
python src\win_prob.py
```

### Step 3: Pressure Estimation

```powershell
python src\pressure.py
```

### Step 4: Expected Performance Modelling

```powershell
python src\expected.py
```

### Step 5: Player Metrics

```powershell
python src\player_metrics.py
```

### Step 6: Empirical-Bayes Shrinkage

```powershell
python src\shrinkage.py
```

### Step 7: Consistency Analysis

```powershell
python src\consistency.py
```

### Step 8: Player Rankings

```powershell
python src\ranking.py
```

### Step 9: Sensitivity Analysis

```powershell
python src\sensitivity.py
```

### Step 10: Reports and Visualizations

```powershell
python src\reportsandcharts.py
```

### Step 11: IPL 2027 Auction Analysis (Optional)

```powershell
python src\auctionfilter.py
python src\rrauctionreport.py
```

All generated reports, rankings, intermediate files, and visualizations are saved in the outputs folder.

---

## Methodology

The framework follows a multi-stage statistical modelling pipeline.

### 1. Match State Preparation

Construct the match state before every delivery, incorporating:

- Current score
- Wickets remaining
- Balls remaining
- Required run rate

### 2. Win Probability Modelling

Estimate the probability of winning at each ball using out-of-fold predictions, with matches separated during model training and validation.

### 3. Pressure Index Calculation

Quantify pressure through expected ball-level changes in win probability, identifying situations where individual contributions have greater match impact.

### 4. Context-Adjusted Performance

Compare actual player contributions against expected performance under equivalent match conditions and normal-situation baselines.

### 5. Empirical-Bayes Shrinkage

Reduce small-sample bias by shrinking unreliable player estimates towards league averages.

### 6. Multi-Dimensional Player Ranking

Players are evaluated across four major performance pillars:

| Pillar | What It Measures |
|---|---|
| High-Pressure WPA | Win probability contributed in high-pressure situations |
| Performance Above Expected | Actual performance relative to contextual expectations |
| Baseline Improvement | Improvement compared with the player's normal-situation performance |
| Match-Level Consistency | Repeatability of performance across matches |

### 7. Robustness and Validation

The final rankings are stress-tested through:

- Alternative weighting schemes
- Different minimum sample-size cutoffs
- Varying pressure thresholds
- 500 match-level bootstrap resamples
- Ranking stability comparisons

This helps identify players whose clutch ratings remain reliable under different modelling assumptions.

---

## Input Data

The pipeline expects the following CSV files inside the data folder:

```text
data/
├── matches.csv
├── competitions.csv
├── ball_by_ball.csv
├── teams.csv
├── playing_eleven.csv
└── competition_team_player_mapping.csv
```

After cloning the project, create a `data` folder in the project root (at the same level as `src`) if it does not already exist. Copy or extract the six CSV files directly into it; do not put them inside an extra nested folder such as `data/data`. On Windows, you can create the folder from the project root with:

```powershell
New-Item -ItemType Directory -Force .\data
```

The `data/` folder is excluded by `.gitignore`, so the datasets are not included when you push or clone the project. Obtain the files separately, and only share or commit them if their license and source allow redistribution.

Data Integrity: Abandoned matches, rain/DLS-affected matches, and no-result matches are excluded where match integrity cannot be preserved.

---

## Key Outputs

| File | Purpose |
|---|---|
| outputs/top10_batters.csv | Top 10 batter rankings |
| outputs/top10_bowlers.csv | Top 10 bowler rankings |
| outputs/batter_sensitivity.csv | Batter sensitivity and bootstrap stability |
| outputs/bowler_sensitivity.csv | Bowler sensitivity and bootstrap stability |
| outputs/final_report.md | Complete Markdown report |
| outputs/final_report.html | Browser-ready final report |
| outputs/chart*.png | Pressure, ranking, baseline, and stability charts |
| outputs/rr_auction_targets_batters.csv | Available batter auction shortlist |
| outputs/rr_auction_targets_bowlers.csv | Available bowler auction shortlist |
| outputs/rr_auction_report.html | RR IPL 2027 auction report |

---

## Understanding the Rankings

The final Pressure Rating is standardized around the eligible player population.

| Pressure Rating | Interpretation |
|---|---|
| 50 | Eligible league average |
| 65 | One standard deviation above the mean |
| 35 | One standard deviation below the mean |

The rankings aim to measure repeatable high-pressure performance relative to a player's match context and normal baseline.

Important: These ratings are descriptive statistical measures, not guarantees of future performance. They should be interpreted alongside sample size, consistency, and sensitivity results.

---

## Project Objective

The ultimate goal is to move beyond traditional cricket statistics and answer a more meaningful question:

Who actually performs when the match is on the line?

By combining ball-by-ball analytics, win probability, statistical shrinkage, and robustness testing, this project provides a systematic approach to identifying and evaluating clutch performers in T20 cricket.

---

Built using Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, and statistical modelling techniques.