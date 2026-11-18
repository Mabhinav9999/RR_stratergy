import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # project folder (superr/)
D = ROOT / "data"                              # put the 6 CSVs here
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

#loading data , cleaning it and joinig it to create a new featuer
b = pd.read_csv(D / "ball_by_ball.csv")
m = pd.read_csv(D / "matches.csv", parse_dates=["match_date"])
c = pd.read_csv(D / "competitions.csv")

m = m.merge(c[["comp_id","comp_name","match_format","season"]], on="comp_id")
m["competition"] = np.where(m.match_format=="T20I", "T20I", "IPL")   # 'T20' format == IPL here
m["rain_affected"] = m.revised_target.notna() | m.first_inn_revised_overs.notna() | m.second_inn_revised_overs.notna()
m["has_result"] = m.win_team_id.notna() & (m.win_team_id != 0) & m.result.str.contains("won", case=False, na=False)

b = b[b.innings.isin([1, 2])].copy()                       # drop super overs
b = b.merge(m[["match_id","match_date","competition","season","stage","knock_out",
               "rain_affected","has_result","win_team_id"]], on="match_id", how="inner")
b = b[b.has_result & ~b.rain_affected].copy()              # keep clean, decided, non-DLS matches
b = b.sort_values(["match_id","innings","innings_ball_num"]).reset_index(drop=True)

# ---------- STEP 2: ball-level fields ----------
et = b.extras_type.fillna("none")
b["is_wide"]   = et.eq("wide")
b["is_noball"] = et.isin(["Overstep"])
b["legal"]     = ~(b.is_wide | b.is_noball)
b["extras_tot"]= b.extras1_runs + b.extras_2_runs
b["total_runs"]= b.runs + b.extras_tot                      # what the batting team scores
b["bat_runs"]  = b.runs                                     # off the bat only
bye_like       = et.isin(["byes","legbyes"])
# Bowler is charged: bat runs + wide/no-ball penalty runs. NOT byes, leg byes, penalty runs,
# or the byes/leg-byes run off a no-ball (those sit in extras_2_runs).
b["bowler_runs"] = b.runs + np.where(et.isin(["wide","Overstep","other","none"]), b.extras1_runs, 0)
b["batter_faced"] = ~b.is_wide                               # wides don't count as balls faced
bowler_wkts = ["Caught","Bowled","LBW","Stump","Hit Wicket"]
b["bowler_wkt"] = b.is_wicket & b.dismissal_type.isin(bowler_wkts)
b["team_wkt"]   = b.is_wicket                                # any dismissal reduces wickets in hand
b["batter_out"] = b.is_wicket & ~b.dismissal_type.isin(["Retire Out"])   # a failure for the batter

# ---------- STEP 3: match state BEFORE each ball ----------
g = b.groupby(["match_id","innings"])
b["score_before"] = g.total_runs.cumsum() - b.total_runs
b["wkts_before"]  = g.team_wkt.cumsum() - b.team_wkt.astype(int)
b["legal_before"] = g.legal.cumsum() - b.legal.astype(int)
b["balls_left"]   = (120 - b.legal_before).clip(lower=0)
b["wkts_left"]    = 10 - b.wkts_before
b["crr"]          = np.where(b.legal_before>0, b.score_before*6/b.legal_before.replace(0,np.nan), 0)

inn1 = b[b.innings==1].groupby("match_id").total_runs.sum().rename("inn1_total")
b = b.merge(inn1, on="match_id", how="left")
chase = b.innings==2
b["target"]      = np.where(chase, b.inn1_total+1, np.nan)
b["runs_needed"] = np.where(chase, b.target-b.score_before, np.nan)
b["rrr"]         = np.where(chase & (b.balls_left>0), b.runs_needed*6/b.balls_left.replace(0,np.nan), np.nan)
b["rrr"]         = b.rrr.clip(upper=36)          # cap extreme values when few balls remain
b["rrr_minus_crr"]= b.rrr - b.crr
b["phase"]       = pd.cut((b.legal_before//6)+1, [0,6,15,20], labels=["PP","Mid","Death"])
b["bat_win"]     = (b.batting_team_id==b.win_team_id).astype(int)

# keep only matches where both innings exist & innings 2 ended sensibly
ok = b.groupby("match_id").innings.nunique().eq(2)
b = b[b.match_id.isin(ok[ok].index)]
b.to_pickle(OUT / "state.pkl")

# --- Step 2 checks ---
chk = b.assign(old_way=b.total_runs - np.where(bye_like | et.eq("Penalty"), b.extras_tot, 0))
print("balls where bowler_runs changed vs old formula:", (chk.old_way != chk.bowler_runs).sum())
print("bowler wickets:", int(b.bowler_wkt.sum()), "| batter outs:", int(b.batter_out.sum()), "| team wkts:", int(b.team_wkt.sum()))
print("matches kept:", b.match_id.nunique(), "| balls:", len(b))
print(b.groupby("competition").match_id.nunique())
print(b[["balls_left","wkts_left","crr","rrr","runs_needed"]].describe().round(2))
print("legal balls per innings:\n", b.groupby(["match_id","innings"]).legal.sum().describe().round(1))
print("chasing side wins when it needs <=0 runs at end:",
      b[chase].groupby("match_id").tail(1).assign(done=lambda d: d.runs_needed - d.total_runs<=0).groupby("done").bat_win.mean().round(2).to_dict())