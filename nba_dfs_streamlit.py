import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# ===============================
# NBA DFS SIMULATOR (FanDuel-style)
# Roster: 2 PG, 2 SG, 2 SF, 2 PF, 1 C | Cap: $60,000 | Max 4 per team
# Works with Rotowire export like rw-nba-player-pool.xlsx
# ===============================

FD_CAP = 60000
MAX_FROM_TEAM = 4
ROSTER_TEMPLATE = ["PG","PG","SG","SG","SF","SF","PF","PF","C"]

st.title("NBA DFS Slate Simulator – Streamlit")
st.caption("FanDuel-style: 2PG,2SG,2SF,2PF,1C · $60k cap · Uses Rotowire player pool")

file = st.file_uploader("Upload Rotowire NBA player pool (xlsx or csv)", type=["xlsx","csv"])
if not file:
    st.stop()

# -------- Load --------
if file.name.endswith("xlsx"):
    try:
        df_raw = pd.read_excel(file)
    except Exception as e:
        st.error("Reading .xlsx requires openpyxl. Run: pip install openpyxl\n"+str(e))
        st.stop()
else:
    df_raw = pd.read_csv(file)

# Expected columns seen in RW sample: PLAYER, POS, TEAM, OPP, SAL, FPTS, MIN, etc.
rename_map = {
    "PLAYER":"name",
    "TEAM":"team",
    "OPP":"opp",
    "POS":"pos",
    "SAL":"salary",
    "FPTS":"proj_mean",
    "MIN":"min"
}

df = df_raw.rename(columns={k:v for k,v in rename_map.items() if k in df_raw.columns}).copy()

# Clean / derive
# POS may be like "PF/C"; keep list of eligible positions
pos_series = df.get("pos", df_raw.get("POS"))
df["pos_list"] = pos_series.astype(str).str.replace(" ", "").str.split("/")
# Normalize main pos for display (first eligible)
df["pos"] = df["pos_list"].apply(lambda x: x[0] if isinstance(x,list) and len(x)>0 else "UTIL")

# Numerics
for col in ["salary","proj_mean","min"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["salary"].fillna(0, inplace=True)
df["proj_mean"].fillna(0.0, inplace=True)
if "min" not in df.columns:
    df["min"] = 30  # fallback

# Projected SD – simple slider (minutes-aware)
sd_base = st.slider("Projection SD factor (as % of mean)", 0.30, 0.80, 0.50)
# Bump variance for low minutes, reduce for high minutes
min_scale = np.clip(40 / np.maximum(df["min"], 10), 0.6, 1.4)
df["proj_sd"] = (df["proj_mean"] * sd_base * min_scale).clip(lower=1.0)

# IDs
if "player_id" not in df.columns:
    df["player_id"] = df.index.astype(str)

# Filter crazy rows
df = df[(df["salary"] > 0) & (df["proj_mean"] > 0)].reset_index(drop=True)

# --- Blowout logic using spread (SPRD) ---
# Compute blowout factor based on Vegas spread
# Negative spread = team favored; larger magnitude = higher blowout risk

def blowout_factor(spread):
    if spread <= -15: return 0.30  # huge favorite
    if spread <= -11: return 0.20  # strong favorite
    if spread <= -8:  return 0.10  # moderate
    return 0.0

if "SPRD" in df.columns:
    df["blowout_f"] = df["SPRD"].apply(blowout_factor)
    # Adjust projections: stars lose small floor, variance rises for volatility
    df["proj_mean"] = df["proj_mean"] * (1 - df["blowout_f"] * 0.5)
    df["proj_sd"]   = df["proj_sd"]   * (1 + df["blowout_f"] * 1.5)
else:
    df["blowout_f"] = 0.0

st.subheader("Preview (first 25)")
st.dataframe(df[["name","team","opp","pos","pos_list","salary","proj_mean","min"]].head(25))

# -------- Correlation Model --------
# Light-touch correlations: teammates +0.10, same game +0.05, position-same slight +0.02
CORR_TEAM = 0.10
CORR_GAME = 0.05
CORR_POS  = 0.02
CORR_CAP  = 0.50

def build_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    m = np.zeros((n,n), dtype=float)
    team = df["team"].to_numpy()
    opp  = df["opp"].to_numpy()
    # infer game ids by sorted tuple
    game = np.array(["-".join(sorted([team[i], opp[i]])) for i in range(n)])
    pos  = df["pos"].to_numpy()
    for i in range(n):
        m[i,i] = 1.0
        for j in range(i+1, n):
            val = 0.0
            if team[i] == team[j]:
                val += CORR_TEAM
            if str(game[i]) == str(game[j]):
                val += CORR_GAME
            if pos[i] == pos[j]:
                val += CORR_POS
            val = max(min(val, CORR_CAP), -CORR_CAP)
            m[i,j] = m[j,i] = val
    return pd.DataFrame(m)

def build_cov_matrix(df: pd.DataFrame, corr: pd.DataFrame) -> np.ndarray:
    sd = df["proj_sd"].to_numpy()
    D = np.diag(sd)
    cov = D @ corr.values @ D
    eig = np.linalg.eigvalsh(cov)
    mine = eig.min()
    if mine < 1e-8:
        cov += np.eye(cov.shape[0]) * (1e-8 - mine + 1e-8)
    return cov

# -------- Roster Validation with Multi-Pos --------
NEEDED = {"PG":2,"SG":2,"SF":2,"PF":2,"C":1}

def fits_roster(counts: dict) -> bool:
    return all(counts.get(p,0) <= need for p,need in NEEDED.items()) and sum(counts.values()) <= 9

def valid_lineup(players: list, cap: int, salary_min: int) -> bool:
    if len(players) != 9:
        return False
    salary = sum(p["salary"] for p in players)
    if salary > cap or salary < salary_min:
        return False
    # assign positions greedily to satisfy requirements
    counts = {k:0 for k in NEEDED}
    remaining = NEEDED.copy()
    # First assign by exact first pos when possible
    for p in players:
        placed = False
        for pos in p["pos_list"]:
            if pos in remaining and remaining[pos] > 0:
                remaining[pos] -= 1
                placed = True
                break
        if not placed:
            return False
    # max from team
    teams = {}
    for p in players:
        teams[p["team"]] = teams.get(p["team"],0)+1
    if any(v > MAX_FROM_TEAM for v in teams.values()):
        return False
    # unique
    ids = [p["player_id"] for p in players]
    if len(ids) != len(set(ids)):
        return False
    return all(v==0 for v in remaining.values())

# -------- Candidate Pool Builder --------

def build_candidate_pool(df: pd.DataFrame, pool_size: int, cap: int, salary_min: int, game_stack_min: int = 0) -> list:
    """Generate a pool of candidate lineups with flexible positions.
    Faster + smarter: prioritize needed positions, check salary feasibility, and allow many more tries on small slates.
    """
    df = df.copy()
    # game key for optional bias
    df["game_key"] = df.apply(lambda r: "-".join(sorted([r["team"], r["opp"]])), axis=1)

    # basic supply diagnostics (silent, but you can uncomment to debug)
    pos_supply = {p: int(df[df["pos_list"].apply(lambda x: p in x if isinstance(x, list) else False)].shape[0]) for p in NEEDED}
    if any(pos_supply[p] < NEEDED[p] for p in NEEDED):
        st.warning(f"Low supply for positions: {pos_supply}. Consider lowering min salary or using CSV with full pool.")

    by_game = {k: v.to_dict("records") for k, v in df.groupby("game_key")}
    records = df.to_dict("records")

    # Precompute sorted salaries for quick feasibility bounds
    all_salaries = sorted([int(r.get("salary", 0)) for r in records])

    def salary_bounds_ok(cur_salary: int, remaining_slots: int) -> bool:
        # Lower bound: even the cheapest remaining must fit under cap
        min_possible = cur_salary + sum(all_salaries[:min(remaining_slots, len(all_salaries))])
        if min_possible > cap:
            return False
        # Upper bound: best case we can still reach salary_min
        max_possible = cur_salary + sum(sorted(all_salaries, reverse=True)[:min(remaining_slots, len(all_salaries))])
        if max_possible < salary_min:
            return False
        return True

    def greedy_assign_ok(players: list) -> bool:
        # quick feasibility using greedy position fill
        rem = NEEDED.copy()
        for q in players:
            placed = False
            for pos in q["pos_list"]:
                if pos in rem and rem[pos] > 0:
                    rem[pos] -= 1
                    placed = True
                    break
            if not placed:
                return False
        return True

    cands = []
    tries = 0
    max_tries = pool_size * pool_multiplier

    # NEW: allow looser build if too few unique
    force_loose = False

    while len(cands) < pool_size and tries < max_tries:
        tries += 1
        lineup = []

        if game_stack_min > 0 and len(by_game) >= 1 and not force_loose:
            gkey = random.choice(list(by_game.keys()))
            pre = random.sample(by_game[gkey], k=min(game_stack_min, len(by_game[gkey])))
            lineup.extend(pre)

        while len(lineup) < 9:
            remaining_slots = 9 - len(lineup)
            cur_sal = sum(x["salary"] for x in lineup)
            if not salary_bounds_ok(cur_sal, remaining_slots) and not force_loose:
                lineup = []
                break

            pool = [p for p in records if p not in lineup]
            if not pool:
                lineup = []
                break

            # LOOSER candidate selection — skip greedy pos filter
            cand = random.choice(pool)
            if cur_sal + cand.get("salary",0) > cap:
                continue
            lineup.append(cand)

        if not lineup:
            continue

        if valid_lineup(lineup, cap, salary_min):
            cands.append([p["player_id"] for p in lineup])

        # if after many tries we have too few, relax
        if tries > pool_size * relax_threshold and len(cands) < 50:
            force_loose = True

    # de-dup but allow large pool to pass
    uniq, seen = [], set()
    for ids in cands:
        key = tuple(sorted(ids))
        if key not in seen:
            uniq.append(ids)
            seen.add(key)
    # if still tiny, return raw list
    if len(uniq) < 100:
        return cands
    return uniq

# -------- Simulation --------

def simulate(df: pd.DataFrame, cov: np.ndarray, sims: int, lineup_ids: list, cash_threshold: float):
    idx = {pid: i for i, pid in enumerate(df["player_id"].tolist())}
    lineup_index = [[idx[i] for i in ids] for ids in lineup_ids]
    means = df["proj_mean"].to_numpy()

    win = np.zeros(len(lineup_ids), dtype=int)
    cash = np.zeros(len(lineup_ids), dtype=int)
    totals = [[] for _ in lineup_ids]

    chunk = max(1, min(400, sims))
    done = 0
    while done < sims:
        m = min(chunk, sims - done)
        draws = np.random.multivariate_normal(mean=means, cov=cov, size=m)
        draws = np.maximum(draws, 0.0)
        scores = np.zeros((m, len(lineup_ids)))
        for j, ix in enumerate(lineup_index):
            scores[:, j] = draws[:, ix].sum(axis=1)
        winners = scores.argmax(axis=1)
        for w in winners:
            win[w] += 1
        cash += (scores >= cash_threshold).sum(axis=0)
        mask = np.random.rand(m) < 0.05
        sm = scores[mask]
        for j in range(len(lineup_ids)):
            totals[j].extend(sm[:, j].tolist())
        done += m

    rows = []
    for j, ids in enumerate(lineup_ids):
        arr = np.array(totals[j]) if totals[j] else np.array([0.0])
        rows.append({
            "lineup_key": ",".join(ids),
            "win_pct": win[j] / sims,
            "cash_pct": cash[j] / sims,
            "mean": float(arr.mean()),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
        })
    stats = pd.DataFrame(rows).sort_values(["win_pct","p90","mean"], ascending=[False, False, False])

    # lineup pool detail
    pool_rows = []
    for j, ids in enumerate(lineup_ids):
        for pid in ids:
            pool_rows.append({"lineup_ix": j, "player_id": pid})
    pool_df = pd.DataFrame(pool_rows)
    return stats, pool_df

# -------- Exposure --------

def player_exposure(stats: pd.DataFrame, pool_df: pd.DataFrame, top_n: int, df_players: pd.DataFrame) -> pd.DataFrame:
    top = stats.head(top_n).index.tolist()
    sub = pool_df[pool_df["lineup_ix"].isin(top)]
    exp = sub.groupby("player_id").nunique()["lineup_ix"].reset_index().rename(columns={"lineup_ix":"lineups"})
    merged = exp.merge(df_players[["player_id","name","team","pos","salary","proj_mean"]], on="player_id", how="left")
    merged["exposure_pct"] = merged["lineups"] / min(top_n, stats.shape[0])
    return merged.sort_values(["exposure_pct","proj_mean"], ascending=[False, False])

# -------- UI Controls --------

# (Reverted) fixed balanced build settings
pool_multiplier = 2000
relax_threshold = 50    # relax later
st.write("### Settings — Simplified Explanations")
st.caption("**Set how the simulator builds and scores lineups. Keep it simple:**")
stack_bias = st.selectbox(
    "Optional game-stack bias (min players from same game)",
    [0,2,3,4],
    index=0,
    help="Force at least X players from same game. 0 = no stacking bias (safer). Higher = more game correlation (higher variance)."
)
lineup_pool = st.number_input(
    "Candidate lineup pool",
    1000, 80000, 10000, 1000,
    help="How many random lineups to generate before simulation. More = stronger lineups, but slower."
)
sims = st.number_input(
    "Simulations",
    500, 20000, 3000, 500,
    help="How many fantasy slates to simulate. Higher = more accurate, slower. 3000 is solid."
)
salary_min = st.number_input(
    "Min salary",
    50000, 60000, 58000, 500,
    help="Lowest allowed salary used. Set slightly below cap so value lineups aren’t excluded."
)
cash_threshold = st.number_input(
    "Cash threshold (FD pts)",
    200.0, 400.0, 300.0, 10.0,
    help="Score needed to count as a cash in sim. ~280-300 for most slates."
)

run = st.button("Run Simulation")
if not run:
    st.stop()

st.write("Building correlation/covariance…")
corr = build_corr_matrix(df)
cov = build_cov_matrix(df, corr)

st.write("Generating candidate lineups…")
ids = build_candidate_pool(df, pool_size=lineup_pool, cap=FD_CAP, salary_min=salary_min, game_stack_min=stack_bias)
if not ids:
    st.error("No valid lineups. Lower salary min, reduce game stack bias, or increase pool size.")
    st.stop()

st.write(f"Generated {len(ids)} candidate lineups. Running simulations…")
stats, pool_df = simulate(df, cov, sims, ids, cash_threshold)

st.success("Done!")

st.subheader("Top Lineups")

# ---- Column tooltips via column_config ----
col_help = {
    "players": "Readable roster: POS Name ($salary) for each of the 9 slots.",
    "total_salary": "Sum of salaries for the lineup. Shown for context only; not graded.",
    "win_pct": "Win rate across the simulation runs (how often this lineup finished 1st). >1% is excellent.",
    "cash_pct": "Cash rate across the simulation runs (how often it reached a cashable score). >60% is elite floor.",
    "mean": "Average simulated fantasy score for the lineup.",
    "p75": "75th percentile outcome — solid night expectation.",
    "p90": "90th percentile outcome — ceiling potential for GPPs.",
    "avg_rank": "Average finishing position across small re-sims. Lower is better (≈1–3 is excellent).",
    "stability_score": "Consistency of ranks across re-sims. Higher = more reliable (≥0.80 elite)."
}


# Stability score will be computed after stats_display is created
stability_settings = True
stab_repeats = st.slider("Re-sim times (small, for stability check)", 1, 5, 3)
stability_sims = int(sims * 0.25)
# mark for later
st.session_state["stab_repeats"] = stab_repeats
st.session_state["stability_sims"] = stability_sims
st.session_state["compute_stability"] = True

# === Add readable player names in results ===
# Map player_id -> name/pos/salary
name_map = df.set_index("player_id")["name"].to_dict()
pos_map  = df.set_index("player_id")["pos"].to_dict()
sal_map  = df.set_index("player_id")["salary"].to_dict()

# Convert lineup_key from ids to text lines

def format_lineup(id_string):
    ids = id_string.split(",")
    txt = []
    for pid in ids:
        nm = name_map.get(pid, pid)
        pos = pos_map.get(pid, "")
        sal = sal_map.get(pid, 0)
        txt.append(f"{pos} {nm} (${sal})")
    return " | ".join(txt)

stats_display = stats.copy()
# add total salary column
sal_map = df.set_index("player_id")["salary"].to_dict()
stats_display["total_salary"] = stats_display["lineup_key"].apply(lambda s: sum(sal_map.get(pid,0) for pid in s.split(",")))
stats_display["players"] = stats_display["lineup_key"].apply(format_lineup)

# Format salary as $xx,xxx for display (neutral — no grading)
stats_display["total_salary_fmt"] = stats_display["total_salary"].map(lambda x: f"${x:,.0f}")

# Simple emoji-based grading (subtle, no salary grading)
def grade_win(x):
    return "🟢" if x>=0.010 else ("🟡" if x>=0.005 else "🔴")

def grade_cash(x):
    return "🟢" if x>=0.60 else ("🟡" if x>=0.50 else "🔴")

def grade_p90(x):
    return "🟢" if x>=345 else ("🟡" if x>=330 else "🔴")

def grade_stab(x):
    return "🟢" if x>=0.80 else ("🟡" if x>=0.65 else "🔴")

def grade_avg_rank(x):
    return "🟢" if x<=2 else ("🟡" if x<=4 else "🔴")

# Compute grade columns if present (will be NaN before stability is merged)
for col, fn in {
    "win_grade": ("win_pct", grade_win),
    "cash_grade": ("cash_pct", grade_cash),
    "p90_grade": ("p90", grade_p90)
}.items():
    metric, f = fn
    stats_display[col] = stats_display[metric].apply(f)

# Base results table (pre-stability)
base_cols = [
    "players",
    "total_salary_fmt",
    "win_grade","win_pct",
    "cash_grade","cash_pct",
    "mean","p75","p90_grade","p90",
]

st.dataframe(
    stats_display[base_cols].head(20),
    column_config={
        "players": st.column_config.TextColumn("Players", help=col_help["players"], width="large"),
        "total_salary_fmt": st.column_config.TextColumn("Total Salary", help=col_help["total_salary"]),
        "win_grade": st.column_config.TextColumn("Win Grade", help="Quick color grade for Win% (🟢 ≥1%, 🟡 0.5–1%, 🔴 <0.5%)"),
        "win_pct": st.column_config.NumberColumn("Win %", help=col_help["win_pct"], format="%.3f"),
        "cash_grade": st.column_config.TextColumn("Cash Grade", help="Quick color grade for Cash% (🟢 ≥60%, 🟡 50–60%, 🔴 <50%)"),
        "cash_pct": st.column_config.NumberColumn("Cash %", help=col_help["cash_pct"], format="%.3f"),
        "mean": st.column_config.NumberColumn("Mean", help=col_help["mean"], format="%.2f"),
        "p75": st.column_config.NumberColumn("p75", help=col_help["p75"], format="%.2f"),
        "p90_grade": st.column_config.TextColumn("p90 Grade", help="Quick color grade for p90 (🟢 ≥345, 🟡 330–344, 🔴 <330)"),
        "p90": st.column_config.NumberColumn("p90", help=col_help["p90"], format="%.2f"),
    },
    use_container_width=True,
)

# Legend below the table
with st.expander("📘 DFS Interpretation Guide", expanded=False):
    st.markdown(
        """
**Win %** — Simulated 1st place rate. **≥1.5%** elite, **1.0–1.4%** top-end, **0.5–0.9%** strong.

**Cash %** — Simulated min-cash rate. **≥60%** elite floor, **50–59%** solid, **<50%** high variance only.

**p90** — 90th percentile score (ceiling). **≥345** slate-breaking, **330–344** tournament viable.

**Stability Score** — Rank consistency across re-sims (higher is better). **≥0.80** elite, **0.65–0.79** strong.

**Avg Rank** — Average finishing position in re-sims (lower is better). **≤2** excellent, **2–4** good.

Salary is shown for transparency only; **no color/grade is applied**.
        """
    )

# --- Stability computation block ---
if st.session_state.get("compute_stability"):
    st.write("### Stability Results (experimental)")
    repeats = st.session_state.get("stab_repeats", 3)
    stability_sims = st.session_state.get("stability_sims", int(sims * 0.25))

    stability_results = []
    for _ in range(repeats):
        s_stats, _ = simulate(df, cov, stability_sims, ids, cash_threshold)
        for rank, key in enumerate(s_stats["lineup_key"].tolist()):
            stability_results.append((key, rank))

    stab_df = pd.DataFrame(stability_results, columns=["lineup_key", "rank"])
    stab_summary = stab_df.groupby("lineup_key").agg(avg_rank=("rank","mean"), rank_var=("rank", "var"))
    stab_summary["stability_score"] = 1 / (1 + stab_summary["rank_var"].fillna(0))

    # merge with display
    stats_display = stats_display.merge(stab_summary, on="lineup_key", how="left")
    stats_display.sort_values(["stability_score", "win_pct"], ascending=[False, False], inplace=True)

    # Recompute grades now that stability is available
if "stability_score" in stats_display.columns:
    stats_display["stab_grade"] = stats_display["stability_score"].apply(grade_stab)
    stats_display["avg_rank_grade"] = stats_display["avg_rank"].apply(grade_avg_rank)

    final_cols = [
        "players",
        "total_salary_fmt",
        "win_grade","win_pct",
        "cash_grade","cash_pct",
        "mean","p75","p90_grade","p90",
        "avg_rank_grade","avg_rank",
        "stab_grade","stability_score",
    ]

    st.dataframe(
        stats_display[final_cols].head(20),
        column_config={
            "players": st.column_config.TextColumn("Players", help=col_help["players"], width="large"),
            "total_salary_fmt": st.column_config.TextColumn("Total Salary", help=col_help["total_salary"]),
            "win_grade": st.column_config.TextColumn("Win Grade", help="Quick color grade for Win% (🟢 ≥1%, 🟡 0.5–1%, 🔴 <0.5%)"),
            "win_pct": st.column_config.NumberColumn("Win %", help=col_help["win_pct"], format="%.3f"),
            "cash_grade": st.column_config.TextColumn("Cash Grade", help="Quick color grade for Cash% (🟢 ≥60%, 🟡 50–60%, 🔴 <50%)"),
            "cash_pct": st.column_config.NumberColumn("Cash %", help=col_help["cash_pct"], format="%.3f"),
            "mean": st.column_config.NumberColumn("Mean", help=col_help["mean"], format="%.2f"),
            "p75": st.column_config.NumberColumn("p75", help=col_help["p75"], format="%.2f"),
            "p90_grade": st.column_config.TextColumn("p90 Grade", help="Quick color grade for p90 (🟢 ≥345, 🟡 330–344, 🔴 <330)"),
            "p90": st.column_config.NumberColumn("p90", help=col_help["p90"], format="%.2f"),
            "avg_rank_grade": st.column_config.TextColumn("Avg Rank Grade", help="Quick grade for avg rank (🟢 ≤2, 🟡 2–4, 🔴 >4)"),
            "avg_rank": st.column_config.NumberColumn("Avg Rank", help=col_help["avg_rank"], format="%.2f"),
            "stab_grade": st.column_config.TextColumn("Stability Grade", help="Quick grade for stability (🟢 ≥0.80, 🟡 0.65–0.79, 🔴 <0.65)"),
            "stability_score": st.column_config.NumberColumn("Stability", help=col_help["stability_score"], format="%.3f"),
        },
        use_container_width=True,
    )

    with st.expander("📘 DFS Interpretation Guide", expanded=False):
        st.markdown(
            """
**Win %** — Simulated 1st place rate. **≥1.5%** elite, **1.0–1.4%** top-end, **0.5–0.9%** strong.

**Cash %** — Simulated min-cash rate. **≥60%** elite floor, **50–59%** solid, **<50%** high variance only.

**p90** — 90th percentile score (ceiling). **≥345** slate-breaking, **330–344** tournament viable.

**Stability Score** — Rank consistency across re-sims (higher is better). **≥0.80** elite, **0.65–0.79** strong.

**Avg Rank** — Average finishing position in re-sims (lower is better). **≤2** excellent, **2–4** good.

Salary is shown for transparency only; **no color/grade is applied**.
            """
        )
    st.caption("Higher stability score + low avg rank = lineup that consistently finishes near the top across re-sims.")
