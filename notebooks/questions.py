# %% [markdown]
# # 02 Specific questions
# Day 3: five questions a farmer (or extension officer) in Kakamega,
# Bungoma or Trans Nzoia might actually ask.
#
# Q1. Is Trans Nzoia's post-2015 shift real or a data artefact?
# Q2. When is it safe to plant, and how often do the rains "false start"?
# Q3. In El Nino years, how much more likely is a wet short-rains season?
# Q4. What did past strong El Nino seasons look like (1997, 2015, 2023)?
# Q5. Is heat stress for maize becoming more common?
#
# Run cell by cell in VS Code, or `python notebooks/02_questions.py`.

# %%
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, mannwhitneyu, theilslopes

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
DB = ROOT / "data" / "weather.db"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = (1991, 2020)
LAST_FULL_YEAR = 2025
OND = [10, 11, 12]

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def savefig(name):
    plt.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


conn = sqlite3.connect(DB)
df = pd.read_sql("SELECT * FROM weather", conn, parse_dates=["date"])
oni = pd.read_sql("SELECT * FROM oni", conn)
conn.close()

df["year"] = df.date.dt.year
df["month"] = df.date.dt.month
df["rain"] = df.precipitation_sum
full = df[df.year <= LAST_FULL_YEAR].copy()

oni_ond = oni[oni.season == "OND"].set_index("year").anomaly.rename("oni")


def phase(a):
    if pd.isna(a):
        return np.nan
    if a >= 0.5:
        return "El Niño"
    if a <= -0.5:
        return "La Niña"
    return "Neutral"


answers = []   # one-line answers collected for the README / post


def answer(q, text):
    print(f"\n>>> {q}: {text}")
    answers.append({"question": q, "answer": text})


# %% [markdown]
# ## Q1. Is Trans Nzoia's post-2015 shift real?
# Pettitt test finds the most likely single change point in a series.
# If temperature AND rainfall jump in the same year across every
# sub-county, it looks more like a change in the data source than a
# gradual climate trend, so we flag it rather than build on it.

# %%
def pettitt(x):
    """Return (index of change point, approximate p-value)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    s = np.sign(x[:, None] - x[None, :])
    u = np.array([s[: t + 1, t + 1:].sum() for t in range(n - 1)])
    k = np.abs(u).max()
    p = min(1.0, 2 * np.exp(-6 * k ** 2 / (n ** 3 + n ** 2)))
    return int(np.abs(u).argmax()), p


annual = (full.groupby(["county", "subcounty", "year"])
          .agg(rain=("rain", "sum"), tmean=("temperature_2m_mean", "mean"))
          .reset_index())

rows = []
for (county, sub), g in annual.groupby(["county", "subcounty"]):
    g = g.sort_values("year")
    for var in ["tmean", "rain"]:
        idx, p = pettitt(g[var].values)
        cp_year = int(g.year.iloc[idx + 1])
        before = g[g.year < cp_year][var].mean()
        after = g[g.year >= cp_year][var].mean()
        rows.append({"county": county, "subcounty": sub, "var": var,
                     "change_year": cp_year, "p_value": round(p, 3),
                     "before": round(before, 2), "after": round(after, 2),
                     "shift": round(after - before, 2)})
cps = pd.DataFrame(rows)
cps.to_csv(OUT / "q1_change_points.csv", index=False)
print(cps.to_string(index=False))

tz = cps[(cps.county == "Trans Nzoia") & (cps.p_value < 0.05)]
same_year = tz.groupby("var").change_year.nunique().le(2).all() and len(tz) >= 4
if len(tz) == 0:
    answer("Q1", "No significant change point in Trans Nzoia; the post-2015 "
                 "pattern is within normal variability.")
elif same_year:
    yrs = sorted(tz.change_year.unique())
    answer("Q1", f"Trans Nzoia shows an abrupt, simultaneous step in temperature "
                 f"and rainfall around {yrs}. A step this sudden across all "
                 f"sub-counties is more consistent with a data change than a "
                 f"gradual climate trend, so treat it with caution and verify "
                 f"against another source (e.g. NASA POWER or KMD station data).")
else:
    answer("Q1", "Trans Nzoia has significant change points, but not all in the "
                 "same year, which fits a real (if uneven) shift better than a "
                 "data artefact.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, var, label in [(axes[0], "tmean", "Mean temperature (°C)"),
                       (axes[1], "rain", "Annual rainfall (mm)")]:
    for sub, g in annual[annual.county == "Trans Nzoia"].groupby("subcounty"):
        ax.plot(g.year, g[var], label=sub)
    for _, r in cps[(cps.county == "Trans Nzoia") & (cps["var"] == var)].iterrows():
        ax.axvline(r.change_year, color="grey", ls="--", lw=1)
    ax.set_title(f"Trans Nzoia: {label}")
axes[0].legend()
savefig("q1_trans_nzoia_change_points")


# %% [markdown]
# ## Q2. When is it safe to plant?
# For each season and county: the typical onset date (median), an early
# and late year (20th and 80th percentile), and how often the first
# heavy rain was a false start (followed by a 10+ day dry spell within
# 30 days, which can kill newly planted seed).

# %%
WINDOWS = {"long_rains": ("02-15", "05-31"), "short_rains": ("09-01", "11-30")}


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def first_wet(vals, dates, end, wet_total=20):
    for i in range(len(vals)):
        if dates[i] > end:
            return None
        if vals[i] >= 1 and vals[i:i + 3].sum() >= wet_total:
            return i
    return None


rows = []
for (county, sub), g in full.groupby(["county", "subcounty"]):
    rain = g.set_index("date").rain.sort_index()
    for year in range(full.year.min(), LAST_FULL_YEAR + 1):
        for season, (s, e) in WINDOWS.items():
            start, end = pd.Timestamp(f"{year}-{s}"), pd.Timestamp(f"{year}-{e}")
            r = rain.loc[start:end + pd.Timedelta(days=40)]
            vals, dates = r.values, r.index
            i = first_wet(vals, dates, end)
            if i is None:
                rows.append({"county": county, "subcounty": sub, "year": year,
                             "season": season, "false_start": np.nan,
                             "onset_doy": np.nan})
                continue
            false = longest_run(vals[i + 3:i + 33] < 1) >= 10
            # true onset: first wet start that is NOT followed by a long dry spell
            j, onset = i, None
            while j is not None:
                if longest_run(vals[j + 3:j + 33] < 1) < 10:
                    onset = dates[j]
                    break
                nxt = first_wet(vals[j + 1:], dates[j + 1:], end)
                j = None if nxt is None else j + 1 + nxt
            rows.append({"county": county, "subcounty": sub, "year": year,
                         "season": season, "false_start": false,
                         "onset_doy": onset.dayofyear if onset is not None else np.nan})
plant = pd.DataFrame(rows)
plant.to_csv(OUT / "q2_onsets.csv", index=False)


def doy_to_date(d):
    return "n/a" if pd.isna(d) else (pd.Timestamp("2025-01-01")
                                     + pd.Timedelta(days=int(d) - 1)).strftime("%d %b")


summary = (plant.groupby(["county", "season"])
           .agg(early=("onset_doy", lambda x: x.quantile(0.2)),
                typical=("onset_doy", "median"),
                late=("onset_doy", lambda x: x.quantile(0.8)),
                false_start_pct=("false_start", lambda x: 100 * x.mean()))
           .reset_index())
for c in ["early", "typical", "late"]:
    summary[c] = summary[c].apply(doy_to_date)
summary["false_start_pct"] = summary.false_start_pct.round(0)
summary.to_csv(OUT / "q2_planting_calendar.csv", index=False)
print(summary.to_string(index=False))

for _, r in summary.iterrows():
    answer("Q2", f"{r.county} {r.season.replace('_', ' ')}: onset typically "
                 f"{r.typical} (early year {r.early}, late year {r.late}); "
                 f"{r.false_start_pct:.0f}% of seasons had a false start.")

plt.figure(figsize=(10, 4))
sns.boxplot(data=plant.dropna(subset=["onset_doy"]), x="county", y="onset_doy",
            hue="season")
plt.ylabel("Onset (day of year)")
plt.title("When do the rains reliably start?")
savefig("q2_onset_distribution")


# %% [markdown]
# ## Q3. How much does El Nino change the odds?
# Terciles: each county's OND totals over 1991-2020 are split into
# below / near / above normal thirds. In a random year each is ~33%.
# We then count how often El Nino years land in each third, and add
# flood-related indicators: heavy rain days (> 95th percentile) and
# the wettest 5-day total.

# %%
ond = (full[full.month.isin(OND)]
       .groupby(["county", "subcounty", "year"])
       .agg(ond_rain=("rain", "sum"),
            max_5day=("rain", lambda x: x.rolling(5).sum().max()))
       .reset_index())

p95 = (full[full.year.between(*BASELINE) & (full.rain >= 1)]
       .groupby(["county", "subcounty"]).rain.quantile(0.95).rename("p95"))
heavy = (full[full.month.isin(OND)].join(p95, on=["county", "subcounty"])
         .assign(heavy=lambda d: d.rain > d.p95)
         .groupby(["county", "subcounty", "year"]).heavy.sum().rename("heavy_days"))
ond = ond.join(heavy, on=["county", "subcounty", "year"])

county_ond = (ond.groupby(["county", "year"])
              [["ond_rain", "max_5day", "heavy_days"]].mean().reset_index())
county_ond = county_ond.join(oni_ond, on="year")
county_ond["phase"] = county_ond.oni.apply(phase)

base = county_ond[county_ond.year.between(*BASELINE)]
terc = base.groupby("county").ond_rain.quantile([1 / 3, 2 / 3]).unstack()
terc.columns = ["t1", "t2"]
county_ond = county_ond.join(terc, on="county")
county_ond["tercile"] = np.select(
    [county_ond.ond_rain < county_ond.t1, county_ond.ond_rain > county_ond.t2],
    ["below", "above"], "near")
county_ond.to_csv(OUT / "q3_ond_by_phase.csv", index=False)

odds = (pd.crosstab([county_ond.county, county_ond.phase], county_ond.tercile,
                    normalize="index") * 100).round(0)
odds = odds.reindex(columns=["below", "near", "above"], fill_value=0)
print("Chance of each OND outcome (%) by ENSO phase:")
print(odds)

for county, g in county_ond.dropna(subset=["phase"]).groupby("county"):
    en, nt = g[g.phase == "El Niño"], g[g.phase == "Neutral"]
    if len(en) < 3 or len(nt) < 3:
        continue
    p_above = (en.tercile == "above").mean() * 100
    p_below = (en.tercile == "below").mean() * 100
    _, p = mannwhitneyu(en.ond_rain, nt.ond_rain, alternative="greater")
    extra = (en.ond_rain.mean() / nt.ond_rain.mean() - 1) * 100
    answer("Q3", f"{county}: in {len(en)} El Niño years, OND rain was above normal "
                 f"{p_above:.0f}% of the time and below normal {p_below:.0f}% "
                 f"(vs ~33% each in a random year). Average OND rain was "
                 f"{extra:+.0f}% vs neutral years (one-sided Mann-Whitney p = {p:.3f}). "
                 f"Heavy rain days: {en.heavy_days.mean():.1f} vs "
                 f"{nt.heavy_days.mean():.1f}.")

plot = odds.reset_index().melt(id_vars=["county", "phase"], value_name="pct")
g = sns.catplot(data=plot, x="phase", y="pct", hue="tercile", col="county",
                kind="bar", order=["La Niña", "Neutral", "El Niño"],
                hue_order=["below", "near", "above"], height=4, aspect=1)
for ax in g.axes.flat:
    ax.axhline(33.3, color="grey", ls="--", lw=1)
g.set_axis_labels("", "% of years")
g.fig.suptitle("Short rains outcome by ENSO phase (dashed line = random chance)",
               y=1.04)
savefig("q3_enso_odds")


# %% [markdown]
# ## Q4. What did past strong El Nino seasons look like?
# Monthly rainfall for OND + January in strong events (ONI >= 1.5),
# compared with the 1991-2020 average. This is the closest historical
# analogue for the 2026 season.

# %%
strong_years = sorted(oni_ond[oni_ond >= 1.5].index)
strong_years = [y for y in strong_years if y <= LAST_FULL_YEAR]
print("Strong El Niño years in record:", strong_years)

monthly = (full.groupby(["county", "year", "month"]).rain.sum()
           .groupby(["county", "year", "month"]).mean().reset_index())
clim = (monthly[monthly.year.between(*BASELINE)]
        .groupby(["county", "month"]).rain.mean().rename("normal"))

if strong_years:
    rows = []
    for y in strong_years:
        for m, yr in [(10, y), (11, y), (12, y), (1, y + 1)]:
            sel = monthly[(monthly.year == yr) & (monthly.month == m)]
            for _, r in sel.iterrows():
                rows.append({"event": y, "county": r.county, "month": m,
                             "rain": r.rain})
    analog = pd.DataFrame(rows).join(clim, on=["county", "month"])
    analog["pct_of_normal"] = (analog.rain / analog.normal * 100).round(0)
    analog.to_csv(OUT / "q4_strong_el_nino.csv", index=False)
    print(analog.pivot_table(index=["county", "event"], columns="month",
                             values="pct_of_normal")
          .reindex(columns=[10, 11, 12, 1]))

    season_pct = (analog.groupby(["county", "event"])
                  .apply(lambda d: d.rain.sum() / d.normal.sum() * 100,
                         include_groups=False).round(0))
    for county, s in season_pct.groupby(level="county"):
        parts = ", ".join(f"{e}: {v:.0f}%" for (_, e), v in s.items())
        answer("Q4", f"{county}: Oct-Jan rainfall in strong El Niño years as % "
                     f"of normal: {parts}.")

    month_names = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan"}
    analog["month_name"] = analog.month.map(month_names)
    g = sns.catplot(data=analog, x="month_name", y="pct_of_normal", hue="event",
                    col="county", kind="bar", order=list(month_names.values()),
                    height=4, aspect=1)
    for ax in g.axes.flat:
        ax.axhline(100, color="grey", ls="--", lw=1)
    g.set_axis_labels("", "% of normal rainfall")
    g.fig.suptitle("Strong El Niño seasons vs normal (dashed line = 100%)", y=1.04)
    savefig("q4_strong_el_nino")
else:
    answer("Q4", "No strong El Niño (ONI >= 1.5) years found in the record.")


# %% [markdown]
# ## Q5. Is heat stress for maize becoming more common?
# Maize yields drop when daily maximum temperature goes above ~30°C,
# especially around flowering. Count days above 30°C per year during
# the main growing months (March to August) and test for a trend.

# %%
HEAT_T = 30
grow = full[full.month.between(3, 8)]
heat = (grow.assign(hot=grow.temperature_2m_max > HEAT_T)
        .groupby(["county", "subcounty", "year"]).hot.sum().reset_index())
county_heat = heat.groupby(["county", "year"]).hot.mean().reset_index()
county_heat.to_csv(OUT / "q5_heat_days.csv", index=False)

for county, g in county_heat.groupby("county"):
    tau, p = kendalltau(g.year, g.hot)
    slope = theilslopes(g.hot, g.year)[0] * 10
    early = g[g.year <= 2000].hot.mean()
    late = g[g.year >= 2016].hot.mean()
    sig = "significant" if p < 0.05 else "not significant"
    answer("Q5", f"{county}: {early:.1f} days above {HEAT_T}°C per growing season "
                 f"in 1991-2000 vs {late:.1f} in 2016-2025 "
                 f"({slope:+.1f} days/decade, {sig}, p = {p:.3f}).")

plt.figure(figsize=(9, 4))
sns.lineplot(data=county_heat, x="year", y="hot", hue="county")
plt.ylabel(f"Days with Tmax > {HEAT_T}°C (Mar-Aug)")
plt.title("Heat stress days in the growing season")
savefig("q5_heat_days")


# %% [markdown]
# ## Answers
# Saved to data/processed/answers.csv. Copy these into the README and
# the Day 78 post.

# %%
ans = pd.DataFrame(answers)
ans.to_csv(OUT / "answers.csv", index=False)
for q, g in ans.groupby("question"):
    print(f"\n{q}")
    for a in g.answer:
        print(f"  - {a}")