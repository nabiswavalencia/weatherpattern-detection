
# %%
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, theilslopes

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
DB = ROOT / "data" / "weather.db"
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "data" / "processed"
FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = (1991, 2020)
LAST_FULL_YEAR = 2025          # 2026 is incomplete, keep it out of trends
SEASONS = {"MAM": [3, 4, 5], "JJA": [6, 7, 8], "OND": [10, 11, 12]}

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
df["site"] = df.county + " / " + df.subcounty
df["rain"] = df.precipitation_sum
full = df[df.year <= LAST_FULL_YEAR]

print(df.shape, df.date.min().date(), "->", df.date.max().date())

# %% [markdown]
# ## 1. Are sub-counties actually different?
# If daily rainfall correlation between two sub-counties is above ~0.95,
# they are probably reading the same ERA5 grid cell and should be merged.

# %%
annual = (
    full.groupby(["county", "subcounty", "year"])
    .agg(rain=("rain", "sum"),
         tmean=("temperature_2m_mean", "mean"),
         tmax=("temperature_2m_max", "mean"),
         tmin=("temperature_2m_min", "mean"),
         et0=("et0_fao_evapotranspiration", "sum"))
    .reset_index()
)
print(annual.groupby(["county", "subcounty"])[["rain", "tmean", "et0"]]
      .mean().round(1))

wide = df.pivot_table(index="date", columns="site", values="rain")
plt.figure(figsize=(9, 7))
sns.heatmap(wide.corr(), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
plt.title("Daily rainfall correlation between sub-counties")
savefig("01_subcounty_correlation")

# %% [markdown]
# ## 2. Seasonal cycle (1991-2020 baseline)

# %%
base = full[full.year.between(*BASELINE)]
monthly = (
    base.groupby(["county", "subcounty", "year", "month"]).rain.sum()
    .groupby(["county", "month"]).mean().reset_index()
)
plt.figure(figsize=(9, 4))
sns.lineplot(data=monthly, x="month", y="rain", hue="county", marker="o")
plt.xticks(range(1, 13), ["J", "F", "M", "A", "M", "J",
                          "J", "A", "S", "O", "N", "D"])
plt.ylabel("Mean monthly rainfall (mm)")
plt.title("Rainfall seasonal cycle, 1991-2020")
savefig("02_seasonal_cycle")

# %% [markdown]
# ## 3. Long-term trends
# Mann-Kendall test (Kendall's tau against year) for significance and
# Sen's slope for the size of the change, reported per decade.

# %%
month_to_season = {m: s for s, ms in SEASONS.items() for m in ms}
full = full.assign(season=full.month.map(month_to_season))
seasonal = (
    full.dropna(subset=["season"])
    .groupby(["county", "subcounty", "year", "season"]).rain.sum()
    .unstack("season").reset_index()
)
metrics = annual.merge(seasonal, on=["county", "subcounty", "year"])


def trend(years, values):
    mask = ~np.isnan(values)
    years, values = years[mask], values[mask]
    tau, p = kendalltau(years, values)
    slope = theilslopes(values, years)[0]
    return slope * 10, p


rows = []
for (county, sub), g in metrics.groupby(["county", "subcounty"]):
    for m in ["rain", "MAM", "JJA", "OND", "tmean", "tmax", "tmin", "et0"]:
        slope, p = trend(g.year.values, g[m].values.astype(float))
        rows.append({"county": county, "subcounty": sub, "metric": m,
                     "per_decade": round(slope, 3), "p_value": round(p, 3),
                     "significant": p < 0.05})
trends = pd.DataFrame(rows)
trends.to_csv(OUT / "trends.csv", index=False)

print(trends.pivot_table(index=["county", "subcounty"], columns="metric",
                         values="per_decade").round(2))
print("\nSignificant trends (p < 0.05):")
print(trends[trends.significant].to_string(index=False))

# %%
county_annual = annual.groupby(["county", "year"])[["rain", "tmean"]].mean()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for county, g in county_annual.groupby(level="county"):
    g = g.droplevel("county")
    axes[0].plot(g.index, g.tmean, label=county)
    axes[1].plot(g.index, g.rain, label=county)
axes[0].set_title("Annual mean temperature (°C)")
axes[1].set_title("Annual rainfall (mm)")
axes[0].legend()
savefig("03_annual_trends")

# %% [markdown]
# ## 4. Rainy season onset and dry spells
# Onset = first wet day (>= 1 mm) where the 3-day total reaches 20 mm
# and no dry spell of 10+ days (< 1 mm/day) follows within 30 days.
# This is a common agronomic definition: it filters out "false starts"
# that kill seedlings.

# %%
ONSET_WINDOWS = {"long_rains": ("02-15", "05-31"),
                 "short_rains": ("09-01", "11-30")}


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def find_onset(rain, start, end, wet_total=20, dry_len=10, look=30):
    r = rain.loc[start:end + pd.Timedelta(days=look + 3)]
    vals, dates = r.values, r.index
    for i in range(len(vals)):
        if dates[i] > end:
            break
        if vals[i] >= 1 and vals[i:i + 3].sum() >= wet_total:
            after = vals[i + 3:i + 3 + look] < 1
            if longest_run(after) < dry_len:
                return dates[i]
    return pd.NaT


rows = []
for (county, sub), g in df.groupby(["county", "subcounty"]):
    rain = g.set_index("date").rain.sort_index()
    for year in range(df.year.min(), LAST_FULL_YEAR + 1):
        for season, (s, e) in ONSET_WINDOWS.items():
            start, end = pd.Timestamp(f"{year}-{s}"), pd.Timestamp(f"{year}-{e}")
            onset = find_onset(rain, start, end)
            months = SEASONS["MAM"] if season == "long_rains" else SEASONS["OND"]
            in_season = rain[(rain.index.year == year)
                             & rain.index.month.isin(months)]
            rows.append({"county": county, "subcounty": sub, "year": year,
                         "season": season, "onset": onset,
                         "onset_doy": onset.dayofyear if pd.notna(onset) else np.nan,
                         "max_dry_spell": longest_run(in_season.values < 1)})
onsets = pd.DataFrame(rows)
onsets.to_csv(OUT / "onsets.csv", index=False)

print("Years with no onset found:",
      onsets.onset.isna().groupby(onsets.season).sum().to_dict())

rows = []
for (county, sub, season), g in onsets.groupby(["county", "subcounty", "season"]):
    for m in ["onset_doy", "max_dry_spell"]:
        slope, p = trend(g.year.values, g[m].values.astype(float))
        rows.append({"county": county, "subcounty": sub, "season": season,
                     "metric": m, "days_per_decade": round(slope, 2),
                     "p_value": round(p, 3)})
onset_trends = pd.DataFrame(rows)
onset_trends.to_csv(OUT / "onset_trends.csv", index=False)
print(onset_trends.to_string(index=False))

# %%
county_onset = (onsets.groupby(["county", "season", "year"]).onset_doy
                .mean().reset_index())
g = sns.relplot(data=county_onset, x="year", y="onset_doy", hue="county",
                col="season", kind="line", facet_kws={"sharey": False},
                height=4, aspect=1.4)
g.set_axis_labels("Year", "Onset (day of year)")
g.fig.suptitle("Rainy season onset over time", y=1.03)
savefig("04_onset_trends")

# %% [markdown]
# ## 5. El Nino vs normal seasons
# OND rainfall anomaly (% of the 1991-2020 mean) grouped by the OND ONI
# value: El Nino >= +0.5, La Nina <= -0.5, otherwise neutral.

# %%
oni_ond = oni[oni.season == "OND"].set_index("year").anomaly.rename("oni")


def phase(a):
    if a >= 0.5:
        return "El Niño"
    if a <= -0.5:
        return "La Niña"
    return "Neutral"


ond = seasonal[["county", "subcounty", "year", "OND"]].copy()
base_ond = (ond[ond.year.between(*BASELINE)]
            .groupby(["county", "subcounty"]).OND.mean().rename("ond_base"))
ond = ond.join(base_ond, on=["county", "subcounty"])
ond["anomaly_pct"] = (ond.OND / ond.ond_base - 1) * 100
ond = ond.join(oni_ond, on="year").dropna(subset=["oni"])
ond["phase"] = ond.oni.apply(phase)

# heavy rain days: above each sub-county's 95th percentile of wet days
p95 = (base[base.rain >= 1].groupby(["county", "subcounty"]).rain
       .quantile(0.95).rename("p95"))
heavy = (full[full.month.isin(SEASONS["OND"])]
         .join(p95, on=["county", "subcounty"])
         .assign(heavy=lambda d: d.rain > d.p95)
         .groupby(["county", "subcounty", "year"]).heavy.sum())
ond = ond.join(heavy, on=["county", "subcounty", "year"])
ond = ond.merge(onsets[onsets.season == "short_rains"]
                [["county", "subcounty", "year", "max_dry_spell"]],
                on=["county", "subcounty", "year"])
ond.to_csv(OUT / "ond_enso.csv", index=False)

order = ["La Niña", "Neutral", "El Niño"]
print(ond.groupby(["county", "phase"])[["anomaly_pct", "heavy", "max_dry_spell"]]
      .mean().round(1).unstack("phase").reindex(columns=order, level=1))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
sns.boxplot(data=ond, x="phase", y="anomaly_pct", hue="county",
            order=order, ax=axes[0])
axes[0].axhline(0, color="grey", lw=1)
axes[0].set_title("OND rainfall anomaly by ENSO phase (%)")
sns.boxplot(data=ond, x="phase", y="heavy", hue="county",
            order=order, ax=axes[1])
axes[1].set_title("OND heavy rain days by ENSO phase")
savefig("05_enso_ond")

# %%
county_ond = ond.groupby(["county", "year"]).agg(
    anomaly_pct=("anomaly_pct", "mean"), oni=("oni", "first")).reset_index()
for county, g in county_ond.groupby("county"):
    tau, p = kendalltau(g.oni, g.anomaly_pct)
    print(f"{county}: ONI vs OND anomaly, tau = {tau:.2f}, p = {p:.3f}")

plt.figure(figsize=(8, 5))
sns.scatterplot(data=county_ond, x="oni", y="anomaly_pct", hue="county")
for _, r in county_ond[county_ond.oni >= 1.5].drop_duplicates("year").iterrows():
    plt.annotate(int(r.year), (r.oni, r.anomaly_pct), fontsize=8)
plt.axhline(0, color="grey", lw=1)
plt.axvline(0.5, color="grey", lw=1, ls="--")
plt.xlabel("ONI (OND)")
plt.ylabel("OND rainfall anomaly (%)")
plt.title("Stronger El Niño, wetter short rains?")
savefig("06_oni_vs_ond")

# %%
strong = county_ond[county_ond.oni >= 1.5]
print("Strong El Niño years (ONI >= 1.5), OND rainfall anomaly %:")
if strong.empty:
    print("  none in the data")
else:
    print(strong.pivot_table(index="year", columns="county",
                             values="anomaly_pct").round(0))


# %%
