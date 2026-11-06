#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "arviz",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "pymc",
# ]
# ///
"""AMD vs Intel server power: Bayesian analysis + management-ready charts.

Input : power.csv with columns `amd`, `intel` (watts), rows time-matched.
Output: PNG per chart in ./charts/ and one PDF: management_report.pdf
Run   : uv run bayes_power_report.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeAlias

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    FloatArray: TypeAlias = NDArray[np.float64]
    BoolArray: TypeAlias = NDArray[np.bool_]
    IntArray: TypeAlias = NDArray[np.intp]
    ModelResult: TypeAlias = tuple[pm.Model, az.InferenceData]
    SensitivityRow: TypeAlias = tuple[str, float, float, float, float]
    FitFn: TypeAlias = Callable[..., ModelResult]

# ============================ CONFIG (edit me) ============================
CSV: Final[str] = "power.csv"
PAIRED: Final[bool] = (
    True  # True if row t of AMD and Intel were measured simultaneously
)
N_SERVERS: Final[int] = 1000  # fleet size to extrapolate to
PRICE_KWH: Final[float] = 0.12  # electricity price, $ per kWh
PUE: Final[float] = 1.4  # data-center overhead (cooling etc.)
ROPE: Final[float] = 0.05  # "meaningful" savings threshold (5%)
HIGH_SAVINGS: Final[float] = 0.10  # secondary savings threshold (10%)
CONF_HIGH: Final[float] = 0.95  # high-confidence probability threshold (95%)
CONF_MED: Final[float] = 0.80  # moderate-confidence probability threshold (80%)
PRIMARY_LINE_COUNT: Final[int] = 7
# ==========================================================================

AMD_C: Final[str] = "#ED1C24"
INTEL_C: Final[str] = "#0071C5"
OK_C: Final[str] = "#2E8B57"
GREY: Final[str] = "#666666"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
    }
)
Path("charts").mkdir(parents=True, exist_ok=True)
FIGS: list[Figure] = []


def _fmt_pct(x: float, _pos: int | None = None) -> str:
    return f"{x:.0%}"


def _fmt_usd_k(x: float, _pos: int | None = None) -> str:
    return f"${x / 1000:,.0f}k"


def title(ax: Axes, main: str, sub: str | None = None) -> None:
    ax.set_title(
        main,
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=34 if sub else 12,
    )
    if sub:
        ax.text(0, 1.03, sub, transform=ax.transAxes, fontsize=12, color=GREY)


def keep(fig: Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"charts/{name}.png", dpi=200, bbox_inches="tight")
    FIGS.append(fig)


# ------------------------------- data -----------------------------------
df: pd.DataFrame = pd.read_csv(CSV).dropna()
amd: FloatArray = df["amd"].to_numpy(dtype=np.float64)
intel: FloatArray = df["intel"].to_numpy(dtype=np.float64)
n: int = len(df)
la: FloatArray = np.log(amd)
li: FloatArray = np.log(intel)
pooled: FloatArray = np.concatenate([la, li])
MU0: float = float(pooled.mean())
SD0: float = float(pooled.std())


# ------------------------------- models ---------------------------------
def fit_independent(
    prior_sd: float = 0.5,
    seed: int = 1,
    draws: int = 2000,
) -> ModelResult:
    with pm.Model() as m:
        mu_i = pm.Normal("mu_intel", MU0, 1.0)
        delta = pm.Normal("delta", 0.0, prior_sd)
        sig_a = pm.HalfNormal("sigma_amd", 2 * SD0)
        sig_i = pm.HalfNormal("sigma_intel", 2 * SD0)
        nu = pm.Gamma("nu", 2, 0.1)
        pm.StudentT("y_amd", nu, mu_i + delta, sig_a, observed=la)
        pm.StudentT("y_intel", nu, mu_i, sig_i, observed=li)
        pm.Deterministic("savings", 1 - pm.math.exp(delta))
        idata: az.InferenceData = pm.sample(
            draws,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )
    return m, idata


def fit_paired(
    prior_sd: float = 0.5,
    seed: int = 1,
    draws: int = 2000,
) -> ModelResult:
    d: FloatArray = la - li
    with pm.Model() as m:
        delta = pm.Normal("delta", 0.0, prior_sd)
        sigma = pm.HalfNormal("sigma", 2 * float(d.std()))
        nu = pm.Gamma("nu", 2, 0.1)
        pm.StudentT("d", nu, delta, sigma, observed=d)
        pm.Deterministic("savings", 1 - pm.math.exp(delta))
        idata: az.InferenceData = pm.sample(
            draws,
            tune=2000,
            chains=4,
            target_accept=0.95,
            random_seed=seed,
            progressbar=False,
        )
    return m, idata


fit: FitFn = fit_paired if PAIRED else fit_independent
m_ind: pm.Model
id_ind: az.InferenceData
m_ind, id_ind = fit_independent()

m_main: pm.Model
id_main: az.InferenceData
m_main, id_main = fit_paired() if PAIRED else (m_ind, id_ind)

# --------------------------- diagnostics --------------------------------
for name, idt in [("independent", id_ind), ("headline", id_main)]:
    print(f"\n[{name}] diagnostics (want r_hat<1.01, ess_bulk>400, 0 divergences)")
    print(az.summary(idt, var_names=["delta", "nu"])[["mean", "r_hat", "ess_bulk"]])
    print("Divergences:", int(idt.sample_stats["diverging"].sum()))

# --------------------------- key numbers --------------------------------
s: FloatArray = id_main.posterior["savings"].to_numpy().ravel()
hdi: FloatArray = np.asarray(az.hdi(s, hdi_prob=CONF_HIGH), dtype=np.float64)
p_less: float = float((s > 0).mean())
p_rope: float = float((s > ROPE).mean())
p_10: float = float((s > HIGH_SAVINGS).mean())
model_label: str = "paired" if PAIRED else "independent-groups"

avg_intel_w: float = float(intel.mean())
usd: FloatArray = s * avg_intel_w * 8760 / 1000 * N_SERVERS * PUE * PRICE_KWH
usd_bounds: FloatArray = np.asarray(np.percentile(usd, [2.5, 97.5]), dtype=np.float64)
usd_lo: float = float(usd_bounds[0])
usd_hi: float = float(usd_bounds[1])

# ------------------ sensitivity to prior assumptions --------------------
sens: list[SensitivityRow] = []
prior_specs: list[tuple[float, str]] = [
    (0.02, "Very skeptical\n(assume ~no difference)"),
    (0.05, "Skeptical"),
    (0.2, "Moderate"),
    (0.5, "Open-minded\n(default)"),
    (1.0, "Very open"),
]
for sd, lab in prior_specs:
    _, idt = fit(prior_sd=sd, draws=1000)
    ss: FloatArray = idt.posterior["savings"].to_numpy().ravel()
    h: FloatArray = np.asarray(az.hdi(ss, hdi_prob=CONF_HIGH), dtype=np.float64)
    sens.append(
        (lab, float(ss.mean()), float(h[0]), float(h[1]), float((ss > 0).mean()))
    )
robust: bool = all(r[4] > CONF_HIGH for r in sens)

# ================================ CHARTS =================================
# 0. Executive summary page
fig: Figure = plt.figure(figsize=(11, 6.2))
fig.text(
    0.05,
    0.92,
    "Executive Summary: AMD vs Intel Server Power",
    fontsize=22,
    fontweight="bold",
)
verdict: str = (
    "AMD servers use LESS power"
    if p_less > CONF_HIGH
    else "Evidence is NOT conclusive that AMD uses less power"
)
fig.text(
    0.05,
    0.82,
    verdict,
    fontsize=18,
    color=OK_C if p_less > CONF_HIGH else "#B8860B",
    fontweight="bold",
)
robust_msg: str = "YES" if robust else "NO, see sensitivity chart"
hdi_str: str = f"{hdi[0]:.1%} to {hdi[1]:.1%}"
usd_str: str = f"${usd_lo:,.0f} to ${usd_hi:,.0f}"
lines: list[str] = [
    f"Expected power savings:  {s.mean():.1%}   (95% credible range: {hdi_str})",
    f"Probability AMD uses less power:  {p_less:.1%}",
    f"Probability savings exceed {ROPE:.0%}:  {p_rope:.1%}",
    f"Average draw:  AMD {amd.mean():.0f} W   vs   Intel {intel.mean():.0f} W",
    f"Estimated annual savings for {N_SERVERS:,} servers:  ${usd.mean():,.0f}",
    f"      (95% range: {usd_str}; incl. cooling overhead PUE {PUE})",
    f"Conclusion robust to skeptical assumptions:  {robust_msg}",
    "",
    f"Data: {n:,} time-matched measurements/server, same workload, memory & disk.",
    f"Method: Bayesian {model_label} model, outlier-robust (Student-T) likelihood.",
]
for i, t in enumerate(lines):
    fig.text(
        0.05,
        0.70 - i * 0.065,
        t,
        fontsize=14 if i < PRIMARY_LINE_COUNT else 11,
        color="black" if i < PRIMARY_LINE_COUNT else GREY,
    )
keep(fig, "0_executive_summary")

# 1. Raw data over time
ax: Axes
fig, ax = plt.subplots(figsize=(11, 5.5))
w: int = max(5, n // 50)
ax.plot(amd, color=AMD_C, alpha=0.15, lw=0.8)
ax.plot(intel, color=INTEL_C, alpha=0.15, lw=0.8)
ax.plot(
    pd.Series(amd).rolling(w, min_periods=1).mean(),
    color=AMD_C,
    lw=2.5,
    label="AMD",
)
ax.plot(
    pd.Series(intel).rolling(w, min_periods=1).mean(),
    color=INTEL_C,
    lw=2.5,
    label="Intel",
)
ax.set_xlabel("Measurement number (time order)")
ax.set_ylabel("Power draw (Watts)")
ax.legend(frameon=False)
title(
    ax,
    "Power draw over time under the same workload",
    f"AMD averages {amd.mean():.0f} W vs Intel {intel.mean():.0f} W "
    "(bold lines = rolling average)",
)
keep(fig, "1_power_over_time")

# 2. Distribution of power
fig, ax = plt.subplots(figsize=(11, 5.5))
bins: FloatArray = np.linspace(
    float(min(amd.min(), intel.min())),
    float(max(amd.max(), intel.max())),
    50,
)
ax.hist(amd, bins, alpha=0.6, color=AMD_C, label="AMD", density=True)
ax.hist(intel, bins, alpha=0.6, color=INTEL_C, label="Intel", density=True)
ax.axvline(float(amd.mean()), color=AMD_C, lw=2, ls="--")
ax.axvline(float(intel.mean()), color=INTEL_C, lw=2, ls="--")
ax.set_xlabel("Power draw (Watts)  |  dashed line = average")
ax.set_ylabel("Share of measurements")
ax.set_yticks([])
ax.legend(frameon=False)
title(
    ax,
    "How much power does each server type typically use?",
    f"Typical AMD server is {intel.mean() - amd.mean():.0f} W lower than Intel",
)
keep(fig, "2_power_distribution")

# 3. Posterior of savings
fig, ax = plt.subplots(figsize=(11, 5.5))
counts: FloatArray
edges: FloatArray
counts, edges = np.histogram(s, bins=100, density=True)
mids: FloatArray = (edges[:-1] + edges[1:]) / 2
width: float = float(edges[1] - edges[0])
inside: BoolArray = (mids >= hdi[0]) & (mids <= hdi[1])
ax.bar(mids[inside], counts[inside], width, color=OK_C, label="95% most likely range")
ax.bar(mids[~inside], counts[~inside], width, color="#BBBBBB")
ax.axvline(0, color="black", lw=2)
ax.axvline(ROPE, color="#B8860B", lw=2, ls="--")
ax.text(0, ax.get_ylim()[1] * 0.97, " No difference", ha="left", va="top", fontsize=10)
ax.text(
    ROPE,
    ax.get_ylim()[1] * 0.90,
    f" {ROPE:.0%} = meaningful",
    ha="left",
    va="top",
    fontsize=10,
    color="#B8860B",
)
ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
ax.set_xlabel("Power saved by choosing AMD over Intel")
ax.set_yticks([])
ax.legend(frameon=False, loc="upper right")
title(
    ax,
    f"AMD saves about {s.mean():.0%} power",
    f"Most likely range: {hdi[0]:.1%} to {hdi[1]:.1%}  "
    "(everything right of the black line = AMD better)",
)
keep(fig, "3_savings_estimate")

# 4. Confidence bars
fig, ax = plt.subplots(figsize=(11, 4.5))
labels: list[str] = [
    "AMD uses less power\n(any amount)",
    f"AMD saves more than {ROPE:.0%}",
    "AMD saves more than 10%",
]
probs: list[float] = [p_less, p_rope, p_10]
bar_colors: list[str] = [
    OK_C if p > CONF_HIGH else "#B8860B" if p > CONF_MED else "#999999"
    for p in probs[::-1]
]
bars: BarContainer = ax.barh(
    labels[::-1],
    np.array(probs[::-1], dtype=np.float64) * 100,
    color=bar_colors,
)
for b, p in zip(bars, probs[::-1], strict=True):
    ax.text(
        min(p * 100 + 1, 88),
        b.get_y() + b.get_height() / 2,
        f"{p:.0%}",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
ax.set_xlim(0, 105)
ax.set_xlabel("Probability (%)")
title(
    ax,
    "How confident are we that AMD is more efficient?",
    "Green = very confident (>95%), gold = fairly likely, grey = uncertain",
)
keep(fig, "4_confidence")

# 5. Sensitivity to assumptions
fig, ax = plt.subplots(figsize=(11, 5.5))
y: IntArray = np.arange(len(sens))[::-1]
for yi, (_lab, mean, lo, hi, p) in zip(y, sens, strict=True):
    ax.plot([lo, hi], [yi, yi], color=OK_C, lw=6, alpha=0.5, solid_capstyle="round")
    ax.plot(mean, yi, "o", color=OK_C, ms=10)
    ax.text(hi + 0.003, int(yi), f"  P(AMD better) = {p:.0%}", va="center", fontsize=11)
ax.axvline(0, color="black", lw=2)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in sens])
ax.xaxis.set_major_formatter(FuncFormatter(_fmt_pct))
ax.set_xlabel("Estimated power savings with AMD  (dot = average, bar = 95% range)")
title(
    ax,
    "Conclusion holds even under skeptical assumptions"
    if robust
    else "Conclusion depends on starting assumptions: interpret with care",
    "We re-ran the analysis assuming progressively less belief that AMD is better",
)
keep(fig, "5_robustness_check")

# 6. Money
fig, ax = plt.subplots(figsize=(11, 5.5))
c: FloatArray
e: FloatArray
c, e = np.histogram(usd, bins=100, density=True)
mids = (e[:-1] + e[1:]) / 2
ins: BoolArray = (mids >= usd_lo) & (mids <= usd_hi)
bin_w: float = float(e[1] - e[0])
ax.bar(mids[ins], c[ins], bin_w, color=OK_C)
ax.bar(mids[~ins], c[~ins], bin_w, color="#BBBBBB")
ax.axvline(0, color="black", lw=2)
ax.xaxis.set_major_formatter(FuncFormatter(_fmt_usd_k))
ax.set_xlabel("Estimated annual electricity savings (USD)  |  green = 95% range")
ax.set_yticks([])
title(
    ax,
    f"Estimated savings: about ${usd.mean():,.0f} per year",
    f"Fleet of {N_SERVERS:,} servers, ${PRICE_KWH}/kWh, cooling overhead (PUE) {PUE}",
)
keep(fig, "6_annual_savings")

# 7. Appendix: model check
with m_ind:
    ppc: az.InferenceData = pm.sample_posterior_predictive(
        id_ind,
        random_seed=1,
        progressbar=False,
    )
axes: NDArray[np.object_]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
ppc_panels: list[tuple[Axes, str, FloatArray, str, str]] = [
    (axes[0], "y_amd", amd, AMD_C, "AMD"),
    (axes[1], "y_intel", intel, INTEL_C, "Intel"),
]
for ax, key, obs, col, nm in ppc_panels:
    sims: FloatArray = np.exp(ppc.posterior_predictive[key].to_numpy().reshape(-1, n))
    b_edges: FloatArray = np.linspace(
        float(np.percentile(obs, 0.5)),
        float(np.percentile(obs, 99.5)),
        40,
    )
    sample_idx: IntArray = np.random.default_rng(1).choice(
        len(sims),
        60,
        replace=False,
    )
    for k in sample_idx:
        ax.hist(
            sims[k],
            b_edges,
            histtype="step",
            density=True,
            color="#999999",
            alpha=0.25,
        )
    ax.hist(
        obs,
        b_edges,
        histtype="step",
        density=True,
        color=col,
        lw=3,
        label=f"Real {nm} data",
    )
    ax.set_xlabel("Power draw (Watts)")
    ax.set_yticks([])
    ax.legend(frameon=False)
fig.suptitle(
    "Appendix: does the statistical model match reality?",
    x=0.01,
    ha="left",
    fontsize=16,
    fontweight="bold",
)
fig.text(
    0.01,
    0.90,
    "Colored line (real data) sitting inside the grey cloud "
    "(model simulations) = good fit",
    fontsize=12,
    color=GREY,
)
fig.subplots_adjust(top=0.78)
FIGS.append(fig)
fig.savefig("charts/7_appendix_model_check.png", dpi=200, bbox_inches="tight")

# ------------------------------ PDF + text --------------------------------
with PdfPages("management_report.pdf") as pdf:
    for f in FIGS:
        pdf.savefig(f, bbox_inches="tight")

print(
    f"\nHeadline ({model_label} model): savings {s.mean():.1%} "
    f"[{hdi[0]:.1%}, {hdi[1]:.1%}], P(AMD<Intel)={p_less:.1%}, "
    f"P(>{ROPE:.0%})={p_rope:.1%}"
)
print(f"Annual savings: ${usd.mean():,.0f}  (${usd_lo:,.0f} to ${usd_hi:,.0f})")
print("Saved: charts/*.png and management_report.pdf")
plt.show()
