#=
AMD vs Intel server power: Bayesian analysis (Turing.jl) + management-ready charts.
Input : power.csv with columns `amd`, `intel` (watts), rows time-matched.
Output: PNG per chart in ./charts/ and management_report.pdf (all charts on one tall page)

Setup (once):
    ] add CSV DataFrames Statistics StatsBase Turing MCMCChains Distributions Plots
Run (4 threads so the 4 chains run in parallel):
    julia -t 4 bayes_power_report.jl
=#
using CSV, DataFrames, Statistics, StatsBase, Random, Printf
using Turing, MCMCChains, Distributions
using Plots, Plots.PlotMeasures

# ============================ CONFIG (edit me) ============================
const CSV_PATH  = "power.csv"
const PAIRED    = true      # true if row t of AMD and Intel were measured simultaneously
const N_SERVERS = 1000      # fleet size to extrapolate to
const PRICE_KWH = 0.12      # electricity price, $ per kWh
const PUE       = 1.4       # data-center overhead (cooling etc.)
const ROPE      = 0.05      # "meaningful" savings threshold (5%)
# ==========================================================================

Random.seed!(1)
const AMD_C, INTEL_C, OK_C, GOLD, GREY = "#ED1C24", "#0071C5", "#2E8B57", "#B8860B", "#666666"
default(size = (1100, 550), titlelocation = :left, titlefontsize = 15,
        guidefontsize = 11, tickfontsize = 10, legendfontsize = 11,
        left_margin = 8mm, bottom_margin = 6mm, top_margin = 4mm, framestyle = :axes)
mkpath("charts")
const FIGS = Any[]
keep!(p, name) = (savefig(p, "charts/$name.png"); push!(FIGS, p); p)

# ---------------------------- helpers -----------------------------------
pct(x; d = 1) = string(round(100x; digits = d), "%")
commas(x) = replace(string(round(Int, x)), r"(?<=[0-9])(?=(?:[0-9]{3})+$)" => ",")
pctfmt = x -> string(round(Int, 100x), "%")

function hdi(x, p = 0.95)                      # shortest interval holding p of the draws
    xs = sort(x); n = length(xs); m = ceil(Int, p * n)
    w = xs[m:end] .- xs[1:(n - m + 1)]
    i = argmin(w)
    return xs[i], xs[i + m - 1]
end

rolling(x, w) = [mean(x[max(1, i - w + 1):i]) for i in eachindex(x)]
draws(ch, s) = vec(Array(ch[s]))

# ------------------------------- data -----------------------------------
df = dropmissing(CSV.read(CSV_PATH, DataFrame))
amd, intel = Float64.(df.amd), Float64.(df.intel)
n = length(amd)
la, li = log.(amd), log.(intel)
pooled = vcat(la, li)
MU0, SD0 = mean(pooled), std(pooled)

# ------------------------------- models ---------------------------------
# delta = log(AMD power / Intel power); savings = 1 - exp(delta)
@model function indep_model(la, li, mu0, sd0, prior_sd)
    mu_i    ~ Normal(mu0, 1.0)
    delta   ~ Normal(0.0, prior_sd)
    sigma_a ~ truncated(Normal(0, 2sd0); lower = 0)
    sigma_i ~ truncated(Normal(0, 2sd0); lower = 0)
    nu      ~ Gamma(2, 10)                       # shape 2, rate 0.1 (scale 10)
    la ~ filldist(LocationScale(mu_i + delta, sigma_a, TDist(nu)), length(la))
    li ~ filldist(LocationScale(mu_i,         sigma_i, TDist(nu)), length(li))
end

@model function paired_model(d, sd_d, prior_sd)
    delta ~ Normal(0.0, prior_sd)
    sigma ~ truncated(Normal(0, 2sd_d); lower = 0)
    nu    ~ Gamma(2, 10)
    d ~ filldist(LocationScale(delta, sigma, TDist(nu)), length(d))
end

run_chain(model; ndraws = 2000) =
    sample(model, NUTS(1000, 0.95), MCMCThreads(), ndraws, 4;
           progress = false, chain_type = MCMCChains.Chains)  # Turing >= 0.45 defaults to FlexiChains

fit_independent(; prior_sd = 0.5, ndraws = 2000) =
    run_chain(indep_model(la, li, MU0, SD0, prior_sd); ndraws)
fit_paired(; prior_sd = 0.5, ndraws = 2000) =
    run_chain(paired_model(la .- li, std(la .- li), prior_sd); ndraws)
fit_main(; kw...) = PAIRED ? fit_paired(; kw...) : fit_independent(; kw...)

ch_ind  = fit_independent()
ch_main = PAIRED ? fit_paired() : ch_ind

# --------------------------- diagnostics --------------------------------
for (name, ch) in [("independent", ch_ind), ("headline", ch_main)]
    println("\n[$name] diagnostics (want rhat < 1.01, ess_bulk > 400, 0 divergences)")
    show(summarystats(ch)); println()
    println("Divergent transitions: ", Int(sum(ch[:numerical_error])))
end

# --------------------------- key numbers --------------------------------
s = 1 .- exp.(draws(ch_main, :delta))
lo, hi = hdi(s)
p_less, p_rope, p_10 = mean(s .> 0), mean(s .> ROPE), mean(s .> 0.10)
model_label = PAIRED ? "paired" : "independent-groups"

usd = s .* mean(intel) .* 8760 / 1000 .* N_SERVERS .* PUE .* PRICE_KWH
usd_lo, usd_hi = hdi(usd)

# ------------------ sensitivity to prior assumptions --------------------
sens = []
for (sd, lab) in [(0.02, "Very skeptical (assume ~no difference)"), (0.05, "Skeptical"),
                  (0.2, "Moderate"), (0.5, "Open-minded (default)"), (1.0, "Very open")]
    ss = 1 .- exp.(draws(fit_main(; prior_sd = sd, ndraws = 1000), :delta))
    l, h = hdi(ss)
    push!(sens, (lab = lab, m = mean(ss), lo = l, hi = h, p = mean(ss .> 0)))
end
robust = all(r.p > 0.95 for r in sens)

# ================================ CHARTS =================================
# 0. Executive summary
verdict = p_less > 0.95 ? "AMD servers use LESS power" :
                          "Evidence is NOT conclusive that AMD uses less power"
lines = [
    "Expected power savings:  $(pct(mean(s)))   (95% credible range: $(pct(lo)) to $(pct(hi)))",
    "Probability AMD uses less power:  $(pct(p_less))",
    "Probability savings exceed $(pct(ROPE; d = 0)):  $(pct(p_rope))",
    "Average draw:  AMD $(round(Int, mean(amd))) W   vs   Intel $(round(Int, mean(intel))) W",
    "Estimated annual savings for $(commas(N_SERVERS)) servers:  \$$(commas(mean(usd)))",
    "      (95% range: \$$(commas(usd_lo)) to \$$(commas(usd_hi)); incl. cooling overhead PUE $PUE)",
    "Conclusion robust to skeptical assumptions:  $(robust ? "YES" : "NO, see robustness chart")",
]
foot = [
    "Data: $(commas(n)) time-matched measurements per server type, same workload, memory and disk.",
    "Method: Bayesian $model_label model (Turing.jl, NUTS), outlier-robust Student-T likelihood.",
]
p0 = plot(; xlims = (0, 1), ylims = (0, 1), framestyle = :none, legend = false, size = (1100, 620))
annotate!(p0, 0.02, 0.94, text("Executive Summary: AMD vs Intel Server Power", :left, 22, :bold))
annotate!(p0, 0.02, 0.83, text(verdict, :left, 18, :bold, p_less > 0.95 ? OK_C : GOLD))
for (i, t) in enumerate(lines)
    annotate!(p0, 0.02, 0.72 - (i - 1) * 0.08, text(t, :left, 13))
end
for (i, t) in enumerate(foot)
    annotate!(p0, 0.02, 0.10 - (i - 1) * 0.06, text(t, :left, 10, GREY))
end
keep!(p0, "0_executive_summary")

# 1. Raw data over time
w = max(5, n ÷ 50)
p1 = plot(amd;   color = AMD_C,   alpha = 0.15, label = "", lw = 1)
plot!(p1, intel; color = INTEL_C, alpha = 0.15, label = "", lw = 1)
plot!(p1, rolling(amd, w);   color = AMD_C,   lw = 3, label = "AMD")
plot!(p1, rolling(intel, w); color = INTEL_C, lw = 3, label = "Intel")
plot!(p1; xlabel = "Measurement number (time order)", ylabel = "Power draw (Watts)",
      title = "Power draw over time under the same workload\nAMD averages $(round(Int, mean(amd))) W vs Intel $(round(Int, mean(intel))) W (bold lines = rolling average)")
keep!(p1, "1_power_over_time")

# 2. Distribution of power
bins = range(min(minimum(amd), minimum(intel)), max(maximum(amd), maximum(intel)); length = 50)
p2 = histogram(amd;   bins, normalize = :pdf, alpha = 0.6, color = AMD_C,   label = "AMD")
histogram!(p2, intel; bins, normalize = :pdf, alpha = 0.6, color = INTEL_C, label = "Intel")
vline!(p2, [mean(amd)];   color = AMD_C,   lw = 2, ls = :dash, label = "")
vline!(p2, [mean(intel)]; color = INTEL_C, lw = 2, ls = :dash, label = "")
plot!(p2; xlabel = "Power draw (Watts)   |   dashed line = average", yticks = false,
      ylabel = "Share of measurements",
      title = "How much power does each server type typically use?\nTypical AMD server is $(round(Int, mean(intel) - mean(amd))) W lower than Intel")
keep!(p2, "2_power_distribution")

# 3. Posterior of savings
h = fit(Histogram, s; nbins = 100); h = normalize(h; mode = :pdf)
edges = collect(h.edges[1]); mids = (edges[1:end-1] .+ edges[2:end]) ./ 2
wd = edges[2] - edges[1]
inside = (mids .>= lo) .& (mids .<= hi)
p3 = bar(mids, h.weights .* inside; bar_width = wd, color = OK_C, linecolor = OK_C, label = "95% most likely range")
bar!(p3, mids, h.weights .* .!inside; bar_width = wd, color = "#BBBBBB", linecolor = "#BBBBBB", label = "")
vline!(p3, [0.0];  color = :black, lw = 2, label = "No difference")
vline!(p3, [ROPE]; color = GOLD,   lw = 2, ls = :dash, label = "$(pct(ROPE; d = 0)) = meaningful")
plot!(p3; xformatter = pctfmt, xlabel = "Power saved by choosing AMD over Intel", yticks = false,
      title = "AMD saves about $(pct(mean(s); d = 0)) power\nMost likely range: $(pct(lo)) to $(pct(hi))  (right of black line = AMD better)")
keep!(p3, "3_savings_estimate")

# 4. Confidence bars
labels = ["AMD uses less power (any amount)", "AMD saves more than $(pct(ROPE; d = 0))", "AMD saves more than 10%"]
probs  = [p_less, p_rope, p_10]
p4 = plot(; legend = false, xlims = (0, 110), size = (1100, 450))
for (i, p) in enumerate(reverse(probs))
    c = p > 0.95 ? OK_C : p > 0.8 ? GOLD : "#999999"
    bar!(p4, [i], [100p]; orientation = :h, color = c, linecolor = c, bar_width = 0.6)
    annotate!(p4, min(100p + 2, 92), i, text(pct(p; d = 0), :left, 14, :bold))
end
plot!(p4; yticks = (1:3, reverse(labels)), xlabel = "Probability (%)",
      title = "How confident are we that AMD is more efficient?\nGreen = very confident (>95%), gold = fairly likely, grey = uncertain")
keep!(p4, "4_confidence")

# 5. Sensitivity to assumptions
ys = collect(length(sens):-1:1)
ms, ls_, hs = [r.m for r in sens], [r.lo for r in sens], [r.hi for r in sens]
p5 = scatter(ms, ys; xerror = (ms .- ls_, hs .- ms), color = OK_C, markersize = 8,
             markerstrokecolor = OK_C, label = "", lw = 3)
vline!(p5, [0.0]; color = :black, lw = 2, label = "")
for (y, r) in zip(ys, sens)
    annotate!(p5, r.hi + 0.004, y, text("  P(AMD better) = $(pct(r.p; d = 0))", :left, 10))
end
plot!(p5; yticks = (ys, [r.lab for r in sens]), xformatter = pctfmt, left_margin = 28mm,
      xlims = (minimum(ls_) - 0.01, maximum(hs) + 0.12),
      xlabel = "Estimated power savings with AMD (dot = average, bar = 95% range)",
      title = (robust ? "Conclusion holds even under skeptical assumptions" :
                        "Conclusion depends on starting assumptions: interpret with care") *
              "\nWe re-ran the analysis assuming progressively less belief that AMD is better")
keep!(p5, "5_robustness_check")

# 6. Money
hu = fit(Histogram, usd; nbins = 100); hu = normalize(hu; mode = :pdf)
eu = collect(hu.edges[1]); mu_ = (eu[1:end-1] .+ eu[2:end]) ./ 2; wu = eu[2] - eu[1]
ins = (mu_ .>= usd_lo) .& (mu_ .<= usd_hi)
p6 = bar(mu_, hu.weights .* ins; bar_width = wu, color = OK_C, linecolor = OK_C, label = "95% range")
bar!(p6, mu_, hu.weights .* .!ins; bar_width = wu, color = "#BBBBBB", linecolor = "#BBBBBB", label = "")
vline!(p6, [0.0]; color = :black, lw = 2, label = "")
plot!(p6; xformatter = x -> "\$" * string(round(Int, x / 1000)) * "k", yticks = false,
      xlabel = "Estimated annual electricity savings (USD)",
      title = "Estimated savings: about \$$(commas(mean(usd))) per year\nFleet of $(commas(N_SERVERS)) servers, \$$PRICE_KWH/kWh, cooling overhead (PUE) $PUE")
keep!(p6, "6_annual_savings")

# 7. Appendix: model check (posterior predictive, independent model)
mu_i, dl = draws(ch_ind, :mu_i), draws(ch_ind, :delta)
sa, si, nu = draws(ch_ind, :sigma_a), draws(ch_ind, :sigma_i), draws(ch_ind, :nu)
idx = rand(1:length(mu_i), 60)
rng_b(x) = range(quantile(x, 0.005), quantile(x, 0.995); length = 40)
pa = plot(; legend = :topright, yticks = false, xlabel = "Power draw (Watts)",
          title = "Appendix: does the model match reality?\nColored line = real data; grey cloud = model simulations", size = (1100, 480))
pi_ = plot(; legend = :topright, yticks = false, xlabel = "Power draw (Watts)", title = "")
for k in idx
    simA = exp.(mu_i[k] + dl[k] .+ sa[k] .* rand(TDist(nu[k]), n))
    simI = exp.(mu_i[k] .+ si[k] .* rand(TDist(nu[k]), n))
    stephist!(pa, simA; bins = rng_b(amd),   normalize = :pdf, color = "#999999", alpha = 0.25, label = "")
    stephist!(pi_, simI; bins = rng_b(intel), normalize = :pdf, color = "#999999", alpha = 0.25, label = "")
end
stephist!(pa, amd;    bins = rng_b(amd),   normalize = :pdf, color = AMD_C,   lw = 3, label = "Real AMD data")
stephist!(pi_, intel; bins = rng_b(intel), normalize = :pdf, color = INTEL_C, lw = 3, label = "Real Intel data")
p7 = plot(pa, pi_; layout = (1, 2), size = (1100, 480))
keep!(p7, "7_appendix_model_check")

# ------------------------------ PDF + text --------------------------------
report = plot(FIGS...; layout = (length(FIGS), 1), size = (1100, 600 * length(FIGS)))
savefig(report, "management_report.pdf")

println("\nHeadline ($model_label model): savings $(pct(mean(s))) [$(pct(lo)), $(pct(hi))], ",
        "P(AMD<Intel)=$(pct(p_less)), P(>$(pct(ROPE; d = 0)))=$(pct(p_rope))")
println("Annual savings: \$$(commas(mean(usd)))  (\$$(commas(usd_lo)) to \$$(commas(usd_hi)))")
println("Saved: charts/*.png and management_report.pdf")
