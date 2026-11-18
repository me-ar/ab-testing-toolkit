"""Full analysis of the Cookie Cats gate experiment (90,189 players).

Question: the first "gate" (forced wait / pay wall) was moved from level 30 to level 40.
Did that change player retention and engagement?

Usage: python analysis/cookie_cats.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import abkit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cookie_cats.csv"
URL = ("https://raw.githubusercontent.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats/"
       "master/cookie_cats.csv")
FIGS = ROOT / "reports" / "figures"
C_COL, T_COL = "#64748b", "#2563eb"


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def main() -> None:
    if not DATA.exists():
        DATA.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(URL, DATA)
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

    df = pd.read_csv(DATA)
    c, t = df[df.version == "gate_30"], df[df.version == "gate_40"]
    out: dict = {"n_control": len(c), "n_treatment": len(t)}

    # 1. Sanity: sample ratio mismatch
    srm_p, srm_flag = abkit.srm_check(len(c), len(t))
    out["srm_p_value"], out["srm_flag"] = srm_p, srm_flag

    # 2. Data quality: one player with ~50k rounds in 14 days is a clear logging outlier
    outlier = df.sum_gamerounds.idxmax()
    out["outlier_rounds_removed"] = int(df.loc[outlier, "sum_gamerounds"])
    clean = df.drop(index=outlier)
    c, t = clean[clean.version == "gate_30"], clean[clean.version == "gate_40"]

    # 3. Primary metrics: retention (binary) — frequentist + Bayesian
    results, pvals = {}, {}
    for m in ["retention_1", "retention_7"]:
        xc, xt = int(c[m].sum()), int(t[m].sum())
        r = abkit.two_proportion_ztest(xc, len(c), xt, len(t), metric=m)
        b = abkit.beta_binomial(xc, len(c), xt, len(t))
        boot = abkit.bootstrap_diff(c[m].to_numpy(float), t[m].to_numpy(float), n_boot=3000)
        results[m] = {**r.to_dict(),
                      "bayes_p_gate40_better": b["p_treatment_better"],
                      "bayes_lift_95": b["lift_95_hdi"],
                      "bayes_expected_loss_ship_gate40": b["expected_loss_ship_treatment"],
                      "bootstrap_ci": (boot["ci_low"], boot["ci_high"])}
        results[m]["_bayes"] = b["_samples"]
        results[m]["_boot"] = boot["samples"]
        pvals[m] = r.p_value

    # 4. Secondary metric: engagement (heavy-tailed) — t-test and rank test
    g = abkit.welch_ttest(c.sum_gamerounds, t.sum_gamerounds, metric="sum_gamerounds")
    mw = abkit.mann_whitney(c.sum_gamerounds, t.sum_gamerounds, metric="sum_gamerounds")
    results["sum_gamerounds"] = {**g.to_dict(), "mann_whitney_p": mw.p_value,
                                 "median_control": mw.control_mean, "median_treatment": mw.treatment_mean}
    pvals["sum_gamerounds_mw"] = mw.p_value
    out["holm_bonferroni_reject"] = abkit.holm_bonferroni(pvals)

    # 5. Design retrospective: was the test powered for the effect it found?
    base7 = results["retention_7"]["control_mean"]
    out["mde_rel_retention_7_at_80pct_power"] = abkit.minimum_detectable_effect(base7, len(c))
    out["n_needed_for_1pct_rel_lift_r7"] = abkit.sample_size_proportions(base7, 0.01)
    out["n_needed_for_2pct_rel_lift_r7"] = abkit.sample_size_proportions(base7, 0.02)

    # 6. Why you must not peek: A/A simulation at this experiment's scale
    peek = abkit.simulate_peeking(n_per_group=20_000, n_looks=20, p=base7, n_sims=2000)
    out["peeking"] = {k: peek[k] for k in ["fpr_fixed_horizon", "fpr_with_peeking"]}

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, ["retention_1", "retention_7"], strict=True):
        pc, pt = results[m]["_bayes"]
        ax.hist(pc * 100, bins=80, alpha=0.7, color=C_COL, label="gate_30 (control)", density=True)
        ax.hist(pt * 100, bins=80, alpha=0.7, color=T_COL, label="gate_40 (treatment)", density=True)
        ax.set(title=f"{m.replace('_', '-')} posterior  ·  P(gate_40 better) = "
                     f"{results[m]['bayes_p_gate40_better']:.3f}",
               xlabel="retention rate (%)", ylabel="density")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "retention_posteriors.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    d = results["retention_7"]["_boot"] * 100
    ax.hist(d, bins=60, color=T_COL, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    ax.axvspan(lo, hi, color=T_COL, alpha=0.12, label=f"95% CI [{lo:.2f}, {hi:.2f}] pp")
    ax.set(title="Bootstrap: change in 7-day retention (gate_40 − gate_30)",
           xlabel="percentage points", ylabel="resamples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "bootstrap_retention_7.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(peek["looks"], np.array(peek["fpr_by_look"]) * 100, "o-", color="#dc2626",
            label="stop at first p < 0.05")
    ax.axhline(5, ls="--", color="grey", label="nominal α = 5%")
    ax.set(title="A/A simulation: false-positive rate when peeking",
           xlabel="users per group at look", ylabel="cumulative false positives (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "peeking_simulation.png")

    mde_grid = np.linspace(0.005, 0.10, 40)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mde_grid * 100, [abkit.sample_size_proportions(base7, m) / 1000 for m in mde_grid],
            color=T_COL)
    ax.axhline(len(c) / 1000, ls="--", color="grey", label=f"actual n per group ({len(c):,})")
    ax.set(yscale="log", title="Sample size needed vs. detectable lift (7-day retention, 80% power)",
           xlabel="relative lift to detect (%)", ylabel="users per group (thousands, log)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "power_curve.png")

    # ---------------- save ----------------
    for m in ["retention_1", "retention_7"]:
        results[m].pop("_bayes")
        results[m].pop("_boot")
    out["results"] = results
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=2, default=float))

    r1, r7, gr = results["retention_1"], results["retention_7"], results["sum_gamerounds"]
    print(f"SRM p={srm_p:.3f} (flag={srm_flag})")
    for name, r in [("1-day", r1), ("7-day", r7)]:
        print(f"{name} retention: {pct(r['control_mean'])} -> {pct(r['treatment_mean'])} "
              f"({r['abs_diff'] * 100:+.2f} pp, {r['rel_lift'] * 100:+.1f}%), p={r['p_value']:.4f}, "
              f"CI=[{r['ci_low'] * 100:.2f}, {r['ci_high'] * 100:.2f}] pp, "
              f"P(gate40 better)={r['bayes_p_gate40_better']:.3f}")
    print(f"Game rounds: mean {gr['control_mean']:.1f} -> {gr['treatment_mean']:.1f}, "
          f"Welch p={gr['p_value']:.3f}, MW p={gr['mann_whitney_p']:.3f}")
    print(f"Holm: {out['holm_bonferroni_reject']}")
    print(f"MDE r7: {out['mde_rel_retention_7_at_80pct_power'] * 100:.1f}% rel; "
          f"n for 1%: {out['n_needed_for_1pct_rel_lift_r7']:,}")
    print(f"Peeking FPR: {out['peeking']}")


if __name__ == "__main__":
    main()
