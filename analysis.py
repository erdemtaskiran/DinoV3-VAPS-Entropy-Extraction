"""All statistics reported in the paper.

usage:  python analysis.py --vaps VAPS_Data_per_Artwork.csv
needs:  results/per_image_measures.csv (from extract.py)
writes: results/stats_*.csv
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, KFold
from statsmodels.stats.multitest import multipletests

SEED, N_BOOT = 20250902, 5000
MEAS = ["H", "C", "FuzzyEn", "LZC"]
DINO = ["H", "FuzzyEn", "LZC"]                     # C omitted in models: r(H, C) = -.997
COV = ["mean_luminance", "contrast", "saturation_mean", "colorfulness", "pad_fraction"]
RAT = {"complexity": "Visual_Complexity", "arousal": "Arousal",
       "valence": "Valence", "liking": "Liking"}
fdr = lambda p: multipletests(p, method="fdr_bh")[1]
zs = lambda D: (D - D.mean()) / D.std()


def load(vaps):
    v = pd.read_csv(vaps, sep=";", decimal=",", encoding="cp1252", dtype={"Picture_Number": str})
    m = pd.read_csv("results/per_image_measures.csv", dtype={"image_id": str})
    # rows in VAPS file order (fixes cross-validation folds and bootstrap resamples)
    d = v.rename(columns={"Picture_Number": "image_id"}).merge(m, on="image_id")
    for k, r in RAT.items():
        d[k] = d[f"All_{r}_M"]
        d[k + "_F"], d[k + "_M"] = d[f"Female_{r}_M"], d[f"Male_{r}_M"]
    d["familiarity"] = d["All_Familiarity_M"]
    d["grp"] = d["Category_Name"] + "|" + d["Epoch_Style"]
    d["in_grp"] = d["grp"].map(d["grp"].value_counts()) >= 10
    return d


def resid(D, cols, ctrl):
    X = np.column_stack([np.ones(len(D)), D[ctrl].values])
    return pd.DataFrame({c: D[c] - X @ np.linalg.lstsq(X, D[c].values, rcond=None)[0]
                         for c in cols})


def within(d, cols):
    g = d[d["in_grp"]]
    return g[cols] - g.groupby("grp")[cols].transform("mean")


def corr(x, y, df, n_eff):
    """r, p (t test with df) and Fisher-z 95% CI (n_eff)."""
    r = np.corrcoef(x, y)[0, 1]
    p = 2 * stats.t.sf(abs(r) * np.sqrt(df / (1 - r * r)), df)
    z, se = np.arctanh(r), 1 / np.sqrt(n_eff - 3)
    return r, p, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def correlations(d):
    n, G = len(d), d.loc[d.in_grp, "grp"].nunique()
    cols = MEAS + list(RAT)
    A, W = resid(d, cols, COV), within(d, cols)
    rows = []
    for m in MEAS:
        for o in RAT:
            rows += [("raw", m, o, *corr(d[m], d[o], n - 2, n), stats.spearmanr(d[m], d[o])[0]),
                     ("adjusted", m, o, *corr(A[m], A[o], n - 2 - len(COV), n - len(COV)), np.nan),
                     ("within", m, o, *corr(W[m], W[o], len(W) - G - 1, len(W) - G + 1), np.nan)]
    t = pd.DataFrame(rows, columns=["spec", "measure", "rating", "r", "p", "lo", "hi", "rho"])
    t["p_fdr"] = t.groupby("spec")["p"].transform(fdr)        # 16 tests per family
    return t


def tertiles(d, rating):
    q = pd.qcut(d[rating], 3, labels=False)
    rows = []
    for m in MEAS:
        z = zs(d[m])
        g = [z[q == i] for i in range(3)]
        F, p = stats.f_oneway(*g)
        eta = sum(len(x) * x.mean() ** 2 for x in g) / ((z - z.mean()) ** 2).sum()
        rows.append(dict(rating=rating, measure=m, F=F, p=p, eta2=eta,
                         p_kruskal=stats.kruskal(*g).pvalue, p_trend=stats.linregress(q, z).pvalue,
                         high_minus_low=g[2].mean() - g[0].mean()))
    return pd.DataFrame(rows), q.value_counts().sort_index().tolist()


def hierarchical(d):
    D = zs(d[COV + ["H_pixel"] + DINO + ["complexity"]])
    base = sm.OLS(D.complexity, sm.add_constant(D[COV + ["H_pixel"]])).fit()
    full = sm.OLS(D.complexity, sm.add_constant(D[COV + ["H_pixel"] + DINO])).fit()
    only = sm.OLS(D.complexity, sm.add_constant(D[DINO])).fit()
    F, p, _ = full.compare_f_test(base)
    print(f"R2 base {base.rsquared:.3f} (adj {base.rsquared_adj:.3f}) -> full {full.rsquared:.3f}"
          f" (adj {full.rsquared_adj:.3f}); dR2 = {full.rsquared - base.rsquared:.3f},"
          f" F(3, {int(full.df_resid)}) = {F:.2f}, p = {p:.2g}; DINOv3 only R2 = {only.rsquared:.3f}")
    ci = full.conf_int()
    for k in DINO + ["H_pixel"]:
        print(f"  beta {k:<8} {full.params[k]:+.3f} [{ci.loc[k, 0]:+.2f}, {ci.loc[k, 1]:+.2f}]"
              f" p = {full.pvalues[k]:.3g}")


def raters(d, rng):
    G = d.loc[d.in_grp, "grp"].nunique()
    cols = ["complexity_F", "complexity_M", "liking_F", "liking_M"] + MEAS
    W = within(d, cols)
    for a, b in [("F", "F"), ("M", "M"), ("F", "M"), ("M", "F")]:
        r0 = np.corrcoef(d["complexity_" + a], d["liking_" + b])[0, 1]
        r1 = np.corrcoef(W["complexity_" + a], W["liking_" + b])[0, 1]
        print(f"  complexity({a}) x liking({b}): pooled r = {r0:.3f}, within r = {r1:.3f}")
    for m in MEAS:
        print(f"  {m:<8} r with complexity: women {np.corrcoef(d[m], d.complexity_F)[0, 1]:+.3f},"
              f" men {np.corrcoef(d[m], d.complexity_M)[0, 1]:+.3f}")
    rows = []                                   # X -> complexity(A) -> liking(B)
    for spec, D in [("pooled", zs(d[cols])), ("within", zs(W))]:
        for a, b in [("F", "M"), ("M", "F")]:
            for m in MEAS:
                x, mm, y = D[m].values, D["complexity_" + a].values, D["liking_" + b].values
                rows.append(dict(spec=spec, complexity=a, liking=b, measure=m,
                                 **path_model(x, mm, y, rng)))
    return pd.DataFrame(rows)


def path_model(x, m, y, rng):
    def ab(i):
        a = np.polyfit(x[i], m[i], 1)[0]
        X = np.column_stack([np.ones(len(i)), x[i], m[i]])
        return a, *np.linalg.lstsq(X, y[i], rcond=None)[0][1:]
    n = len(x)
    a, c_dir, b = ab(np.arange(n))
    boot = [np.prod(ab(rng.integers(0, n, n))[::2]) for _ in range(N_BOOT)]
    X = sm.add_constant(np.column_stack([x, m]))
    fit = sm.OLS(y, X).fit()
    return dict(a=a, b=b, c_direct=c_dir, p_direct=fit.pvalues[1], indirect=a * b,
                ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5))


def reliability(d):
    out = {}
    for k, r in RAT.items():
        rh = np.corrcoef(d[k + "_F"], d[k + "_M"])[0, 1]
        rel = 2 * rh / (1 + rh)                                  # Spearman-Brown
        n_raters = np.where(d["Subset_Rated_by_Everyone"] == 1, 120, 20)
        rel_sd = 1 - np.mean(d[f"All_{r}_SD"] ** 2 / n_raters) / d[k].var()
        out[k] = dict(r_women_men=rh, reliability=rel, ceiling_r=np.sqrt(rel), reliability_sd=rel_sd)
        print(f"  {k:<11} women-men r = {rh:.3f}, reliability {rel:.3f} (SD-based {rel_sd:.3f}),"
              f" ceiling r = {np.sqrt(rel):.3f}")
    c = out["complexity"]["reliability"]
    for m in MEAS:
        r = np.corrcoef(d[m], d.complexity)[0, 1]
        print(f"  {m:<8} |r|/ceiling = {abs(r) / np.sqrt(c):.0%}, r^2/reliability = {r * r / c:.0%}")
    return pd.DataFrame(out).T


def cross_validation(d):
    y, rows = d.complexity.values, []
    models = {"image covariates": COV, "H alone": ["H"], "H, FuzzyEn, LZC": DINO,
              "H, FuzzyEn, LZC + covariates": DINO + COV}

    def score(X, splits):
        pred = np.zeros_like(y)
        for tr, te in splits:
            pred[te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.corrcoef(pred, y)[0, 1]

    for name, cols in models.items():
        X = d[cols].values
        rep = [score(X, KFold(10, shuffle=True, random_state=SEED + i).split(X)) for i in range(10)]
        g = score(X, GroupKFold(10).split(X, y, d.grp))
        rows.append(dict(model=name, cv_R2=np.mean([r[0] for r in rep]),
                         cv_R2_sd=np.std([r[0] for r in rep]), cv_r=np.mean([r[1] for r in rep]),
                         grouped_R2=g[0], grouped_r=g[1]))
    return pd.DataFrame(rows)


def familiarity(d):
    G = d.loc[d.in_grp, "grp"].nunique()
    cols = MEAS + ["complexity", "liking"]
    W = within(d, cols + ["familiarity"])
    specs = {"| familiarity": (resid(d, cols, ["familiarity"]), 1),
             "| familiarity + covariates": (resid(d, cols, ["familiarity"] + COV), 1 + len(COV)),
             "within | familiarity": (resid(W, cols, ["familiarity"]), G + 1),
             "familiarity < 3 only": (d[d.familiarity < 3][cols], 0)}
    rows = []
    for spec, (D, k) in specs.items():
        for o in ["complexity", "liking"]:
            for m in MEAS:
                rows.append((spec, o, m, *corr(D[m], D[o], len(D) - 2 - k, len(D) - k)[:2]))
    t = pd.DataFrame(rows, columns=["spec", "rating", "measure", "r", "p"])
    t["p_fdr"] = t.groupby(["spec", "rating"])["p"].transform(fdr)    # 4 tests per row
    return t


def categories(d):
    """Exploratory: r per content category, raw and within period (period groups
    >= 10 paintings in the category); liking also with complexity held constant;
    slope heterogeneity across categories; inverted U for liking per category."""
    import statsmodels.formula.api as smf
    cols = MEAS + list(RAT)
    rows, het, quad = [], [], []
    for cat, c in d.groupby("Category_Name"):
        big = c[c["grp"].map(c["grp"].value_counts()) >= 10]
        G, W = big["grp"].nunique(), big[cols] - big.groupby("grp")[cols].transform("mean")
        P = resid(W, MEAS + ["liking"], ["complexity"])
        for o in RAT:
            for m in MEAS:
                rows.append((cat, o, m, "raw", len(c), *corr(c[m], c[o], len(c) - 2, len(c))))
                rows.append((cat, o, m, "within period", len(W),
                             *corr(W[m], W[o], len(W) - G - 2, len(W) - G)))
        for m in MEAS:
            rows.append((cat, "liking", m, "within period | complexity", len(P),
                         *corr(P[m], P["liking"], len(P) - G - 3, len(P) - G - 1)))
            z = zs(c[m])
            f = smf.ols("y ~ z + I(z**2)", pd.DataFrame({"y": c.liking, "z": z})).fit()
            b1, b2 = f.params["z"], f.params["I(z ** 2)"]
            quad.append(dict(category=cat, measure=m, b2=b2, p=f.pvalues["I(z ** 2)"],
                             vertex_pct=stats.percentileofscore(z, -b1 / (2 * b2))))
    t = pd.DataFrame(rows, columns=["category", "rating", "measure", "spec", "n", "r", "p", "lo", "hi"])
    t["p_fdr"] = t.groupby(["rating", "spec"])["p"].transform(fdr)  # 5 categories x 4 measures
    for o in RAT:
        for m in MEAS:
            D = pd.DataFrame({"y": zs(d[o]), "x": zs(d[m]), "cat": d.Category_Name, "per": d.Epoch_Style})
            for label, extra in [("category only", ""), ("+ period effects", " + C(per)")]:
                full = smf.ols("y ~ C(cat) * x" + extra, D).fit()
                F, p, df = full.compare_f_test(smf.ols("y ~ C(cat) + x" + extra, D).fit())
                het.append(dict(rating=o, measure=m, model=label, F=F, df1=int(df),
                                df2=int(full.df_resid), p=p))
    quad = pd.DataFrame(quad)
    quad["p_fdr"] = fdr(quad.p)
    quad["optimum"] = (quad.b2 < 0) & (quad.p_fdr < .05) & quad.vertex_pct.between(5, 95)
    return t, pd.DataFrame(het), quad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vaps", required=True, help="VAPS_Data_per_Artwork.csv")
    d = load(ap.parse_args().vaps)
    rng = np.random.default_rng(SEED)
    pd.set_option("display.width", 200)
    print(f"n = {len(d)} paintings; within-group analyses: {d.in_grp.sum()} paintings,"
          f" {d.loc[d.in_grp, 'grp'].nunique()} genre x period groups\n")

    print("DESCRIPTIVES\n", d[MEAS + ["H_pixel"]].describe().T[["mean", "std", "min", "max"]].round(3))
    print("\nINTERCORRELATIONS\n", d[MEAS + ["H_pixel"]].corr().round(3))

    t = correlations(d)
    t.to_csv("results/stats_correlations.csv", index=False)
    print("\nCORRELATIONS WITH RATINGS (FDR across 16 tests per spec)\n",
          t.round(3).to_string(index=False))
    r, p, lo, hi = corr(d.H_pixel, d.complexity, len(d) - 2, len(d))
    print(f"\npixel H x complexity: r = {r:.3f} [{lo:.3f}, {hi:.3f}], p = {p:.2g}")

    print("\nTERTILES")
    for rating in ["complexity", "liking"]:
        tt, sizes = tertiles(d, rating)
        print(f" {rating} tertiles n = {sizes}\n", tt.round(3).to_string(index=False))

    print("\nHIERARCHICAL REGRESSION (perceived complexity, z-scored)")
    hierarchical(d)

    print("\nSEPARATE RATER GROUPS")
    pm = raters(d, rng)
    pm.to_csv("results/stats_path_models.csv", index=False)
    print(pm.round(3).to_string(index=False))

    print("\nRELIABILITY")
    reliability(d).to_csv("results/stats_reliability.csv")

    print("\nHELD-OUT PREDICTION OF PERCEIVED COMPLEXITY")
    cv = cross_validation(d)
    cv.to_csv("results/stats_cross_validation.csv", index=False)
    print(cv.round(3).to_string(index=False))

    print("\nFAMILIARITY")
    f = d.familiarity
    print(f"  M = {f.mean():.2f}, SD = {f.std():.2f}, {(f < 3).mean():.1%} below 3;"
          + "".join(f" r({c}) = {np.corrcoef(f, d[c])[0, 1]:+.3f}" for c in ["complexity", "liking"] + MEAS))
    fm = familiarity(d)
    fm.to_csv("results/stats_familiarity.csv", index=False)
    print(fm.round(3).to_string(index=False))

    print("\nPADDING:", {m: round(np.corrcoef(d.pad_fraction, d[m])[0, 1], 3) for m in MEAS})

    print("\nCATEGORIES (exploratory)")
    ct, het, quad = categories(d)
    ct.to_csv("results/stats_categories.csv", index=False)
    het.to_csv("results/stats_category_heterogeneity.csv", index=False)
    quad.to_csv("results/stats_category_quadratic.csv", index=False)
    for o in ["complexity", "liking"]:
        print(ct[(ct.rating == o) & (ct.measure != "C")].pivot_table(
            index="category", columns=["spec", "measure"], values="r").round(2).to_string())
    print(het[het.rating.isin(["complexity", "liking"])].round(4).to_string(index=False))
    print(f"  inverted-U optima found: {int(quad.optimum.sum())} of {len(quad)}")


if __name__ == "__main__":
    main()
