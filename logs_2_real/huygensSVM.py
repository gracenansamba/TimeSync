#!/usr/bin/env python3

import argparse, glob, os
import numpy as np
import pandas as pd

def train_linear_svm(X, y, reg=1e-2, lr=0.1, epochs=2000):
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        margins = y * (X @ w + b)
        mask = margins < 1.0
        if np.any(mask):
            grad_w = reg*w - (X[mask].T @ y[mask]) / n
            grad_b = - y[mask].sum() / n
        else:
            grad_w = reg*w
            grad_b = 0.0
        w -= lr*grad_w
        b -= lr*grad_b
    return w, b

def load_probes(globs):
    files = []
    for pat in globs: files += glob.glob(pat)
    if not files: raise FileNotFoundError(f"No files matched: {globs}")
    dfs = []
    for f in sorted(files):
        df = pd.read_csv(f)
        df["__source"] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def filter_pure(df, eps_fallback=1e-5):
    cols = {c.lower(): c for c in df.columns}
    if "keep_ab" in cols and "keep_ba" in cols:
        pure = df[(df[cols["keep_ab"]]==1) & (df[cols["keep_ba"]]==1)].copy()
        how = "flags keep_ab==1 && keep_ba==1"
        return pure, how

    parts = []
    for (_, g) in df.groupby(["rank","resync_count","__source"], dropna=False):
        g = g.sort_values("tx_a").copy()
        g["_d_txAB"] = g["tx_a"].diff()
        g["_d_rxAB"] = g["rx_b"].diff()
        good_ab = g["_d_txAB"].notna() & g["_d_rxAB"].notna() & ((g["_d_rxAB"]-g["_d_txAB"]).abs() <= eps_fallback)

        if "rx_a" in g.columns and "tx_b" in g.columns:
            g["_d_txBA"] = g["tx_b"].diff()
            g["_d_rxBA"] = g["rx_a"].diff()
            good_ba = g["_d_txBA"].notna() & g["_d_rxBA"].notna() & ((g["_d_rxBA"]-g["_d_txBA"]).abs() <= eps_fallback)
            good = good_ab & good_ba
        else:
            good = good_ab
        parts.append(g[good])
    pure = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    return pure, f"fallback epsilon={eps_fallback}s (no keep_* flags)"

def build_points(df):
    rows = []
    for _, r in df.iterrows():
        # +1 upper band
        rows.append([int(r["resync_count"]), float(r["tx_a"]), float(r["rx_b"])-float(r["tx_a"]),  1.0, r["__source"]])
        # -1 lower band (only if rx_a present)
        rx_a = float(r.get("rx_a", 0.0))
        if rx_a > 0.0:
            rows.append([int(r["resync_count"]), rx_a, float(r["tx_b"])-rx_a, -1.0, r["__source"]])
    pts = pd.DataFrame(rows, columns=["resync","x","y","label","__source"])
    if pts.empty: raise ValueError("No training points after filtering.")
    # center x per window to improve conditioning
    pts["x0_reference_sec"] = pts.groupby("resync")["x"].transform("min")
    pts["x_ctr"] = pts["x"] - pts["x0_reference_sec"]
    return pts

def fit_per_resync(pts, reg=1e-2, lr=0.1, epochs=2000, scale_us=True):
    out = []
    for g, sub in pts.groupby("resync"):
        X = sub[["x_ctr","y"]].to_numpy()
        y = sub["label"].to_numpy()
        # optional: scale to microseconds so margins aren’t tiny
        if scale_us:
            X = X * 1e6  # both columns scaled equally (units: usec)
        if len(sub) < 4 or len(np.unique(y)) < 2:
            continue
        w, b = train_linear_svm(X, y, reg=reg, lr=lr, epochs=epochs)
        if abs(w[1]) < 1e-12:  # degenerate
            continue
        alpha = -w[0]/w[1]
        beta  = -b     /w[1]
        margins = y * (X @ w + b)
        acc = float((margins > 0).mean())
        violations = int((margins < 1.0).sum())
        # which files contributed
        srcs = ",".join(sorted(set(sub["__source"])))
        out.append({
            "resync": int(g),
            "alpha_drift_per_sec": float(alpha),
            "beta_offset_sec_at_x0": float(beta),
            "n_points": int(len(sub)),
            "acc": acc,
            "violations": violations,
            "x0_reference_sec": float(sub["x0_reference_sec"].iloc[0]),
            "sources": srcs
        })
    return pd.DataFrame(out).sort_values("resync")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help='CSV glob(s), e.g. logs_7/probes_p6_rank*.csv')
    ap.add_argument("-o","--out", default="svm_fit_summary.csv")
    ap.add_argument("--eps-fallback", type=float, default=1e-5)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--no-scale-us", action="store_true", help="don’t scale features to microseconds")

    args = ap.parse_args()
    df = load_probes(args.inputs)
    pure, how = filter_pure(df, eps_fallback=args.eps_fallback)
    pts = build_points(pure)

    summary = fit_per_resync(pts, reg=args.reg, lr=args.lr, epochs=args.epochs, scale_us=not args.no_scale_us)

    if summary.empty:
        print("No models produced (not enough pure points / only one class).")
        return
    summary.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    print(summary.to_string(index=False))
    print(f"(Purity selection via: {how})")

if __name__ == "__main__":
    main()

