"""
Paired statistics for the final adaptive SNN evaluation.

Expected input:
    results/final_test_per_image.csv

Required columns:
    condition
    correct
    prediction
    true_label

For McNemar:
    adaptive_T64 vs fixed_T32
    adaptive_T64 vs fixed_T64

Also computes:
    - bootstrap 95% CI for accuracy
    - bootstrap 95% CI for paired accuracy difference
    - discordant-pair counts
"""

import os
import math
import numpy as np
import pandas as pd

CSV = "results/final_test_per_image.csv"
B = 10000
SEED = 42

try:
    from statsmodels.stats.contingency_tables import mcnemar
except ImportError:
    raise SystemExit("Install statsmodels: pip install statsmodels")

if not os.path.exists(CSV):
    raise FileNotFoundError(CSV)

df = pd.read_csv(CSV)

required = {"condition", "correct", "prediction", "true_label"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

def pivot_condition(df, condition):
    out = df[df["condition"] == condition][
        ["image_index", "true_label", "prediction", "correct"]
    ].copy()
    out = out.rename(columns={
        "prediction": f"{condition}_prediction",
        "correct": f"{condition}_correct",
    })
    return out

adaptive = pivot_condition(df, "adaptive_T64")
t32 = pivot_condition(df, "fixed_T32")
t64 = pivot_condition(df, "fixed_T64")

def compare(a, b, name_a, name_b):
    x = a.merge(
        b,
        on=["image_index", "true_label"],
        how="inner"
    )

    c = pd.crosstab(
        x[f"{name_a}_correct"],
        x[f"{name_b}_correct"]
    ).reindex(index=[0,1], columns=[0,1], fill_value=0)

    # statsmodels table uses:
    # [[both fail, A correct/B wrong],
    #  [A wrong/B correct, both correct]]
    table = np.asarray(c, dtype=int)

    exact = mcnemar(
        table,
        exact=True,
        correction=False
    )

    asym = mcnemar(
        table,
        exact=False,
        correction=True
    )

    diff = (
        x[f"{name_a}_correct"].mean()
        - x[f"{name_b}_correct"].mean()
    )

    print("\n" + "="*70)
    print(f"{name_a} vs {name_b}")
    print("="*70)
    print("Paired N:", len(x))
    print("\nContingency table:")
    print(table)
    print(f"\nAccuracy {name_a}: {x[f'{name_a}_correct'].mean()*100:.2f}%")
    print(f"Accuracy {name_b}: {x[f'{name_b}_correct'].mean()*100:.2f}%")
    print(f"Paired accuracy difference: {diff*100:.3f} percentage points")
    print(f"McNemar exact p-value: {exact.pvalue:.6g}")
    print(f"McNemar corrected chi-square p-value: {asym.pvalue:.6g}")

    return x, table, exact.pvalue

def bootstrap_accuracy(correct, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct, dtype=np.int8)
    n = len(arr)
    counts = rng.binomial(n, arr.mean(), size=B)
    dist = counts / n
    return np.percentile(dist, [2.5, 97.5]) * 100

def bootstrap_difference(x, a_col, b_col, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(x[a_col], dtype=np.int8)
    b = np.asarray(x[b_col], dtype=np.int8)
    d = a - b
    n = len(d)
    idx = rng.integers(0, n, size=(B, n))
    dist = d[idx].mean(axis=1) * 100
    return np.percentile(dist, [2.5, 97.5])

for cond in ["fixed_T4", "fixed_T8", "fixed_T16", "fixed_T32", "fixed_T64", "adaptive_T64"]:
    g = df[df["condition"] == cond]
    ci = bootstrap_accuracy(g["correct"])
    print(f"{cond}: accuracy={g['correct'].mean()*100:.2f}%, bootstrap 95% CI={ci[0]:.2f}–{ci[1]:.2f}%")

x32, table32, p32 = compare(
    adaptive, t32,
    "adaptive_T64", "fixed_T32"
)
x64, table64, p64 = compare(
    adaptive, t64,
    "adaptive_T64", "fixed_T64"
)

ci32 = bootstrap_difference(
    x32, "adaptive_T64_correct", "fixed_T32_correct"
)
ci64 = bootstrap_difference(
    x64, "adaptive_T64_correct", "fixed_T64_correct"
)

print("\nPaired bootstrap 95% CI for accuracy difference:")
print(f"Adaptive - Fixed T=32: {ci32[0]:.3f} to {ci32[1]:.3f} percentage points")
print(f"Adaptive - Fixed T=64: {ci64[0]:.3f} to {ci64[1]:.3f} percentage points")
