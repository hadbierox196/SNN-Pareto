# Revised SNN-Pareto analysis package

This package contains the revised manuscript, publication figures, master results table, related-work comparison, and a paired-statistics script.

## Important final-analysis note

The revised manuscript reports accuracy bootstrap intervals from the final 10,000-image aggregate results. Exact paired McNemar tests require the per-image prediction file from the final Colab run:

`results/final_test_per_image.csv`

That 60,000-row CSV was not available in the artifact workspace used to build this package. The script `scripts/paired_statistics.py` computes the exact paired McNemar tests and paired bootstrap confidence intervals as soon as that CSV is placed in `results/`.

Do not insert McNemar p-values into the manuscript until the script has been run on the actual final per-image file.

## Main files

- `SNN_Pareto_Revised_Manuscript.md`
- `figures/figure_accuracy_energy_pareto.png`
- `figures/figure_accuracy_energy_pareto.pdf`
- `figures/figure_adaptive_stopping_cdf.png`
- `figures/figure_adaptive_stopping_cdf.pdf`
- `results/master_results_table.csv`
- `results/related_work_comparison.csv`
- `scripts/paired_statistics.py`
