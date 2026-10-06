# Pipeline summary

> **SMOKE TEST, NOT EVIDENCE.** Only 5 names in the
> cross-section, so the statistical thresholds are relaxed to
> their structural minimum and the sorts are coarse. Treat
> every number below as a check that the plumbing works.
> Rebuild with the full universe
> (`osap.portfolio_signals: all` in real-data) before quoting
> anything.

Panel: 1158 months x 5 names; signals: fmom_1m, fmom_12_2, fmom_12m, fvol_12m, post_pub, years_since_pub

## Lookahead demonstration (should look absurd)

| feature       |     IC |   IC_t |
|:--------------|-------:|-------:|
| leaky_feature |  0.268 | 15.635 |
| honest_noise  | -0.006 | -0.377 |


## Per-signal evaluation

| signal          |    IC |   IC_t |   ICIR |   ann_ret_gross |   sharpe_gross |   nw_t_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_months |
|:----------------|------:|-------:|-------:|----------------:|---------------:|-------------:|--------------:|-------------:|-----------:|-------------------:|-----------:|
| fmom_1m         | 0.083 |  4.618 |  0.134 |           0.082 |          0.294 |        3.247 |         0.055 |        0.195 |      2.135 |              1.155 |       1158 |
| fmom_12_2       | 0.114 |  5.932 |  0.184 |           0.072 |          0.262 |        2.588 |         0.064 |        0.234 |      2.307 |              0.318 |       1158 |
| fmom_12m        | 0.116 |  5.879 |  0.186 |           0.082 |          0.302 |        2.814 |         0.075 |        0.278 |      2.579 |              0.271 |       1158 |
| fvol_12m        | 0.087 |  4.632 |  0.151 |           0.022 |          0.083 |        0.832 |         0.018 |        0.068 |      0.686 |              0.163 |       1158 |
| post_pub        | 0.055 |  1.949 |  0.115 |           0.062 |          0.34  |        1.388 |         0.061 |        0.338 |      1.382 |              0.011 |        156 |
| years_since_pub | 0.027 |  1.036 |  0.054 |           0.022 |          0.129 |        0.735 |         0.022 |        0.128 |      0.732 |              0.004 |        371 |


## Fama-MacBeth (multivariate marginal power)

| variable        |   mean_coef |   nw_tstat |   n_periods |
|:----------------|------------:|-----------:|------------:|
| const           |         nan |        nan |           0 |
| fmom_1m         |         nan |        nan |           0 |
| fmom_12_2       |         nan |        nan |           0 |
| fmom_12m        |         nan |        nan |           0 |
| fvol_12m        |         nan |        nan |           0 |
| post_pub        |         nan |        nan |           0 |
| years_since_pub |         nan |        nan |           0 |


## Decay (McLean-Pontiff pattern)

|    | note                                                                                                                     |
|---:|:-------------------------------------------------------------------------------------------------------------------------|
|  0 | skipped: no publication dates in feature meta (see real-data factor_decay.csv for the per-factor McLean-Pontiff exhibit) |


## Model comparison (purged walk-forward, out-of-sample)

| model      |   oos_IC |   oos_ICIR |   ann_ret_gross |   sharpe_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_oos_months |
|:-----------|---------:|-----------:|----------------:|---------------:|--------------:|-------------:|-----------:|-------------------:|---------------:|
| elasticnet |    0.107 |      0.185 |           0.074 |          0.339 |         0.054 |        0.245 |      2.495 |              0.854 |           1032 |
| lightgbm   |    0.042 |      0.075 |           0.037 |          0.181 |         0.017 |        0.081 |      0.74  |              0.846 |           1032 |
| icnet      |    0.114 |      0.197 |           0.074 |          0.343 |         0.06  |        0.275 |      2.783 |              0.607 |           1032 |
| pulse      |    0.024 |      0.041 |           0.037 |          0.173 |         0.022 |        0.105 |      1.006 |              0.597 |           1032 |


## Factor-controlled alpha (net strategies)

| model      |   alpha_ann |   alpha_t |    r2 |   n |
|:-----------|------------:|----------:|------:|----:|
| elasticnet |       0.031 |     1.072 | 0.242 | 732 |
| lightgbm   |       0.011 |     0.298 | 0.109 | 732 |
| icnet      |       0.016 |     0.617 | 0.339 | 732 |
| pulse      |       0.004 |     0.149 | 0.067 | 732 |


## Deflated Sharpe (icnet)

| model   |   dsr |   sr_star_monthly |   n_trials |   psr_vs_zero |
|:--------|------:|------------------:|-----------:|--------------:|
| icnet   |  0.79 |             0.053 |         11 |         0.992 |

