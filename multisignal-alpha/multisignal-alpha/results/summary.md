# Pipeline summary

Panel: 299 months x 500 names; signals: sig_momentum, sig_liquidity, sig_volatility, sig_value, sig_quality, sig_dead

## Lookahead demonstration (should look absurd)

| feature       |    IC |    IC_t |
|:--------------|------:|--------:|
| leaky_feature | 0.395 | 178.888 |
| honest_noise  | 0.006 |   2.124 |


## Per-signal evaluation

| signal         |    IC |   IC_t |   ICIR |   ann_ret_gross |   sharpe_gross |   nw_t_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_months |
|:---------------|------:|-------:|-------:|----------------:|---------------:|-------------:|--------------:|-------------:|-----------:|-------------------:|-----------:|
| sig_momentum   | 0.037 | 10.951 |  0.807 |           0.102 |          2.541 |       10.591 |         0.089 |        2.234 |      9.321 |              0.509 |        298 |
| sig_liquidity  | 0.023 |  7.907 |  0.486 |           0.059 |          1.464 |        6.4   |         0.051 |        1.256 |      5.492 |              0.354 |        298 |
| sig_volatility | 0.023 |  9.091 |  0.494 |           0.065 |          1.575 |        7.96  |         0.053 |        1.28  |      6.468 |              0.505 |        298 |
| sig_value      | 0.01  |  4.172 |  0.231 |           0.028 |          0.682 |        3.525 |         0.023 |        0.549 |      2.841 |              0.232 |        298 |
| sig_quality    | 0.013 |  4.981 |  0.292 |           0.037 |          0.917 |        4.528 |         0.031 |        0.753 |      3.713 |              0.277 |        298 |
| sig_dead       | 0.005 |  1.897 |  0.113 |           0.015 |          0.424 |        1.971 |         0.003 |        0.095 |      0.441 |              0.5   |        298 |


## Fama-MacBeth (multivariate marginal power)

| variable       |   mean_coef |   nw_tstat |   n_periods |
|:---------------|------------:|-----------:|------------:|
| const          |      0.0066 |     2.537  |         298 |
| sig_momentum   |      0.0054 |    11.2146 |         298 |
| sig_liquidity  |      0.0033 |     8.3075 |         298 |
| sig_volatility |      0.0033 |     8.9548 |         298 |
| sig_value      |      0.0016 |     4.5059 |         298 |
| sig_quality    |      0.0019 |     5.0338 |         298 |
| sig_dead       |      0.0007 |     1.9487 |         298 |


## Decay (McLean-Pontiff pattern)

| signal         |   in_sample_sharpe |   post_sample_sharpe |   post_pub_sharpe |   post_sample_retention |   post_pub_retention |
|:---------------|-------------------:|---------------------:|------------------:|------------------------:|---------------------:|
| sig_momentum   |              3.643 |                2.161 |             1.451 |                   0.684 |                0.396 |
| sig_liquidity  |              1.668 |                3.555 |             0.709 |                   2.038 |                0.456 |
| sig_volatility |              1.639 |                2.067 |             1.357 |                   1.105 |                0.683 |
| sig_value      |              0.877 |                1.146 |             0.11  |                   0.885 |                0.115 |
| sig_quality    |              1.066 |                1.508 |             0.27  |                   1.445 |                0.242 |
| sig_dead       |              0.586 |                1.146 |            -0.015 |                   1.967 |               -0.028 |


## Model comparison (purged walk-forward, out-of-sample)

| model      |   oos_IC |   oos_ICIR |   ann_ret_gross |   sharpe_gross |   ann_ret_net |   sharpe_net |   nw_t_net |   one_way_turnover |   n_oos_months |
|:-----------|---------:|-----------:|----------------:|---------------:|--------------:|-------------:|-----------:|-------------------:|---------------:|
| elasticnet |    0.043 |      0.925 |           0.124 |          2.929 |         0.113 |        2.663 |      8.244 |              0.474 |            168 |
| lightgbm   |    0.06  |      1.379 |           0.174 |          4.625 |         0.152 |        4.047 |     17.79  |              0.907 |            168 |
| icnet      |    0.07  |      1.5   |           0.207 |          4.842 |         0.195 |        4.572 |     19.946 |              0.484 |            168 |
| pulse      |    0.072 |      1.553 |           0.215 |          5.357 |         0.203 |        5.052 |     20.408 |              0.509 |            168 |


## Factor-controlled alpha (net strategies)

| model      |   alpha_ann |   alpha_t |    r2 |   n |
|:-----------|------------:|----------:|------:|----:|
| elasticnet |       0.114 |     8.274 | 0.032 | 168 |
| lightgbm   |       0.154 |    18.671 | 0.061 | 168 |
| icnet      |       0.196 |    19.627 | 0.051 | 168 |
| pulse      |       0.203 |    20.061 | 0.038 | 168 |


## Deflated Sharpe (pulse)

| model   |   dsr |   sr_star_monthly |   n_trials |   psr_vs_zero |
|:--------|------:|------------------:|-----------:|--------------:|
| pulse   |     1 |             0.795 |         11 |             1 |

