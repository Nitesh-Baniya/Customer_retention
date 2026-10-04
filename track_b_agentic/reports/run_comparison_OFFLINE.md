> OFFLINE STUB RUN - scripted policy, not a real model. Plumbing check only.

| version   | gate   |   completion |   strict_completion |   tool_correct |   verif_recall |   coverage |   error_rate |   pct_tests_passed |   correctness |   completeness |   avg_iters |   avg_tool_calls |   avg_tokens |   avg_latency_s |
|:----------|:-------|-------------:|--------------------:|---------------:|---------------:|-----------:|-------------:|-------------------:|--------------:|---------------:|------------:|-----------------:|-------------:|----------------:|
| v1        | FAIL   |        0.375 |               0.375 |           0.75 |              0 |      0.625 |            0 |              81.25 |         0.75  |          0.875 |       1.75  |             0.75 |      1668.75 |           0.004 |
| v2        | FAIL   |        0.375 |               0.375 |           1    |              0 |      0.75  |            0 |              81.25 |         0.75  |          0.875 |       2     |             1    |      2467.38 |           0.007 |
| v3        | PASS   |        1     |               1     |           1    |              1 |      1     |            0 |              93.75 |         0.875 |          1     |       2.375 |             2    |      4025.25 |           0.011 |
