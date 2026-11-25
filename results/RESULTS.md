# Own your planner: results

Success = the plan runs in the simulator and the object ends in the goal room. 95% bootstrap CIs. test = 100 unseen houses (6-8 rooms); test_big = 100 larger houses (10-12 rooms).

| system | test (unseen houses) | test_big (10-12 rooms) | test_names (unseen room names) | locked-door houses (test) | sec per plan | arithmetic sanity |
|---|---|---|---|---|---|---|
| base | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | 4.34 | 0.05 |
| fewshot | 0.04 [0.01, 0.08] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 | 6.37 | 0.05 |
| ref-gemma4_12b | 0.78 [0.69, 0.86] | nan [nan, nan] | nan [nan, nan] | 0.72 | 14.09 | - |

Self-training round 1: 0/600 training tasks had at least one verified success among 8 samples; 0 training examples.

## Why plans fail (share of all plans)

```
err                        bad pick  bad place  goal not reached  locked door  no door  no plan    ok  unknown action
system         split                                                                                                 
base           test            0.03       0.00              0.00         0.02     0.94     0.00  0.00            0.01
               test_big        0.00       0.00              0.00         0.01     0.99     0.00  0.00            0.00
               test_names      0.02       0.00              0.01         0.00     0.97     0.00  0.00            0.00
fewshot        test            0.20       0.00              0.00         0.04     0.67     0.05  0.04            0.00
               test_big        0.12       0.00              0.01         0.02     0.77     0.08  0.00            0.00
               test_names      0.16       0.01              0.02         0.05     0.75     0.01  0.00            0.00
ref-gemma4_12b test            0.01       0.00              0.01         0.11     0.09     0.00  0.78            0.00
```

## Paired tests vs base (same houses; sign-flip permutation; Holm)

- fewshot minus base: +0.01 (p = 0.1280, Holm p = 0.1280)
- ref-gemma4_12b minus base: +0.78 (p = 0.0002, Holm p = 0.0004)
