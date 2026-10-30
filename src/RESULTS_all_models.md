# All models × N × entropy — n=100 (median / mean)

Values are **median / mean** over 100 seeded episodes, unified `evaluate.py --condition sinr` harness.
`—` = not evaluated at n=100. `n/a` = structurally impossible (FFN can't run at non-trained N).

## Default entropy (0.01)

| model | N=3 | N=4 | N=5 |
|---|---|---|---|
| greedy | 0 / 445 | 0 / 548 | 7 / 639 |
| FFN (per-N) | 908 / 1192 | 554 / 1144 | 612 / 1162 |
| ST (train N=4) | 1281 / 1524 | 1534 / 1688 | 1964 / 1797 |
| DeepSets (train N=4) | 992 / 1409 | 1281 / 1570 | 1667 / 1793 |

ST/DeepSets N=3 and N=5 are **zero-shot transfer** from the N=4 checkpoint.
FFN values are independently trained per-N (no transfer; FFN at a non-trained N is n/a — shape mismatch).

## Entropy 0.03 variants (retrained with entropy coefficient 0.03)

| model | N=3 | N=4 | N=5 |
|---|---|---|---|
| FFN ent=0.03 | trained, not n=100-eval'd | — | trained, not n=100-eval'd (train-curve ~1336) |

`ffn_n3_ent03` checkpoints: 4320000/4440000/4560000. `ffn_n5_ent03`: converged ~1336 on train-curve eval_reward, never run through evaluate.py at n=100.

## Significance (n=100, Mann-Whitney U one-sided vs greedy)

ST→N3 p=4.7e-13 · ST→N5 p=1.4e-12 · DeepSets→N3 p=3.9e-12 · DeepSets→N5 p=1.1e-12
ST vs DeepSets paired Wilcoxon: n.s. (N=3 p=0.12, N=5 p=0.87)