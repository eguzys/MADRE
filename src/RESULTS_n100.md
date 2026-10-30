# PADRE results — n=100, unified harness (evaluate.py, --condition sinr)

All values: 100 seeded episodes, same eval harness. Median is primary (std >> mean; greedy is zero-heavy).

## Per-N (each model at its trained N)

| model | median | IQR | mean | std |
|---|---|---|---|---|
| greedy N=3 | 0 | 0–530 | 445 | 824 |
| FFN N=3 | 908 | 3–2253 | 1192 | 1244 |
| greedy N=4 | 0 | 0–950 | 548 | 869 |
| FFN N=4 | 554 | 12–2472 | 1144 | 1300 |
| ST N=4 | 1534 | 254–2993 | 1688 | 1367 |
| DeepSets N=4 | 1281 | 277–2945 | 1570 | 1292 |
| greedy N=5 | 7 | 0–1185 | 639 | 950 |
| FFN N=5 | 612 | 157–2190 | 1162 | 1168 |

## Zero-shot transfer (trained N=4 → evaluated at N∈{3,5})

| model | median | IQR | mean | std |
|---|---|---|---|---|
| ST→N=3 | 1281 | 148–2886 | 1524 | 1298 |
| DeepSets→N=3 | 992 | 127–2629 | 1409 | 1280 |
| ST→N=5 | 1964 | 366–2952 | 1797 | 1302 |
| DeepSets→N=5 | 1667 | 625–2921 | 1793 | 1286 |

## Significance (Mann-Whitney U, one-sided, transfer/FFN > greedy)

- ST→N3 > greedy_n3: p=4.7e-13
- ST→N5 > greedy_n5: p=1.4e-12
- DeepSets→N3 > greedy_n3: p=3.9e-12
- DeepSets→N5 > greedy_n5: p=1.1e-12
- ST vs DeepSets (paired Wilcoxon): n.s. at N=3 (p=0.12) and N=5 (p=0.87) — perm-invariant class generalizes, no within-class winner

## Source runs

- FFN/ST/DeepSets native-N: `results/eval_all_20260611_235155`
- ST/DeepSets transfer N=3: `results/eval_fix_20260612_132308`
- DeepSets transfer N=5 + greedy: `results/eval_fix2_20260612_142939`