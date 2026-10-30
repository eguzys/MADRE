# Canonical ladder names (OM2G compression ladder)

Cells were historically named two ways — `L<latent>` and `f<frames>` — for the
same ladder. **Report everything as `L<latent>`.** Latent width is the knob that
is actually set (`OM2G_LATENT_DIM`); frame count is a consequence of it.

    beaconBytes = 4 * latent + 56        frames = ceil(payload / 96 B)

| canonical | legacy cell names          | latent | beacon B | frames |
|-----------|----------------------------|--------|----------|--------|
| L1        | m1000L1                    |      1 |       60 |      1 |
| L2        | m1000L2, m1200L2, m1465L2  |      2 |       64 |      1 |
| L8        | w1000talk, x1200talk, x1465talk (default arm) | 8 | 88 | 1 |
| L32       | m1000L32, m1200L32, m1465L32 |    32 |      184 |      2 |
| L56       | m1000f3, m1200f3, m1465f3  |     56 |      280 |      3 |
| L80       | m1000f4, m1200f4, m1465f4  |     80 |      376 |      4 |
| L128      | m1000L128, m1200L128, m1465L128 | 128 |    568 |      6 |
| L680      | u1000talk, u1200talk (arm=uncompressed) | 680 | 2776 | 29 |
| deaf      | *sil* cells (OM2G_MSG_ABLATE=mask_out) | – | 88 | 1 |
| quiet     | n*quiet (arm=nocontrol)    |      – |        4 |      1 |

**Do NOT rename the cell scripts or result CSVs.** Cell names are embedded in
run directories, telemetry filenames and every tag already written; renaming
mid-campaign would break provenance on results that cost cluster-days. The
mapping above is the reporting layer.

`f3`/`f4` are the only names that can be misread: they say "3 frames" and
"4 frames", which is true, but every other rung is named by latent. Use L56/L80.
