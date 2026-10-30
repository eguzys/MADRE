# MADRE: Medium-Aware Multi-Agent Data-Driven Drone Relay Relocation

This branch holds the code behind the bachelor thesis and the paper on MADRE. A
team of drones relays traffic between ground users. Each drone decides where to
fly from what it senses, and it broadcasts a compressed summary of its user
observation to the drones in radio range. Drones and users share one IEEE
802.15.4 channel, so every byte a drone sends takes airtime from the users. The
experiments vary the compression level `L` of that message and measure user
throughput in a packet-level OMNeT++ 6.4 / INET 4.6 simulation. Training is
multi-agent PPO (BenchMARL / TorchRL).

The steps below rebuild the simulator, train the policies, fine-tune them at
100 users, evaluate them and recompute the numbers of the paper's figures.

## Contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Build the simulator](#build-the-simulator)
4. [Python environment](#python-environment)
5. [Smoke test](#smoke-test)
6. [How configuration works](#how-configuration-works)
7. [Reproduce the paper](#reproduce-the-paper)
8. [Names in the code and in the paper](#names-in-the-code-and-in-the-paper)
9. [Results stored in this branch](#results-stored-in-this-branch)
10. [Troubleshooting](#troubleshooting)

## Repository layout

```
om2g/                    OM2G track (802.15.4): constants, config, environment, launchers
om2g/ub/                 launcher used by the 100-user cells (selects the simulator build)
omnet_module/            simulator bridge, reward (max-flow LP), evaluation, baselines, trainers
omnet_smoke/             OMNeT++ C++ modules, NED networks and .ini files
src/                     training stack (BenchMARL / TorchRL), models, Hydra configs
arch_out/cells/          one Slurm script per training run, exactly as submitted
pipeline/eval/           evaluation shards (learned policies and the greedy baseline)
pipeline/suite/          baseline suite (GTM, heuristic message, group relay)
pipeline/finetune/       tools that prepare and submit the 100-user fine-tunes
pipeline/campaign/       submitters used for the evaluation families of the paper
pipeline/scoring/        scorers that turn evaluation logs into figure rows
results/                 evaluation CSVs and archives
requirements-cluster.txt pip freeze of the Python environment all cluster jobs used
```

Main files:

| file | role |
|---|---|
| `om2g/om2g_track.py` | physical constants and derived message sizes |
| `om2g/om2g_config.py` | turns the constants into the environment block and the OMNeT++ arguments |
| `om2g/om2g_env.py` | OM2G environment: message delivery, codec, observation layout |
| `omnet_module/omnet_env.py` | base environment and the ZMQ bridge to the simulator |
| `omnet_module/max_flow_reward.py` | throughput as a max-flow LP (reward) |
| `omnet_module/evaluate_omnet.py` | evaluation of trained policies and baselines |
| `omnet_module/signal_baselines.py` | greedy, heuristic message and relay baselines |
| `omnet_smoke/RLBridge.cc` | simulator side of the bridge |
| `omnet_smoke/UserBeaconApp.cc` | presence beacon of the ground users (`OM2G_USER_BEACON=1`) |
| `src/architectures/user_encoder.py` | message encoders, including the round-trip codec |
| `src/architectures/st_model.py` | set-transformer policy |

## Requirements

- Python 3.11 (the cluster environment ran 3.11.15)
- OMNeT++ 6.4.0 and INET 4.6.0, built with ZMQ (`libzmq3-dev`)
- Apptainer on a cluster, or a native OMNeT++ build on a workstation
- `highspy` for the reward LP
- Slurm for the full reproduction. The scripts in `arch_out/cells/` and
  `pipeline/` were written for the ETH ITET cluster.

Training runs on CPU. The cells set `OMNET_ALLOW_CPU=1` and request 20 CPUs and
48 GB per run (64 GB for the 100-user fine-tunes).

## Build the simulator

Container build (the route used on the cluster):

```bash
mkdir -p /tmp/omnet_stage
cp -r omnetpp-6.4.0 inet4.6 /tmp/omnet_stage/
apptainer build omnet.sif omnet.def
export OMNET_APPTAINER_SIF=$PWD/omnet.sif
```

Then compile the project modules inside the container:

```bash
apptainer exec --bind $PWD:/work omnet.sif bash -c '
  cd /work/omnet_smoke &&
  opp_makemake -f --deep -o omnet_smoke -KINET_PROJ=/opt/inet4.6 -DINET_IMPORT \
      -I/opt/inet4.6/src -L/opt/inet4.6/src -lINET -lzmq &&
  make MODE=release'
```

This produces `omnet_smoke/out/clang-release/omnet_smoke`. The Makefile is
generated and not tracked. Notes on the native build are in
`omnet_smoke/compile2.md` and `omnet_module/compile.md`.

The environment starts the binary as `./out/clang-release/omnet_smoke` from
`<OMNET_HOST_CODE>/omnet_smoke`, where `OMNET_HOST_CODE` is mounted as `/work`
in the container. To run a job against a different build, set `OMNET_HOST_CODE`
to the parent directory of that `omnet_smoke` tree. On the cluster the
100-user cells point it at a separate build tree (`$S/ub_smoke`) compiled from
these sources, which contain `UserBeaconApp`. The user beacon is off unless
`OM2G_USER_BEACON=1`.

## Python environment

On the cluster every job ran in the conda environment whose `pip freeze` is
`requirements-cluster.txt`:

```bash
python3.11 -m venv padre_env
padre_env/bin/pip install -r requirements-cluster.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

For local work `poetry install` installs the package (`benchmarl_padre`,
`architectures`, `evaluation`) from `pyproject.toml`. Whichever interpreter
runs the code needs `torch`, `zmq` and `highspy`.

## Smoke test

Run this before anything long. It starts the simulator, builds the model and
runs one short episode.

```bash
export OM2G_TRACK=1 OM2G_CODEC_MODE=e2e OM2G_LATENT_DIM=32
eval "$(python om2g/om2g_config.py --arm compressed --arch st --shell)"
OM2G_ARM=compressed python om2g/om2g_smoke.py
```

The last line is `=== OM2G SMOKE PASS (arm=compressed) ===`.
`OM2G_SMOKE_STEPS` shortens the episode.

## How configuration works

`om2g/om2g_track.py` holds the constants and `om2g/om2g_config.py` pushes them
to the simulator, the reward LP, the environment and the model. Export the
config before running anything by hand:

```bash
eval "$(python om2g/om2g_config.py --arm compressed --arch st --shell)"
python om2g/om2g_config.py --selftest
```

The launchers do this themselves. `omnet_smoke/om2g.ini` holds defaults only;
the world of a run is set by the `export` lines of its cell file and passed to
OMNeT++ on the command line. `docs/deployed_parameters_verified.md` lists the
values the deployed runs used.

The world of every cell in the paper (`arch_out/cells/br12*.sh`):

| variable | value | meaning |
|---|---|---|
| `OM2G_DRONES`, `OMNET_NUM_DRONES` | 8 | drones |
| `OM2G_ARENA` | 3800 | area side in m |
| `OM2G_USER_SPAWN`, `OM2G_CLUSTERS` | archipelago, 4 | users in 4 clusters |
| `OM2G_MINSEP`, `OM2G_MAXNN` | 1200, 1500 | cluster separation band in m |
| `OM2G_SPREAD`, `OM2G_PCROSS` | 25, 0.9 | cluster spread, cross-cluster call probability |
| `OM2G_SPAWN` | explore (training) | drone start positions |
| `OM2G_LATENT_DIM` | L | compression level |
| `OM2G_CODEC_MODE` | `e2e_rt` for L up to 128, `e2e` for L = 256 and 512 | message codec |
| `OM2G_RT_WIDTH` | 128 | receiver width of the round-trip codec |
| `OM2G_CTRL_SPREADING` | 8 | control-plane spreading factor during training |
| `OM2G_USERS` | 40 (base), 100 (fine-tune) | users |
| `OM2G_SENDER_PROB` | chi | active user ratio (100-user fine-tunes) |

`OM2G_CTRL_SPREADING` multiplies the simulator's drone transmit power and the
airtime charge of a drone message (`om2g/om2g_track.py`). The training cells set
8. Every evaluation reported in the paper runs with spreading factor 1
(`DIAG_SPREADING=1`, drone transmit power 6.59 mW), see step 3 below.

A message at compression level `L` carries `4L + 56` bytes of payload. The
paper's ladder is `L` in {8, 16, 32, 64, 128, 256, 512}.

## Reproduce the paper

The paths below are those of the ITET cluster. Every script sets them in its
first lines (`S=`, `RB=`, `SCRATCH=`, `D=`). For another site, edit those
lines, or set `OM2G_SCRATCH` for the launchers in `om2g/` and `omnet_module/`
that read it.

| symbol | path | content |
|---|---|---|
| `$D` | `/usr/itetnas04/data-scratch-01/eguzys/data` | scratch root |
| `$RB` | `$D/resilient-backbone` | this repository; run outputs go to `src/out/OM2G/` |
| `$S` | `$D/u100diag` | evaluation scripts, logs (`logs/`), CSVs (`out/`) |
| `$S/suite` | | baseline suite scripts, logs and CSVs |
| `$D/u100diag_ckpts` | | widened 100-user starting checkpoints |
| `$D/ub_smoke/omnet_smoke` | | simulator build used by the beacon-on jobs |
| `$D/omnet.sif`, `$D/padre_env` | | container and Python environment |

Deploy the pipeline scripts once:

```bash
cp pipeline/eval/*  $S/
mkdir -p $S/suite && cp pipeline/suite/* $S/suite/
mkdir -p ~/u100diag/c4s/s1
cp pipeline/finetune/widen2.py pipeline/finetune/dropbuf.py ~/u100diag/
cp pipeline/finetune/*.sh pipeline/campaign/*.sh ~/u100diag/c4s/
cp pipeline/campaign/s1/* ~/u100diag/c4s/s1/
```

### Step 1. Train the 40-user policies

One cell per compression level and training run, 300,000 frames each:

```bash
cd $RB
sbatch arch_out/cells/br12rtL64s1.sh
```

| L | cells | training runs |
|---|---|---|
| 8 | `br12rtL8s<k>.sh` | 1, 2, 3, 5 |
| 16 | `br12rtL16s<k>.sh` | 1, 2, 3, 4 |
| 32 | `br12rtL32s<k>.sh` | 1, 2, 3, 5 |
| 64 | `br12rtL64s<k>.sh` | 1, 2, 3, 4 |
| 128 | `br12rtL128s<k>.sh` | 1, 2, 3, 5 |
| 256 | `br12e2eL256s<k>.sh` | 1, 2, 3, 4 |
| 512 | `br12e2eL512s<k>.sh` | 1, 2, 3, 4 |

A cell calls `om2g/om2g_launch_arton.sh compressed st <k> 300000`, which exports
the config and runs `omnet_module/train_omnet_arton.sh` (Hydra entry point
`src/run.py`). Checkpoints are written every 24,000 frames to
`src/out/OM2G/OM2G_compressed_st_s<k>_<cell>/<run>/checkpoints/`, and the job log
to `om2g_logs/cells/<cell>.out`. A finished run prints `=== TRAIN DONE`.

The checkpoint used everywhere is the one with the highest training return:

```bash
python om2g/pick_best_ckpt.py src/out/OM2G/OM2G_compressed_st_s1_br12rtL64s1/<run>
python om2g/pick_best_ckpt.py src/out/OM2G/OM2G_compressed_st_s1_br12rtL64s1/<run> --table
```

### Step 2. Fine-tune at 100 users with the user beacon

Each 40-user run is continued for 48,000 frames at 100 users, once per active
user ratio chi in {0.05, 0.1, 0.2, 0.4, 0.6, 1.0}, with the user presence beacon
on. Three scripts do this.

a. Widen the best checkpoint of a base run to 100 users and create the six
per-chi starting points:

```bash
bash ~/u100diag/c4s/build_stage.sh br12rtL64s4 4 L64 15972
```

Arguments: base cell, training run, stage prefix, observation width at 100
users. The widths are:

| L | prefix | width |
|---|---|---|
| 8 | L8 | 15524 |
| 16 | L16 | 15588 |
| 32 | L32 | 15716 |
| 64 | L64 | 15972 |
| 128 | L128 | 16484 |
| 256 | eL256 | 17508 |
| 512 | eL512 | 19556 |

`widen2.py` inserts the new user columns into the critic and checks every other
tensor on reload. `dropbuf.py` removes the rollout buffer, which PPO refills
before its first update. The stages land in
`$D/u100diag_ckpts/<prefix>_u100sp<chi>UB_s<k>/`.

b. Write the fine-tune cells from the training-run-1 templates in
`arch_out/cells/` (`br12rtL64s1u100sp10UB.sh` and so on, `br12e2eL512s2...` for
L = 512):

```bash
bash ~/u100diag/c4s/gen_cells.sh L64 4 sp05 sp1 sp2 sp4 sp6 sp10
```

The suffix `sp05 ... sp10` is chi = 0.05 ... 1.0. A generated cell differs from
its template only in the tag, the resume checkpoint, the frame limit (best + 48,000)
and the training-run number.

c. Submit them:

```bash
bash ~/u100diag/c4s/submit_cells.sh br12rtL64s4u100sp05UB br12rtL64s4u100sp10UB
```

`submit_cells.sh` submits held jobs (`sbatch -H`); release them with
`scontrol release <jobid>`. The fine-tune cells call `om2g/ub/om2g_launch_arton.sh`,
which is `om2g/om2g_launch_arton.sh` with one difference: it keeps an
`OMNET_HOST_CODE` that the cell sets, so the job runs the beacon build in
`$D/ub_smoke`. Each fine-tune writes into a directory nested in its base run,
`src/out/OM2G/OM2G_compressed_st_s<k>_<base cell>/<prefix>_u100sp<chi>UB_s<k>/`.

### Step 3. Evaluate

Every evaluation family is 100 episodes, run as 10 shards of 10 episodes with
episode seeds 0 to 99 (`seed_start` 0, 10, ..., 90). All families in the paper
are evaluated with `DIAG_SPREADING=1`.

A learned shard:

```
sbatch [--export=...] $S/shard_spr.sbatch    <cell> <users> <seed_start> <episodes> <checkpoint> <name>
sbatch [--export=...] $S/shard_spr_ub.sbatch <cell> <users> <seed_start> <episodes> <checkpoint> <name>
```

`<checkpoint>` is the path relative to `src/` returned by `pick_best_ckpt.py`.
The `_ub` variant is the one with the user beacon; it needs the beacon exports:

```bash
UB=OM2G_USER_BEACON=1,OM2G_USER_BEACON_INTERVAL=1.0,OM2G_USER_BEACON_BYTES=1,OMNET_HOST_CODE=$D/ub_smoke,OMNET_SMOKE_DIR=$D/ub_smoke/omnet_smoke
```

Example, one shard of the 40-user evaluation of `br12rtL64s1`:

```bash
cd $RB
CK=$(cd src && python ../om2g/pick_best_ckpt.py out/OM2G/OM2G_compressed_st_s1_br12rtL64s1/<run> | tail -1)
for ss in 0 10 20 30 40 50 60 70 80 90; do
  sbatch --export=ALL,DIAG_SPREADING=1 --job-name=S1_FS_L64s1_s$ss \
      $S/shard_spr.sbatch br12rtL64s1 40 $ss 10 "$CK" S1_FS_L64s1
done
```

The shard reads the world from the cell file, applies the overrides below and
runs `omnet_module/evaluate_omnet.py`. It writes `$S/logs/<name>_s<ss>.log` (one
line `ep seed= ... reward= ... steps= 250` per episode) and
`$S/out/OM2G_trained_<cell>_<name>_n10_seed<ss>.csv`.

Overrides, all passed through `--export`:

| variable | effect |
|---|---|
| `DIAG_SPREADING=1` | spreading factor 1 at evaluation |
| `DIAG_MINSEP=<d>` | cluster separation band `d` to `d + 300` m |
| `DIAG_SPAWN=uniform` | drones start uniformly over the area (default: on the clusters) |
| `DIAG_SENDERPROB=<chi>` | active user ratio |
| `DIAG_CLUSTERS=4` | number of user clusters (baseline shards) |

Families behind each figure (`<k>` = training run, `<c>` = 100 x chi):

| figure / claim | family | how |
|---|---|---|
| Fig. separation, 1200 m point | `S1_<fam1200>` | `shard_spr.sbatch <base cell> 40`, best checkpoint |
| Fig. separation, 1300 to 1700 m | `S1_SEP<d>_L<L>s<k>` | as above plus `DIAG_MINSEP=<d>` |
| uniform start | `S1_USPAWN_L<L>s<k>` | as above plus `DIAG_SPAWN=uniform` |
| last checkpoint | `S1_CKFIN_L<L>s<k>` | checkpoint `checkpoint_300000.pt` of the base run |
| Fig. outages | `S1_NDUB<c>_L64s<k>` | `shard_spr_ub.sbatch <base cell> 40`, `DIAG_SENDERPROB`, `$UB` |
| Fig. density, MADRE line of Fig. baselines | `S1_PDUB<c>_L<L>s<k>` | `shard_spr_ub.sbatch <fine-tune cell> 100`, `DIAG_SENDERPROB`, `$UB`, best checkpoint of the fine-tune directory |

A 100-user shard finds its run through `src/out/OM2G/OM2G_compressed_st_s<k>_<fine-tune cell>/`;
create that directory (empty) before submitting. L = 256 and 512 shards need
`--mem=32G`.

Baselines (the shard sources the world of `br12rtL128s1.sh`):

```
sbatch --export=ALL,DIAG_SPREADING=1,DIAG_CLUSTERS=4[,DIAG_SENDERPROB=<chi>,$UB] \
    $S/baseline_shard_cl_spr.sbatch    engineered <users> <seed_start> 10 <name>   # beacon off
    $S/baseline_shard_cl_ub_spr.sbatch engineered <users> <seed_start> 10 <name>   # beacon on
sbatch --export=ALL,DIAG_SPREADING=1,DIAG_SENDERPROB=<chi>,$UB[,SUITE_GTM_OBJ=global] \
    $S/suite/suite_shard2_ub_spr.sbatch gtm|aggregate <users> <seed_start> 10 <name>
sbatch --export=ALL,DIAG_SPREADING=1,DIAG_SENDERPROB=<chi>,$UB \
    $S/suite/relay2_shard_ub_spr.sbatch relay_1hop <users> <seed_start> 10 <name>
```

| baseline | 40 users (Fig. outages, separation) | 100 users (Fig. density, Fig. baselines) |
|---|---|---|
| Greedy (`engineered`) | `S1_SEP<d>_ENG`, `S1_USPAWN_ENG`, `S1_NDUB<c>_ENG` | `S1_ENUB_N100_<c>` |
| GTM (`gtm`, `SUITE_GTM_OBJ=global`) | `S1_NDUB<c>_GTMG` | `S1_DUB<c>_GTMG` |
| Heuristic message (`aggregate`) | `S1_NDUB<c>_AGG` | `S1_DUB<c>_AGG` |
| Group relay (`relay_1hop`) | `S1_NDUB<c>_R1H` | `S1_DUB<c>_R1H` |

GTM runs need `--time=16:00:00` at 40 users and `--time=1-12:00:00` at 100 users.
The 40-user greedy reference at 1200 m is the family `SPR1_ENG`.

The scripts that submitted these families for the paper are in
`pipeline/campaign/`: `submit40.sh` (40-user families of one base run),
`submit_outage_baselines.sh`, and `s1/`. `plan_s1.py` writes the list of
families (`plan_s1.tsv`) with cell, checkpoint, overrides and sender probability,
and `reuse_s1.tsv`, which names the `SPR1_*` families that already hold an
identical spreading-1 evaluation. `submit_s1.sh` submits the list through the
functions in `s1lib.sh`. The 1200 m family names `<fam1200>` per compression
level and training run are given by `fam1200()` in `plan_s1.py`. The scorers
read every family under the name without the `S1_` prefix; the pull script
does that mapping with `plan_s1.tsv` and `reuse_s1.tsv`.

### Step 4. Score

The scorers in `pipeline/scoring/` read every `ep seed=` line of the shard logs
and the per-episode CSVs, and write the rows of the paper's figures:

```bash
bash pipeline/scoring/four_seed_staging/rescore_4seed_s1.sh
```

`rescore_4seed_s1.sh` copies the logs and CSVs of all families from the cluster
(`pull_from_cluster_s1.sh`, read-only, needs ssh access) into
`pipeline/scoring/four_seed_staging/pull_raw_s1/`, then runs:

| scorer | output |
|---|---|
| `density_beacon_k3_oct07/score_density_4seed.py` | Fig. density and the MADRE line of Fig. baselines |
| `baselines_beacon_final_oct07/score_baselines_4seed.py` | Fig. baselines |
| `outages_beacon_oct06/score_outages_4seed.py` | Fig. outages |
| `outages_baselines_oct08/score_outage_baselines_4seed.py` | outage probability of the baselines |
| `separation_4seed_oct09/score_separation_4seed.py` | Fig. separation |
| `robustness_4seed_oct09/score_robustness_4seed.py` | uniform start and last checkpoint |

Outputs go to `pipeline/scoring/four_seed_scored/<figure>_4seed_s1/`. Throughput
is `episode reward / 250 x 1000` in kb/s, averaged over all episodes of the pooled
training runs. Confidence intervals are 95 % percentile intervals of a two-level
bootstrap (training runs, then episodes) with B = 20,000 and a fixed RNG seed.

Each scorer also cross-checks its rows against the figure source of the paper,
so it needs the paper's LaTeX source: pass `--paper-tex <file>` (and, for the
density and baselines scorers, `--old-tex <file>`, the reference source with the
beacon-off rows of Fig. density). These files are not part of this repository.

## Names in the code and in the paper

| code | paper |
|---|---|
| `OM2G_LATENT_DIM` | compression level L |
| `OM2G_SENDER_PROB`, `DIAG_SENDERPROB` | active user ratio chi |
| `engineered` | Greedy |
| `gtm` | GTM |
| `aggregate` | Heuristic message |
| `relay_1hop` | Group relay |
| training run `s<k>` in a cell name | independent training run |
| `OM2G` | the 802.15.4 track (all paper results) |
| `OMNET` | the earlier 802.11 track (`omnet_module/` launchers, `omnet_smoke/omnetpp.ini`) |

`om2g_track.require_om2g()` refuses to run unless `OM2G_TRACK=1`, and the OMNET
environment refuses to run with it set, so the two tracks never mix.

## Results stored in this branch

```
results/ladder-20260909/om2g_out/            per-episode CSVs of the 1200 m ladder and spawn runs
results/paper-20260917/om2g_results.tar.gz   per-episode CSVs of the thesis runs
results/paper-20260917/om2g_configs.tar.gz   launcher and config files of those runs
results/aux-20260906/                        merged per-arm CSVs and earlier archives
pipeline/scoring/*/                          inputs of the scorers (episode tables, CSVs)
```

Checkpoints are not stored in the repository (`*.pt` is ignored). On the ITET
cluster they are under `$RB/src/out/OM2G/` and the widened stages under
`$D/u100diag_ckpts/`.

## Troubleshooting

**`AssertionError: rate cap not applied`**: the config was not exported. Run
`eval "$(python om2g/om2g_config.py --arm <arm> --shell)"` first.

**`arm=compressed but $OM2G_CODEC_PATH is unset`**: set `OM2G_CODEC_MODE=e2e`
(or `e2e_rt`), or train a frozen codec with `om2g/om2g_collect_codec_data.py`
and `src/encoder_experiments/train_frozen_omnet_encoder.py`.

**`Class "inet::UserBeaconApp" not found`**: the job ran a build without the
beacon module. Point `OMNET_HOST_CODE` at the parent of a build compiled from
the current `omnet_smoke/`.

**`FATAL: no run dir`** in an evaluation log: create the empty directory
`src/out/OM2G/OM2G_compressed_st_s<k>_<cell>/` (100-user evaluations).

**`FATAL: highspy missing`**: install `highspy`; the reward LP needs it.

**`RuntimeError: No ZMQ reply from OMNeT within 300s`**: the simulator did not
start; the first error in the job's `.err` file names the cause (often the
container mount).

**A tensor-shape mismatch when loading a checkpoint**: `OM2G_LATENT_DIM`,
`OM2G_CODEC_MODE` or the user count does not match the run that wrote the
checkpoint. Evaluate through the shard scripts, which read these from the cell.
