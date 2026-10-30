#!/bin/bash
#SBATCH --exclude=tikgpu02,tikgpu03,tikgpu04,tikgpu05,tikgpu06,tikgpu07,tikgpu08,tikgpu09,tikgpu10
#SBATCH --job-name=br12rtL32s1u100sp6UB
#SBATCH -o /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/om2g_logs/cells/br12rtL32s1u100sp6UB.out
#SBATCH -e /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/om2g_logs/cells/br12rtL32s1u100sp6UB.err
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -p cpu.normal,gpu.normal,disco.med
# OM2G training cell: 1200 m cluster world, 8 drones, 40 users.
export OM2G_USER_SPAWN=archipelago
export OM2G_CLUSTERS=4
export OM2G_ARENA=3800
export OM2G_SPREAD=25
export OM2G_PCROSS=0.9
export OM2G_DRONES=8
export OMNET_NUM_DRONES=8
export OM2G_MINSEP=1200
export OM2G_MAXNN=1500
export OM2G_MARGIN=300
export OM2G_SPAWN=explore
export OM2G_SHAPE_FIEDLER=65.97
export OM2G_SHAPE_VORONOI=33.24
export OM2G_SHAPE_EDGE=9.089
export OM2G_SHAPE_SPREAD=0
export OMNET_ALLOW_CPU=1
export OM2G_LP_PRUNE_BPS=100
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OM2G_CTRL_SPREADING=8
export OM2G_PHYSICAL_METRIC=1
export OM2G_P_SCAFFOLD=0.5
export OM2G_P_RANDOM=0.75
export OM2G_USER_PARTNER_TAG=1
export OM2G_PARTNER_TAG_MODE=group
export OMNET_PEER_LINK_MASK=1
export OM2G_PEER_MEMORY=1
export OM2G_EXTRA_OVERRIDES="experiment.lr=1e-3 experiment.checkpoint_interval=24000 experiment.sampling_device=cpu experiment.train_device=cpu experiment.buffer_device=cpu"
export OM2G_RUN_TAG=br12rtL32s1u100sp6UB
export OM2G_TMPDIR=/usr/itetnas04/data-scratch-01/eguzys/data/tmp/br12rtL32s1u100sp6UB
mkdir -p $OM2G_TMPDIR
export OM2G_LATENT_DIM=32
# Round-trip codec (OM2G_CODEC_MODE=e2e_rt): the message carries 4L+56 B and
# the receiver expands the L wire floats to a constant OM2G_RT_WIDTH before
# the set transformer, so the peer token has the same shape at every L.
export OM2G_CODEC_MODE=e2e_rt
export OM2G_RT_WIDTH=128
export OM2G_RT_HIDDEN=256
cd /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
# ---- shaping weights for the 1200 m world ----
export OM2G_SHAPE_FIEDLER=9.603
export OM2G_SHAPE_VORONOI=2.789
export OM2G_SHAPE_EDGE=0.7233
# ---- 100-user fine-tune, senderProb 0.4, +48k frames ----
# Resumes from the base run's checkpoint with the highest training return,
# widened to a 100-user cap by pipeline/finetune/widen2.py (critic input
# columns and buffer shapes adjusted, all other tensors unchanged). The
# shaping weights are those of the 40-user base run.
export OM2G_USERS=100
export OM2G_SENDER_PROB=0.6
export OM2G_RESUME_FROM=/usr/itetnas04/data-scratch-01/eguzys/data/u100diag_ckpts/L32_u100sp6UB_s1/checkpoints/checkpoint_216000.pt
export OMNET_RESUME_MAX_FRAMES=264000
export OMNET_EXPECT_ENT=0.01
export OMNET_RESUME_LR=1e-3
export OM2G_STALL_SECONDS=21600
export OMNET_HOST_CODE=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke
export OM2G_USER_BEACON=1
export OM2G_USER_BEACON_INTERVAL=1.0
export OM2G_USER_BEACON_BYTES=1
export OMNET_SMOKE_DIR=/usr/itetnas04/data-scratch-01/eguzys/data/ub_smoke/omnet_smoke
bash om2g/ub/om2g_launch_arton.sh compressed st 1 300000
