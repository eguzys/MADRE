#!/bin/bash
#SBATCH --exclude=tikgpu02,tikgpu03,tikgpu04,tikgpu05,tikgpu06,tikgpu07,tikgpu08,tikgpu09,tikgpu10
#SBATCH --job-name=r1465fwdL8
#SBATCH -o /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/om2g_logs/cells/r1465fwdL8.out
#SBATCH -e /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone/om2g_logs/cells/r1465fwdL8.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH -p cpu.normal
# ---- the banded archipelago -------------------------------------------------
export OM2G_USER_SPAWN=archipelago
export OM2G_CLUSTERS=4
export OM2G_ARENA=3800
export OM2G_SPREAD=25
export OM2G_PCROSS=0.9
export OM2G_DRONES=8
export OMNET_NUM_DRONES=8
export OM2G_MINSEP=1465
export OM2G_MAXNN=1765
export OM2G_MARGIN=300
# ---- EXPLORING STARTS: training only. Eval MUST use OM2G_SPAWN=cluster ------
export OM2G_SPAWN=explore
# ---- shaping, calibrated against sigma(D_i)=3.78 ----------------------
# targets: fiedler 0.10*sigma, voronoi 0.05, edge 0.02, spread off
export OM2G_SHAPE_FIEDLER=65.97
export OM2G_SHAPE_VORONOI=33.24
export OM2G_SHAPE_EDGE=9.089
export OM2G_SHAPE_SPREAD=0
export OM2G_EXTRA_OVERRIDES="experiment.lr=2.5e-4 experiment.checkpoint_interval=24000"
export OM2G_RUN_TAG=r1465fwdL8
export OM2G_FORWARD_HOPS=1
export OM2G_TMPDIR=/usr/itetnas04/data-scratch-01/eguzys/data/tmp/r1465fwdL8
mkdir -p $OM2G_TMPDIR
export OM2G_EXTRA_OVERRIDES="experiment.lr=1e-3 experiment.checkpoint_interval=24000"
export OMNET_ALLOW_CPU=1
export OM2G_LP_PRUNE_BPS=100
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OM2G_EXTRA_OVERRIDES="$OM2G_EXTRA_OVERRIDES experiment.sampling_device=cpu experiment.train_device=cpu experiment.buffer_device=cpu"
export OM2G_SHAPE_FIEDLER=65.97
export OM2G_SHAPE_VORONOI=33.24
export OM2G_SHAPE_EDGE=9.089
export OM2G_CTRL_SPREADING=8
export OM2G_PHYSICAL_METRIC=1
export OM2G_P_SCAFFOLD=0.5
export OM2G_P_RANDOM=0.75
export OM2G_USER_PARTNER_TAG=1
export OM2G_PARTNER_TAG_MODE=group
export OMNET_PEER_LINK_MASK=1
export OM2G_PEER_MEMORY=1
# TALKING arm: peers are visible, message is transmitted
cd /usr/itetnas04/data-scratch-01/eguzys/data/resilient-backbone
bash om2g/om2g_launch_arton.sh compressed st 1 300000
