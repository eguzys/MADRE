# Deployed parameter table (machine-verified 2026-09-08)

Every number below was PRINTED BY THE CODE under the exact environment of a
deployed 1200 m cell (`arch_out/cells/w512rtL512s3.sh`), not transcribed from a
config file. Reproduce with the command block at the bottom.

These are the values of a training cell. The evaluations reported in the paper
run the same cells with `DIAG_SPREADING=1`, which sets the control-plane
spreading factor to 1 and the drone control-plane transmit power to 6.59 mW
(see step 3 of the top-level README).

## ⚠️ Do NOT read parameters out of `omnet_smoke/om2g.ini`

The ini is a set of DEFAULTS and several of them are overridden. `om2g/om2g_config.py`
sets them on the OMNeT++ command line at launch:

| ini says | deployed | overridden at |
|---|---|---|
| `*.numDrones = 5` | **8** | `om2g_config.py:168` |
| `*.rlBridge.arenaWidth = 1200` | **3800 m** | `om2g_config.py:137` |
| `*.rlBridge.maxClusters = 2` | **4** | env `OM2G_CLUSTERS` |
| `*.rlBridge.droneSpawnMode = "staging"` | **explore** (train) / **cluster** (eval) | env `OM2G_SPAWN` |
| `*.rlBridge.pCross = 0.5` | **0.9** | env `OM2G_PCROSS` |
| `*.rlBridge.userSpread = 916.67` | **25 m effective** | `om2g_config.py:157` |
| `host transmitter.power = 6.59 mW` | **52.72 mW** | `om2g_config.py:164` |

Use the "deployed" column, not the ini values, when reporting parameters.

## Radio and channel

| Parameter | Value | Source |
|---|---|---|
| PHY | IEEE 802.15.4 narrowband (INET 4.6 `Ieee802154NarrowbandRadio`) | ini, not overridden |
| Bit rate | 250 kb/s | `OM2G_BITRATE_BPS` |
| Centre frequency | 2.45 GHz | `OM2G_CENTER_FREQ_GHZ` |
| Bandwidth | 2.8 MHz | `OM2G_BANDWIDTH_HZ` |
| Path loss | log-normal shadowing, alpha 2.5, sigma 4 dB | `OM2G_PATHLOSS_ALPHA/SIGMA` |
| Noise floor | -96.616 dBm | `OM2G_NOISE_FLOOR_DBM` |
| Receiver sensitivity | -100 dBm | `OM2G_SENSITIVITY_DBM` |
| SNIR threshold | -8 dB | `OM2G_SNIR_THRESHOLD_DB` |
| Drone TX power (data plane) | 6.590 mW | `OM2G_DRONE_TX_MW` |
| Drone TX power (control plane) | **52.720 mW** | `OM2G_CTRL_TX_MW` |
| Control-plane spreading gain | 8x = 9.031 dB | `OM2G_CTRL_SPREADING` |
| User TX power | 0.3295 mW | `OM2G_USER_TX_MW` |
| Drone-drone target range | 523.06 m | `OM2G_TARGET_RANGE_DRONE_M` |
| User-drone target range | 157.82 m | `OM2G_TARGET_RANGE_USER_M` |
| MTU | 118 B | `OM2G_MTU_BYTES` |
| IP+UDP header | 28 B | `OM2G_IP_UDP_HEADER_BYTES` |
| UDP payload per frame | 90 B | `OM2G_MAX_UDP_PAYLOAD_ONE_FRAME` |
| Radio capacity (app layer) | 191 406.25 b/s | `OM2G_RADIO_CAP_APP_BPS` |

The control plane transmits 8x louder than the data plane by design: control
reach without data reach. This is the knob that makes "airtime competes with
payload" measurable rather than trivially true.

## World

| Parameter | Value |
|---|---|
| Arena | 3800 x 3800 m |
| Drones | 8 |
| Users | 40, static |
| User clusters | 4 (archipelago) |
| Cluster min separation | 1200 m (the second world uses 1600 m) |
| Max neighbour separation | 1500 m |
| Wall margin | 300 m |
| User spread within a cluster | 25 m effective |
| Cross-cluster traffic probability | 0.9 |
| Episode length | 250 steps |
| Sync interval | 1 s per step |
| Drone step | 60 m, 2D, `maxSpeed` 60 m/s is a CEILING not a scale |
| Traffic | BonnMotion/PADRE call model, MELPe 21 B every 67.5 ms = 5807 b/s per call, 64 B on air |

## The message-size ladder (this is the independent variable)

`floats = 14 fixed + L` for a compressed message; `14 + 40 x 17` uncompressed.
Frames are `ceil(bytes / 90)`.

| arm | L | floats | bytes | 802.15.4 frames |
|---|---|---|---|---|
| compressed | 8 | 22 | **88** | 1 |
| compressed | 32 | 46 | **184** | 2 |
| compressed | 128 | 142 | **568** | 6 |
| compressed | 256 | 270 | **1080** | 12 |
| compressed | 512 | 526 | **2104** | 22 |
| uncompressed | - | 694 | **2776** | 29 |
| nocontrol | - | 0 | **0** | 0 |

`OM2G_OBS_FEATURES` is **17**, not 13. It is gated: 17 when
`OM2G_USER_PARTNER_TAG` is set (every deployed cell sets it), 13 otherwise
(`om2g_track.py:324`). Dumping the constants without that variable gives 13 and
different byte counts, so export the deployed block below first.

## Reproduce

```bash
cd <repo root>
env OM2G_USER_SPAWN=archipelago OM2G_CLUSTERS=4 OM2G_ARENA=3800 OM2G_SPREAD=25 \
    OM2G_PCROSS=0.9 OM2G_DRONES=8 OMNET_NUM_DRONES=8 OM2G_MINSEP=1200 \
    OM2G_MAXNN=1500 OM2G_MARGIN=300 OM2G_SPAWN=explore OM2G_CTRL_SPREADING=8 \
    OM2G_PHYSICAL_METRIC=1 OM2G_USER_PARTNER_TAG=1 OM2G_PARTNER_TAG_MODE=group \
    OMNET_PEER_LINK_MASK=1 OM2G_PEER_MEMORY=1 CUDA_VISIBLE_DEVICES="" \
    python3 -c "import om2g.om2g_track as T; [print(k, getattr(T,k)) for k in sorted(dir(T)) if k.startswith('OM2G_') and isinstance(getattr(T,k),(int,float,str,bool))]"
```
