# OMNeT++ RL Module — Compile Notes

## What's in this folder

Two custom OMNeT++/INET modules that wire a running simulation to a Python RL policy via ZMQ:

| File | Role |
|------|------|
| `RLBridge.{ned,h,cc}` | Singleton `cSimpleModule`. Owns a ZMQ REQ socket, collects observations from all drones each sync tick, sends JSON to Python, blocks on reply, stores actions. |
| `RLControlledMobility.{ned,h,cc}` | INET mobility module (extends `MovingMobilityBase`). Each drone instance reads its `(dx,dy)` velocity from `RLBridge` inside `move()`. |
| `smoke_test.py` | Python REP counterpart — not modified. |
| `Makefile.test` | Standalone compile-check Makefile; produces `.o` files only (no link). |

## Architecture

```
Python (smoke_test.py)
    ↑ REP socket (tcp://127.0.0.1:5555)
    |   JSON: {"type":"obs","step":N,"done":bool,"positions":[[x,y],...]}
    ↓ REQ socket
RLBridge (cSimpleModule)
    ↕ actionBuffer_[droneId] = {dx, dy}
RLControlledMobility × N  (MovingMobilityBase)
    → reads lastVelocity from bridge each move()
```

One ZMQ REQ/REP cycle per sync tick. Bridge schedules itself at `syncInterval` steps; drones are passive readers.

## Build notes

### 1. Local `#include` paths in both `.cc` files

`RLBridge.h` and `RLControlledMobility.h` are standalone files in this directory, not part of INET's source tree, so both `.cc` files include them by local path:

**`RLBridge.cc`**:
```cpp
#include "RLBridge.h"
#include "RLControlledMobility.h"
```

**`RLControlledMobility.cc`**:
```cpp
#include "RLControlledMobility.h"
#include "RLBridge.h"
```

### 2. `-I.` in `Makefile.test`

The local headers are found through `-I.`, the first include flag:
```makefile
CXXFLAGS = -std=c++17 -fPIC -c \
           -I. \
           -I$(INET)/src \
           -I$(OMNETPP)/include \
           -I/usr/include \
           -DNDEBUG
```

## INET 4.6 API — things that were checked and are fine

- `Coord::ZERO` — exists in `inet/common/geometry/common/Coord.h`
- `getCurrentPosition()` — declared pure virtual in `IMobility`; accessible through `MobilityBase*` casts
- `lastVelocity`, `lastPosition`, `lastUpdate` — all protected members of `MovingMobilityBase`; visible to `RLControlledMobility` which inherits from it
- `handleIfOutside(BorderPolicy, Coord&, Coord&)` — correct 3-argument overload exists in `MobilityBase`
- `inet/common/INETMath.h` — exists in INET 4.6 at the expected path
- `zmq::socket_type::req`, `zmq::send_flags::none`, `zmq::recv_flags::none` — modern cppzmq API, correct

## Build

```bash
cd omnet_module
make -f Makefile.test        # produces RLBridge.o + RLControlledMobility.o
make -f Makefile.test clean  # remove .o files
```

Tested with: clang++, OMNeT++ 6.4.0, INET 4.6.0, libzmq3-dev, nlohmann-json3-dev.

## Next step

Integrate into INET's build system (add to an INET-based project's `src/` tree and let `opp_makemake` wire it up).
