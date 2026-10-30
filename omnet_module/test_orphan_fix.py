#!/usr/bin/env python3
"""
Verify that OMNeT++ subprocesses do not leak as orphaned processes.

No OMNeT++ binary is required — `sleep 3600` stands in for the subprocess.

Tests:
  1. _kill_proc() pattern kills the subprocess (clean close path).
  2. prctl PR_SET_PDEATHSIG: children die automatically when parent is SIGKILL'd.
  3. Baseline (no prctl): confirms orphans DO survive without prctl.

Run:
  python test_orphan_fix.py
"""
import ctypes
import os
import signal
import subprocess
import sys
import time

# ── helpers ───────────────────────────────────────────────────────────────────

def proc_alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def _pdeathsig_preexec():
    try:
        ctypes.CDLL(None, use_errno=True).prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def spawn_sleep(pdeathsig: bool) -> subprocess.Popen:
    return subprocess.Popen(
        ["sleep", "3600"],
        process_group=0,
        preexec_fn=_pdeathsig_preexec if pdeathsig else None,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── Test 1: _kill_proc() pattern kills subprocess ────────────────────────────

def test_kill_proc():
    p = spawn_sleep(pdeathsig=False)
    pid = p.pid
    assert proc_alive(pid), f"Process {pid} not alive before kill"

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        p.wait(timeout=5)
    except Exception:
        pass

    time.sleep(0.1)
    assert not proc_alive(pid), f"Process {pid} still alive after killpg (unexpected)"
    print("[PASS] test_kill_proc: _kill_proc() pattern kills subprocess")


# ── Test 2: prctl — no orphans after parent SIGKILL ──────────────────────────

def test_prctl_no_orphans():
    """
    Fork a simulated 'training parent' that spawns 3 sleep processes with
    prctl PR_SET_PDEATHSIG. Kill the parent with SIGKILL. Verify all children die.
    """
    pid_file = "/tmp/_omnet_orphan_test_pids"

    parent = os.fork()
    if parent == 0:
        procs = [spawn_sleep(pdeathsig=True) for _ in range(3)]
        with open(pid_file, "w") as f:
            f.write("\n".join(str(p.pid) for p in procs))
        time.sleep(60)
        os._exit(0)

    time.sleep(0.5)

    with open(pid_file) as f:
        child_pids = [int(x) for x in f.read().strip().split("\n")]

    for pid in child_pids:
        assert proc_alive(pid), f"Child {pid} not alive before kill"

    os.kill(parent, signal.SIGKILL)
    os.waitpid(parent, 0)
    time.sleep(0.4)

    orphans = [pid for pid in child_pids if proc_alive(pid)]
    assert not orphans, f"FAIL: orphaned PIDs survived parent SIGKILL: {orphans}"
    print(f"[PASS] test_prctl_no_orphans: 0/{len(child_pids)} orphans after parent SIGKILL")


# ── Test 3: baseline — WITHOUT prctl, orphans DO survive ─────────────────────

def test_baseline_orphans_exist():
    """
    Confirm that without prctl subprocesses do survive parent SIGKILL.
    Cleans up the survivors before returning.
    """
    pid_file = "/tmp/_omnet_baseline_pids"

    parent = os.fork()
    if parent == 0:
        procs = [spawn_sleep(pdeathsig=False) for _ in range(3)]
        with open(pid_file, "w") as f:
            f.write("\n".join(str(p.pid) for p in procs))
        time.sleep(60)
        os._exit(0)

    time.sleep(0.5)

    with open(pid_file) as f:
        child_pids = [int(x) for x in f.read().strip().split("\n")]

    for pid in child_pids:
        assert proc_alive(pid), f"Child {pid} not alive before kill"

    os.kill(parent, signal.SIGKILL)
    os.waitpid(parent, 0)
    time.sleep(0.4)

    survivors = [pid for pid in child_pids if proc_alive(pid)]

    # Clean up so they don't linger after the test
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    if survivors:
        print(f"[PASS] test_baseline_orphans_exist: {len(survivors)}/{len(child_pids)} "
              f"processes survived without prctl (bug confirmed, fix is necessary)")
    else:
        print("[WARN] test_baseline_orphans_exist: no orphans without prctl on this system "
              "(unexpected on Linux — prctl fix may be redundant here)")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.platform != "linux":
        print("SKIP: orphan tests require Linux /proc filesystem and prctl(2)")
        sys.exit(0)

    print("=" * 60)
    print("OMNeT++ subprocess orphan-prevention tests")
    print("=" * 60)

    print("\n[1] _kill_proc() kills subprocess (clean close path)")
    test_kill_proc()

    print("\n[2] prctl PR_SET_PDEATHSIG — no orphans after parent SIGKILL")
    test_prctl_no_orphans()

    print("\n[3] Baseline — without prctl, orphans survive (confirms the bug)")
    test_baseline_orphans_exist()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
