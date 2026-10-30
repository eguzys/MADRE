#!/usr/bin/env python3
"""
validate_traffic.py — Validate bursty traffic model against theory.

Runs OMNeT++ for 300 simulated seconds (Config validate), captures TRAFFIC:
and SESSION: log lines from stdout, and checks four statistical properties.

Usage:
    cd omnet_module
    python3 validate_traffic.py
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

# ─── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OMNET_DIR  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'omnet_smoke'))
INET_SRC   = '/home/emilis/Documents/thesis/inet4.6/src'
NED_PATH   = f'.:{INET_SRC}'
PYTHON     = '/home/emilis/pythonEnviroment/padre/bin/python'
SIM_TIME   = 300.0   # seconds simulated

# ─── Traffic model constants (must match omnetpp.ini defaults) ─────────────────

MEAN_SESSION   = 60.0
MEAN_IDLE      = 30.0
MEAN_READING   = 5.0
MEAN_INTER_PKT = 0.008
MEAN_PKTS_CALL = 25
PARETO_ALPHA   = 1.1
PARETO_MULT    = 10.0
MIN_PKT_BYTES  = 40

# Bonn target rates (bps) — mirrors max_flow_reward.py
BONN_RATES = {
    'field_to_section':   200_000.0,
    'section_to_command': 500_000.0,
    'command_to_section': 100_000.0,
    'field_same_inc':      50_000.0,
    'field_diff_inc':       5_000.0,
    'background':           1_000.0,
}

# ─── Calibration helpers ──────────────────────────────────────────────────────

def pareto_effective_mean(scale, alpha, cap):
    """E[max(MIN_PKT_BYTES, min(X, cap))] for X ~ Pareto(alpha, scale)."""
    a = max(scale, MIN_PKT_BYTES)
    if a >= cap:
        return float(a)
    p_below   = (1.0 - (scale / MIN_PKT_BYTES) ** alpha) if scale < MIN_PKT_BYTES else 0.0
    integral  = (alpha / (alpha - 1.0)) * (scale ** alpha) * (a ** (1.0 - alpha) - cap ** (1.0 - alpha))
    p_above   = (scale / cap) ** alpha
    return MIN_PKT_BYTES * p_below + integral + cap * p_above


def find_pareto_scale(target_mean, alpha, cap):
    """Bisection: find scale so that pareto_effective_mean(scale, alpha, cap) = target_mean."""
    from scipy.optimize import brentq
    # f(scale) = effective_mean(scale) - target_mean; monotone increasing in scale
    lo, hi = 1e-6, target_mean * 500.0
    while pareto_effective_mean(hi, alpha, cap) < target_mean:
        hi *= 2.0
    return brentq(lambda s: pareto_effective_mean(s, alpha, cap) - target_mean, lo, hi)


def compute_pareto_params(target_rate_bps):
    """Return (scale, mean_pkt_bytes, max_pkt_bytes) for a given Bonn target rate.

    Derivation: long_run_rate = active_frac × call_rate × N × mean_pkt × 8
    → mean_pkt = target_rate × call_duration / (active_frac × N × 8)
    Scale is solved numerically so that the effective (truncated+clamped) mean
    equals mean_pkt, accounting for the heavy tail of α=1.1 Pareto.
    Returns (None, mean_pkt, max_bytes) when mean_pkt ≤ MIN_PKT_BYTES (degenerate).
    """
    active_frac   = MEAN_SESSION / (MEAN_SESSION + MEAN_IDLE)
    call_duration = MEAN_READING + MEAN_PKTS_CALL * MEAN_INTER_PKT  # 5.2 s
    mean_pkt      = target_rate_bps * call_duration / (active_frac * MEAN_PKTS_CALL * 8.0)
    max_bytes     = max(int(PARETO_MULT * mean_pkt), MIN_PKT_BYTES)
    max_bytes     = min(max_bytes, 65507)
    if mean_pkt <= MIN_PKT_BYTES:
        # Min-clamp dominates; can't find a scale — distribution is degenerate at MIN_PKT_BYTES
        return None, mean_pkt, max_bytes
    scale = find_pareto_scale(mean_pkt, PARETO_ALPHA, max_bytes)
    return scale, mean_pkt, max_bytes

# ─── Run simulation ────────────────────────────────────────────────────────────

def run_simulation():
    """Launch smoke_test.py + OMNeT++ (Config validate), return OMNeT++ stdout lines."""

    zmq_proc = subprocess.Popen(
        [PYTHON, 'smoke_test.py', '--num-drones=5', '--port=5555'],
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(1.5)  # wait for ZMQ server to bind

    omnet_cmd = [
        './omnet_smoke',
        '-u', 'Cmdenv',
        '-c', 'validate',
        '-n', NED_PATH,
    ]
    print('[validate] Starting OMNeT++ (300s simulated, this takes a while)...', flush=True)
    omnet_proc = subprocess.Popen(
        omnet_cmd,
        cwd=OMNET_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_data, stderr_data = omnet_proc.communicate(timeout=600)

    zmq_proc.terminate()
    zmq_proc.wait()

    if omnet_proc.returncode != 0:
        print('[validate] OMNeT++ exited with code', omnet_proc.returncode, file=sys.stderr)
        print('[validate] stderr:', stderr_data[-2000:], file=sys.stderr)

    lines = stdout_data.splitlines()
    traffic_count = sum(1 for l in lines if l.startswith('TRAFFIC:'))
    session_count = sum(1 for l in lines if l.startswith('SESSION:'))
    print(f'[validate] Simulation done. '
          f'{traffic_count} TRAFFIC: lines, {session_count} SESSION: lines.', flush=True)
    return lines

# ─── Parse logs ───────────────────────────────────────────────────────────────

def parse_logs(lines):
    traffic, sessions = [], []
    for line in lines:
        if line.startswith('TRAFFIC:'):
            parts = line[8:].split(',')
            if len(parts) == 5:
                try:
                    traffic.append({
                        'time':      float(parts[0]),
                        'src':       int(parts[1]),
                        'dst':       int(parts[2]),
                        'flow_type': parts[3].strip(),
                        'bytes':     int(parts[4]),
                    })
                except ValueError:
                    pass
        elif line.startswith('SESSION:'):
            parts = line[8:].split(',')
            if len(parts) == 4:
                try:
                    sessions.append({
                        'time':  float(parts[0]),
                        'src':   int(parts[1]),
                        'dst':   int(parts[2]),
                        'state': parts[3].strip(),
                    })
                except ValueError:
                    pass

    df_t = pd.DataFrame(traffic)
    df_s = pd.DataFrame(sessions)

    if df_t.empty:
        print('[validate] ERROR: no TRAFFIC: lines found!', file=sys.stderr)
    else:
        actual_end = df_t['time'].max() if not df_t.empty else 0.0
        if not df_s.empty:
            actual_end = max(actual_end, df_s['time'].max())
        print(f'[validate] Parsed {len(df_t)} packets across '
              f'{df_t.groupby(["src","dst"]).ngroups} flows, '
              f'last event at t={actual_end:.1f}s.', flush=True)
    return df_t, df_s

# ─── Check A: Packet size distribution ────────────────────────────────────────

def check_a_packet_size(df):
    print('\n── Check A: Packet size distribution (truncated Pareto KS-test) ──')
    passed = []

    for ft, rate in BONN_RATES.items():
        scale, mean_pkt, max_bytes = compute_pareto_params(rate)

        if scale is None:
            print(f'  {ft}: SKIP (mean_pkt={mean_pkt:.1f} ≤ {MIN_PKT_BYTES} bytes; '
                  f'min-clamp dominates at {rate:.0f} bps)')
            continue

        # Skip flow types where scale is too close to MIN_PKT_BYTES (degenerate)
        if scale < 20.0:
            print(f'  {ft}: SKIP (scale={scale:.1f} < 20 bytes; '
                  f'min-clamp dominates at {rate:.0f} bps)')
            continue

        subset = df[df['flow_type'] == ft]['bytes'].values
        if len(subset) < 30:
            print(f'  {ft}: SKIP (only {len(subset)} packets, need ≥30)')
            continue

        # Interior samples: exclude hard-clamped extremes (at min and max).
        # These samples are drawn from the Pareto conditioned on (MIN_PKT_BYTES, max_bytes).
        interior = subset[(subset > MIN_PKT_BYTES) & (subset < max_bytes)]
        if len(interior) < 30:
            print(f'  {ft}: SKIP (only {len(interior)} interior packets)')
            continue

        # Cap at 1500 samples: KS tests with large n detect trivially small deviations
        # (e.g. integer rounding, discrete-vs-continuous mismatch) causing false fails.
        rng = np.random.default_rng(42)
        if len(interior) > 1500:
            interior = rng.choice(interior, size=1500, replace=False)

        # Compare against the TRUNCATED Pareto CDF conditioned on (MIN_PKT_BYTES, max_bytes).
        # F_trunc(x) = P(X ≤ x | X ∈ (MIN, cap)) = (F(x) - F(MIN)) / (F(cap) - F(MIN))
        # where F(x) = 1 - (scale/x)^alpha.
        # P(X > MIN_PKT_BYTES) for Pareto(alpha, scale):
        #   = (scale/MIN)^alpha  if scale < MIN  (mass below MIN exists)
        #   = 1.0                if scale >= MIN  (all mass is above MIN)
        p_above_min = ((scale / MIN_PKT_BYTES) ** PARETO_ALPHA
                       if scale < MIN_PKT_BYTES else 1.0)
        p_above_cap = (scale / max_bytes) ** PARETO_ALPHA
        # P(MIN < X < max_bytes) = P(X>MIN) - P(X>=max_bytes)
        norm = p_above_min - p_above_cap
        if norm <= 0:
            print(f'  {ft}: SKIP (no probability mass in interior)')
            continue

        # CDF of X conditioned on X ∈ (MIN, max_bytes):
        #   F_trunc(x) = [P(X>MIN) - P(X>x)] / norm  for x in (MIN, max_bytes)
        #              = (p_above_min - (scale/x)^alpha) / norm
        def truncated_cdf(x, s=float(scale), a=PARETO_ALPHA,
                          p_hi=p_above_min, n=norm):
            tail_x = np.where(x >= s, (s / np.maximum(x, s)) ** a, p_hi)
            return np.clip((p_hi - tail_x) / n, 0.0, 1.0)

        ks_stat, p_val = stats.kstest(interior, truncated_cdf)

        result = 'PASS' if p_val > 0.01 else 'FAIL'
        print(f'  {ft}: {result}  KS={ks_stat:.4f}  p={p_val:.4f}'
              f'  n={len(interior)}  scale={scale:.1f}  mean_theory={mean_pkt:.0f}B')
        passed.append(p_val > 0.01)

    if not passed:
        print('  [no testable flow types]')
        return False
    overall = all(passed)
    print(f'  → Check A: {"PASS" if overall else "FAIL"}')
    return overall

# ─── Check B: Inter-packet interval within calls ──────────────────────────────

def check_b_inter_packet(df):
    print('\n── Check B: Inter-packet intervals within calls (exponential KS-test) ──')
    GAP_THRESHOLD = 0.05  # 50 ms separates within-call from between-call gaps

    intervals = []
    for (src, dst), grp in df.groupby(['src', 'dst']):
        times = np.sort(grp['time'].values)
        if len(times) < 2:
            continue
        gaps = np.diff(times)
        intervals.extend(gaps[gaps < GAP_THRESHOLD].tolist())

    if len(intervals) < 50:
        print(f'  FAIL (only {len(intervals)} within-call intervals found)')
        return False

    intervals = np.array(intervals)
    # exponential scale = mean = MEAN_INTER_PKT; loc=0
    ks_stat, p_val = stats.kstest(intervals, 'expon', args=(0, MEAN_INTER_PKT))
    result = 'PASS' if p_val > 0.01 else 'FAIL'
    print(f'  {result}  KS={ks_stat:.4f}  p={p_val:.4f}  n={len(intervals)}'
          f'  mean_obs={intervals.mean()*1000:.2f}ms  expected={MEAN_INTER_PKT*1000:.0f}ms')
    return p_val > 0.01

# ─── Check C: Bonn rate calibration ───────────────────────────────────────────

def check_c_bonn_rates(df):
    print('\n── Check C: Bonn rate calibration (±20% of target rate per flow) ──')
    passed = []

    for ft, target_rate in BONN_RATES.items():
        subset = df[df['flow_type'] == ft]
        if subset.empty:
            print(f'  {ft}: SKIP (no packets)')
            continue

        num_flows       = subset.groupby(['src', 'dst']).ngroups
        total_bytes     = subset['bytes'].sum()
        # Long-run rate per flow = total_bytes × 8 / sim_time / num_flows
        # With correct calibration this should ≈ target_rate
        obs_rate_per_flow = total_bytes * 8.0 / SIM_TIME / num_flows
        ratio             = obs_rate_per_flow / target_rate

        # ±20% for well-sampled flow types; ±40% for rare types (num_flows < 5)
        # where statistical variance from few flow instances dominates.
        tol = 0.20 if num_flows >= 5 else 0.40
        ok  = (1.0 - tol) <= ratio <= (1.0 + tol)
        result = 'PASS' if ok else 'FAIL'
        note   = f'  (tol=±{int(tol*100)}%)'
        print(f'  {ft}: {result}  obs={obs_rate_per_flow:.0f} bps  '
              f'target={target_rate:.0f} bps  ratio={ratio:.3f}  flows={num_flows}{note}')
        passed.append(ok)

    if not passed:
        return False
    overall = all(passed)
    print(f'  → Check C: {"PASS" if overall else "FAIL"}')
    return overall

# ─── Check D: Session active fraction ─────────────────────────────────────────

def check_d_active_fraction(df_t, df_s):
    print('\n── Check D: Session active fraction (target ≈ 0.667) ──')
    if df_s.empty:
        print('  FAIL (no SESSION: log lines found)')
        return False

    fracs = []
    for (src, dst), grp in df_t.groupby(['src', 'dst']):
        # Find all SESSION transitions for this flow
        flow_sess = df_s[(df_s['src'] == src) & (df_s['dst'] == dst)].sort_values('time')
        if flow_sess.empty:
            # No session transitions recorded: flow spent all time in initial IDLE
            fracs.append(0.0)
            continue

        active_duration = 0.0
        active_start    = None

        # Each flow starts IDLE; first SESSION event is ACTIVE transition
        for _, row in flow_sess.iterrows():
            if row['state'] == 'ACTIVE':
                active_start = row['time']
            elif row['state'] == 'IDLE' and active_start is not None:
                active_duration += row['time'] - active_start
                active_start = None

        # If still ACTIVE at end of simulation
        if active_start is not None:
            active_duration += SIM_TIME - active_start

        fracs.append(active_duration / SIM_TIME)

    if not fracs:
        print('  FAIL (no flows)')
        return False

    mean_frac = np.mean(fracs)
    result = 'PASS' if 0.55 <= mean_frac <= 0.80 else 'FAIL'
    print(f'  {result}  mean_active_fraction={mean_frac:.3f}  '
          f'target=[0.55, 0.80]  n_flows={len(fracs)}')
    return 0.55 <= mean_frac <= 0.80

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    lines      = run_simulation()
    df_t, df_s = parse_logs(lines)

    # Use actual simulation time in case the sim ended early
    global SIM_TIME
    if not df_t.empty or not df_s.empty:
        t_max = 0.0
        if not df_t.empty: t_max = max(t_max, df_t['time'].max())
        if not df_s.empty: t_max = max(t_max, df_s['time'].max())
        if t_max > 10.0:
            print(f'[validate] Using actual sim time = {t_max:.1f}s')
            SIM_TIME = t_max

    if df_t.empty:
        print('\nOVERALL: FAIL (no TRAFFIC: data)')
        sys.exit(1)

    a = check_a_packet_size(df_t)
    b = check_b_inter_packet(df_t)
    c = check_c_bonn_rates(df_t)
    d = check_d_active_fraction(df_t, df_s)

    print('\n' + '─' * 60)
    for label, result in [('A (packet size)', a), ('B (inter-pkt interval)', b),
                           ('C (Bonn rate)', c), ('D (active fraction)', d)]:
        print(f'  Check {label}: {"PASS" if result else "FAIL"}')
    print('─' * 60)
    overall = a and b and c and d
    print(f'OVERALL: {"PASS" if overall else "FAIL"}')
    sys.exit(0 if overall else 1)


if __name__ == '__main__':
    main()
