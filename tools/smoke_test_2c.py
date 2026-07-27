#!/usr/bin/env python3
"""
smoke_test_2c.py
=================
Phase 2C Smoke Test — Sovereign Logic Core
Real Engine.step() loop with live metrics, thermal monitoring,
and clean shutdown on thermal throttle or SCAR_LOCK.

Usage (mock engine, 150 cycles):
    python3 tools/smoke_test_2c.py --cycles 150

Usage (real model, Snapdragon Adreno GPU offload):
    python3 tools/smoke_test_2c.py --cycles 300 \
        --model ~/models/phi3-mini-4k-instruct-q4.gguf \
        --gpu-layers 33 \
        --threads 6

Usage (CPU-only subprocess fallback):
    python3 tools/smoke_test_2c.py --cycles 200 \
        --model ~/models/mistral-7b-q4_k_m.gguf \
        --backend subprocess

Stop conditions (automatic):
    - Thermal zone 0 >= THERMAL_THROTTLE_C  (default 48°C)
    - Engine reports SCAR_LOCK (SIC rank violation)
    - KeyboardInterrupt (Ctrl+C) — saves report cleanly

Output:
    - Live table printed every LIVE_INTERVAL cycles
    - Final summary + JSON report on exit
    - Optional per-cycle JSONL log (--log-file)
"""

import argparse
import json
import os
import sys
import time
import signal
import logging
import numpy as np
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional

# ── Project root on path ────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger("slc.smoke2c")

# ── Constants ────────────────────────────────────────────────────────────────
THERMAL_THROTTLE_C = 48.0    # °C — abort threshold
THERMAL_WARN_C     = 43.0    # °C — warning threshold
LIVE_INTERVAL      = 50      # print live table every N cycles
SCAR_LOCK_STATES   = {       # engine step() statuses that indicate hard fault
    "sic_rank_violation",
    "scar_lock",
    "topology_corrupted",
}
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"


# ============================================================================
# THERMAL MONITOR
# ============================================================================

class ThermalMonitor:
    """
    Reads CPU/SOC thermal zones from sysfs.
    Works on Android/Termux and standard Linux.
    Falls back gracefully if no thermal zones found.
    """

    SYSFS_ROOT   = "/sys/class/thermal"
    TSENS_ZONES  = [
        # Snapdragon 8 Elite (SM8750) zone names (approximate)
        "thermal_zone0",   # CPU cluster 0 (efficiency)
        "thermal_zone1",   # CPU cluster 1 (performance)
        "thermal_zone2",   # GPU
        "thermal_zone4",   # SOC / TSENS aggregate
        "thermal_zone7",   # skin / battery
    ]

    def __init__(self):
        self._available_zones = self._detect_zones()
        if not self._available_zones:
            logger.warning("No thermal zones found — thermal monitoring disabled.")

    def _detect_zones(self):
        zones = []
        if not os.path.isdir(self.SYSFS_ROOT):
            return zones
        for zname in self.TSENS_ZONES:
            path = os.path.join(self.SYSFS_ROOT, zname, "temp")
            if os.path.exists(path):
                zones.append(path)
        # If named zones not found, try zone0 and zone1 generically
        if not zones:
            for z in ["thermal_zone0", "thermal_zone1"]:
                path = os.path.join(self.SYSFS_ROOT, z, "temp")
                if os.path.exists(path):
                    zones.append(path)
        return zones

    def read_all(self) -> Dict[str, float]:
        """Returns dict: zone_path → temperature in °C."""
        result = {}
        for path in self._available_zones:
            try:
                raw = open(path).read().strip()
                result[path] = float(raw) / 1000.0
            except Exception:
                pass
        return result

    def max_temp(self) -> Optional[float]:
        """Returns highest temperature across all monitored zones, or None."""
        temps = self.read_all()
        return max(temps.values()) if temps else None

    def available(self) -> bool:
        return len(self._available_zones) > 0

    def status_line(self) -> str:
        if not self.available():
            return "thermal: N/A"
        temps = self.read_all()
        if not temps:
            return "thermal: read failed"
        max_t = max(temps.values())
        color = (ANSI_RED if max_t >= THERMAL_THROTTLE_C
                 else ANSI_YELLOW if max_t >= THERMAL_WARN_C
                 else ANSI_GREEN)
        return f"thermal: {color}{max_t:.1f}°C{ANSI_RESET}"


# ============================================================================
# LIVE METRICS TABLE
# ============================================================================

def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"

def _f3(v: float) -> str:
    return f"{v:.4f}"

def _color(v, lo, hi, invert=False):
    ok = lo <= v <= hi
    if invert:
        ok = v <= hi
    return ANSI_GREEN if ok else ANSI_YELLOW

def print_live_table(
    cycle: int,
    total: int,
    elapsed: float,
    stats: Dict[str, Any],
    thermal: ThermalMonitor,
) -> None:
    W = 66
    eta_s = elapsed / cycle * (total - cycle) if cycle > 0 else 0
    eta   = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
    spd   = f"{cycle / elapsed:.1f} cyc/s" if elapsed > 0 else "—"
    therm = thermal.status_line()

    pg  = stats.get("pregate_pass_rate", 0)
    cg  = stats.get("commit_accept_rate", 0)
    cr  = stats.get("crystallization_rate", 0)
    rs  = stats.get("avg_risk_score", 0)
    lv  = stats.get("avg_logit_variance", 0)
    pr  = stats.get("avg_pressure", 0)
    sc  = stats.get("scars_admitted", 0)
    lt  = stats.get("avg_latency_ms", 0)

    print(f"\n{'─' * W}")
    print(f"  {ANSI_BOLD}CYCLE {cycle}/{total}{ANSI_RESET}  "
          f"{ANSI_DIM}{spd}  ETA {eta}  {therm}{ANSI_RESET}")
    print(f"{'─' * W}")
    print(f"  PreGate pass    {_color(pg, 0.60, 0.80)}{_pct(pg)}{ANSI_RESET}   "
          f"CommitGate acc  {_color(cg, 0.70, 0.85)}{_pct(cg)}{ANSI_RESET}   "
          f"Crystallized  {_color(cr, 0.50, 0.65)}{_pct(cr)}{ANSI_RESET}")
    print(f"  Avg risk        {_f3(rs)}   "
          f"Avg logvar      {_color(lv, 0.82, 0.92)}{_f3(lv)}{ANSI_RESET}   "
          f"Pressure      {_color(pr, 0.30, 0.60, invert=True)}{_f3(pr)}{ANSI_RESET}")
    print(f"  Scars admitted  {sc:4d}   "
          f"GGUF latency    {lt:6.0f}ms")

    top_fail = stats.get("top_commit_fail")
    if top_fail:
        print(f"  {ANSI_DIM}Top commit fail: {top_fail}{ANSI_RESET}")
    print(f"{'─' * W}")


# ============================================================================
# STATS ACCUMULATOR (wraps Engine diagnostics into calibration-friendly form)
# ============================================================================

class StepStatsAccumulator:
    def __init__(self):
        self.n          = 0
        self.n_pregate_pass  = 0
        self.n_commit_pass   = 0
        self.n_commit_total  = 0
        self.n_cryst         = 0
        self.risk_scores     : list = []
        self.logvars         : list = []
        self.pressures       : list = []
        self.latencies_ms    : list = []
        self.commit_failures : Dict[str, int] = {}
        self.outcomes        : Dict[str, int] = {}

    def record(self, report: Dict[str, Any], engine_diag: Dict[str, Any]) -> None:
        self.n += 1
        status = report.get("status", "unknown")
        self.outcomes[status] = self.outcomes.get(status, 0) + 1

        steps = report.get("steps", {})

        # PreGate
        pg = steps.get("04_pre_gate", {})
        if pg.get("gate_pass", False):
            self.n_pregate_pass += 1
        if "risk_score" in pg:
            self.risk_scores.append(pg["risk_score"])

        # GGUF latency
        gguf = steps.get("05_gguf", {})
        if "latency_ms" in gguf:
            self.latencies_ms.append(gguf["latency_ms"])
        if "logit_variance" in gguf:
            self.logvars.append(gguf["logit_variance"])

        # Commit gate
        cg = steps.get("06_commit_gate", {})
        if cg:
            self.n_commit_total += 1
            if cg.get("passed", False):
                self.n_commit_pass += 1
            else:
                for check, ok in cg.get("all_checks", {}).items():
                    if not ok:
                        self.commit_failures[check] = self.commit_failures.get(check, 0) + 1

        # Scar
        if status == "success":
            self.n_cryst += 1

        # Memory pressure from engine diagnostics
        cm = engine_diag.get("cryst_memory", {})
        if "cumulative_pressure" in cm:
            self.pressures.append(cm["cumulative_pressure"])

    def summary(self) -> Dict[str, Any]:
        n = self.n
        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0
        top_fail = (max(self.commit_failures, key=self.commit_failures.get)
                    if self.commit_failures else None)
        return {
            "cycles_run":           n,
            "pregate_pass_rate":    self.n_pregate_pass / n if n else 0.0,
            "commit_accept_rate":   (self.n_commit_pass / self.n_commit_total
                                     if self.n_commit_total else 0.0),
            "crystallization_rate": self.n_cryst / n if n else 0.0,
            "avg_risk_score":       safe_mean(self.risk_scores),
            "avg_logit_variance":   safe_mean(self.logvars),
            "avg_pressure":         safe_mean(self.pressures),
            "avg_latency_ms":       safe_mean(self.latencies_ms),
            "scars_admitted":       self.n_cryst,
            "top_commit_fail":      top_fail,
            "commit_failures":      self.commit_failures,
            "outcomes":             self.outcomes,
        }


# ============================================================================
# MAIN SMOKE TEST RUNNER
# ============================================================================

HEALTH_BANDS = {
    "crystallization_rate": (0.45, 0.70),
    "pregate_pass_rate":    (0.60, 0.85),
    "commit_accept_rate":   (0.65, 0.90),
}

EXIT_OK, EXIT_UNHEALTHY, EXIT_INTERRUPTED = 0, 1, 130


def verdict(cr, pg, cg, abort_reason=None):
    """Return (healthy, exit_code, text).

    The previous implementation returned `0 if (healthy or not abort_reason)`.
    Since `healthy` already required `not abort_reason`, that expression
    reduced to "did it abort?" -- the three health bands could never change
    the exit code, so the run could print REQUIRES TUNING and still exit 0.
    A gate that cannot fail is not a gate.

    An operator Ctrl+C is reported separately (130) rather than as a health
    failure: the run was stopped, not found unhealthy.
    """
    if abort_reason == "KeyboardInterrupt":
        return False, EXIT_INTERRUPTED, "■  INTERRUPTED — no verdict"
    lo, hi = HEALTH_BANDS["crystallization_rate"]
    ok_cr = lo <= cr <= hi
    lo, hi = HEALTH_BANDS["pregate_pass_rate"]
    ok_pg = lo <= pg <= hi
    lo, hi = HEALTH_BANDS["commit_accept_rate"]
    ok_cg = lo <= cg <= hi
    healthy = ok_cr and ok_pg and ok_cg and not abort_reason
    if healthy:
        return True, EXIT_OK, "\u2713  HEALTHY \u2014 ready for Phase 2D"
    if abort_reason:
        return False, EXIT_UNHEALTHY, f"\u26a0  ABORTED \u2014 {abort_reason}"
    bad = [n for n, ok in (("crystallization", ok_cr), ("pregate", ok_pg),
                           ("commit-accept", ok_cg)) if not ok]
    return False, EXIT_UNHEALTHY, "\u26a0  REQUIRES TUNING \u2014 out of band: " + ", ".join(bad)


def _selftest():
    """Prove the gate discriminates. Both directions, plus the old bug."""
    checks = []
    mid = (0.55, 0.72, 0.77)          # all three inside their bands
    checks.append(("P1 all rates in band -> exit 0",
                   verdict(*mid)[:2] == (True, EXIT_OK)))
    checks.append(("P2 crystallization too low -> exit 1",
                   verdict(0.10, 0.72, 0.77)[1] == EXIT_UNHEALTHY))
    checks.append(("P3 crystallization too high -> exit 1",
                   verdict(0.99, 0.72, 0.77)[1] == EXIT_UNHEALTHY))
    checks.append(("P4 pregate out of band -> exit 1",
                   verdict(0.55, 0.20, 0.77)[1] == EXIT_UNHEALTHY))
    checks.append(("P5 commit-accept out of band -> exit 1",
                   verdict(0.55, 0.72, 0.10)[1] == EXIT_UNHEALTHY))
    checks.append(("P6 thermal abort with good rates -> exit 1 (was 0)",
                   verdict(*mid, abort_reason="thermal throttle at 49.0C")[1]
                   == EXIT_UNHEALTHY))
    checks.append(("P7 SCAR_LOCK abort -> exit 1",
                   verdict(*mid, abort_reason="SCAR_LOCK: locked")[1]
                   == EXIT_UNHEALTHY))
    checks.append(("P8 Ctrl+C reported as interrupted, not unhealthy",
                   verdict(*mid, abort_reason="KeyboardInterrupt")[1]
                   == EXIT_INTERRUPTED))
    checks.append(("P9 band edges are inclusive",
                   verdict(0.45, 0.60, 0.65)[1] == EXIT_OK
                   and verdict(0.70, 0.85, 0.90)[1] == EXIT_OK))
    checks.append(("P10 just outside an edge fails",
                   verdict(0.7001, 0.72, 0.77)[1] == EXIT_UNHEALTHY))
    checks.append(("P11 failing bands are named in the text",
                   "pregate" in verdict(0.55, 0.20, 0.77)[2]))
    old = lambda h, a: 0 if (h or not a) else 1
    checks.append(("P12 old expression WAS vacuous (regression guard)",
                   old(False, None) == 0 and verdict(0.10, 0.20, 0.10)[1] == 1))
    w = max(len(n) for n, _ in checks)
    for n, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {n.ljust(w)}")
    p = sum(1 for _, ok in checks if ok)
    print(f"\n{p}/{len(checks)} checks passed")
    return 0 if p == len(checks) else 1


def run_smoke_test(args) -> int:
    """Run smoke test. Returns exit code (0=ok, 1=throttle/error)."""

    # ── Imports ──────────────────────────────────────────────────────────────
    try:
        from core.engine import Engine
        _have_engine = True
    except ImportError:
        _have_engine = False

    try:
        from gguf_adapter import build_gguf_engine
    except ImportError:
        sys.path.insert(0, _SCRIPT_DIR)
        from gguf_adapter import build_gguf_engine

    # ── Build GGUF engine ────────────────────────────────────────────────────
    print(f"\n{ANSI_BOLD}[Phase 2C Smoke Test]{ANSI_RESET}  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cycles:  {args.cycles}")
    print(f"  Backend: {args.backend}")
    print(f"  Model:   {args.model or '(mock)'}\n")

    gguf_engine = build_gguf_engine(
        model_path=args.model,
        backend=args.backend,
        n_gpu_layers=args.gpu_layers,
        n_threads=args.threads,
        n_ctx=args.ctx,
        temperature=args.temperature,
    )

    if args.warmup and args.model:
        print("  Warming up engine...")
        gguf_engine.warmup(n=3)
        print("  Warmup done.\n")

    # ── Build Engine ─────────────────────────────────────────────────────────
    if _have_engine:
        print("  Loading real Engine (core/)...")
        engine = Engine(
            d=args.sic_dim,
            rank=args.sic_rank,
            gguf_engine=gguf_engine,
            cryst_memory_window=20,
            pre_gate_config={
                "weights":   (0.25, 0.30, 0.25, 0.20),
                "threshold": args.gate_thr,
                "steepness": 5.0,
            },
            transfer_controller_config={
                "fisher_threshold":      args.fisher_thr,
                "spectral_norm_max":     2.0,
                "geodesic_distance_max": args.geodesic,
                "thermal_multiplier":    args.thermal_mult,
            },
        )
        print(f"  Engine ready: d={args.sic_dim}, rank={args.sic_rank}")
    else:
        print(f"  {ANSI_YELLOW}core/engine.py not importable — "
              f"running standalone GovernanceStepProxy{ANSI_RESET}")
        engine = _GovernanceStepProxy(gguf_engine, args)

    # ── Setup ─────────────────────────────────────────────────────────────────
    thermal  = ThermalMonitor()
    stats    = StepStatsAccumulator()
    log_file = open(args.log_file, "w") if args.log_file else None
    cycle_reports = []
    abort_reason  = None
    t_start       = time.monotonic()

    # Graceful Ctrl+C
    _interrupted = [False]
    def _sigint(sig, frame):
        _interrupted[0] = True
        print(f"\n{ANSI_YELLOW}  Interrupted — saving report...{ANSI_RESET}")
    signal.signal(signal.SIGINT, _sigint)

    if thermal.available():
        print(f"  Thermal monitoring: {len(thermal._available_zones)} zones\n")
    else:
        print(f"  {ANSI_YELLOW}Thermal monitoring unavailable{ANSI_RESET}\n")

    print(f"{'═' * 66}")
    print(f"  Running {args.cycles} cycles — "
          f"live update every {LIVE_INTERVAL} cycles")
    print(f"{'═' * 66}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):
        if _interrupted[0]:
            abort_reason = "KeyboardInterrupt"
            break

        # Thermal check
        t_max = thermal.max_temp()
        if t_max is not None and t_max >= THERMAL_THROTTLE_C:
            abort_reason = f"thermal throttle at {t_max:.1f}°C"
            print(f"\n{ANSI_RED}  ✗ THERMAL ABORT: {t_max:.1f}°C >= "
                  f"{THERMAL_THROTTLE_C}°C{ANSI_RESET}")
            break

        # Run one engine cycle
        try:
            report   = engine.step()
            diag     = engine.state_summary() if hasattr(engine, "state_summary") else {}
        except Exception as e:
            logger.error(f"engine.step() exception at cycle {cycle}: {e}")
            report = {"status": "error", "error": str(e), "steps": {}}
            diag   = {}

        stats.record(report, diag)

        # SCAR_LOCK check
        if report.get("status") in SCAR_LOCK_STATES:
            abort_reason = f"SCAR_LOCK: {report['status']}"
            print(f"\n{ANSI_RED}  ✗ SCAR_LOCK: {report['status']}{ANSI_RESET}")
            break

        # Per-cycle log
        if log_file:
            entry = {
                "cycle":   cycle,
                "ts":      time.monotonic() - t_start,
                "status":  report.get("status"),
                "temp_c":  t_max,
                **{k: v for k, v in stats.summary().items()
                   if k in ("avg_risk_score", "avg_logit_variance",
                             "avg_pressure", "crystallization_rate")},
            }
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()

        if args.verbose:
            cycle_reports.append(report)

        # Live table
        if cycle % LIVE_INTERVAL == 0 or cycle == args.cycles:
            elapsed = time.monotonic() - t_start
            print_live_table(cycle, args.cycles, elapsed, stats.summary(), thermal)

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed_total = time.monotonic() - t_start
    final_stats   = stats.summary()

    print(f"\n{'═' * 66}")
    print(f"  {ANSI_BOLD}FINAL SUMMARY{ANSI_RESET}")
    print(f"{'═' * 66}")
    print(f"  Cycles completed:     {stats.n} / {args.cycles}")
    print(f"  Wall time:            {elapsed_total:.1f}s  "
          f"({stats.n / elapsed_total:.1f} cyc/s)")
    if abort_reason:
        print(f"  {ANSI_RED}Abort reason:         {abort_reason}{ANSI_RESET}")

    print(f"\n  PreGate pass rate:    {_pct(final_stats['pregate_pass_rate'])}")
    print(f"  CommitGate acc rate:  {_pct(final_stats['commit_accept_rate'])}")
    print(f"  Crystallization rate: {_pct(final_stats['crystallization_rate'])}")
    print(f"  Avg logit variance:   {_f3(final_stats['avg_logit_variance'])}")
    print(f"  Avg GGUF latency:     {final_stats['avg_latency_ms']:.0f}ms")
    print(f"  Scars admitted:       {final_stats['scars_admitted']}")

    if final_stats.get("commit_failures"):
        print(f"\n  Commit Gate failures:")
        for check, count in sorted(final_stats["commit_failures"].items(),
                                   key=lambda x: -x[1]):
            print(f"    {check:<28} {count}")

    # Verdict
    cr = final_stats["crystallization_rate"]
    pg = final_stats["pregate_pass_rate"]
    cg = final_stats["commit_accept_rate"]
    healthy, exit_code, verdict_text = verdict(cr, pg, cg, abort_reason)
    verdict_color = ANSI_GREEN if healthy else ANSI_YELLOW
    print(f"\n  {verdict_color}{ANSI_BOLD}{verdict_text}{ANSI_RESET}")
    print(f"{'═' * 66}\n")

    # Save JSON report
    report_path = args.out or (
        f"smoke2c_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    final_stats.update({
        "cycles_requested": args.cycles,
        "elapsed_s":        elapsed_total,
        "abort_reason":     abort_reason,
        "sic_dim":          args.sic_dim,
        "sic_rank":         args.sic_rank,
        "backend":          args.backend,
        "model":            args.model,
        "generated_at":     datetime.now().isoformat(),
        "gate_threshold":   args.gate_thr,
        "fisher_threshold": args.fisher_thr,
        "engine_type":      "real" if _have_engine else "proxy",
    })
    with open(report_path, "w") as f:
        json.dump(final_stats, f, indent=2, default=str)
    print(f"  Report saved → {report_path}\n")

    if log_file:
        log_file.close()
        print(f"  Per-cycle log → {args.log_file}\n")

    return exit_code


# ============================================================================
# GOVERNANCE STEP PROXY
# (used when core/engine.py is not importable but gguf_adapter is available)
# ============================================================================

class _GovernanceStepProxy:
    """
    Minimal stand-in for Engine when core/ is not on path.
    Runs PreGate → GGUF → CommitGate → SIC using inline stubs.
    Used only for standalone smoke test without the full codebase.
    """
    def __init__(self, gguf_engine, args):
        self.gguf    = gguf_engine
        self.args    = args
        self._cycle  = 0
        self._scars  = 0
        self._d      = args.sic_dim
        self._rank   = args.sic_rank
        # Minimal SIC
        import numpy as np
        Q, _ = np.linalg.qr(np.random.randn(self._d, self._rank))
        self.U = Q.astype(np.float32)
        self.V = np.zeros((self._d, self._rank), dtype=np.float32)

        try:
            sys.path.insert(0, os.path.join(_PROJECT_ROOT, "core"))
            from crystallization_memory import CrystallizationMemory
            from pre_inference_gate import PreInferenceGate
            from transfer_controller import TransferController
        except ImportError:
            # Fall back to calibrate_governance stubs
            sys.path.insert(0, _SCRIPT_DIR)
            try:
                from calibrate_governance import (
                    MockCrystallizationMemory as CrystallizationMemory,
                    PreInferenceGate,
                    TransferController,
                )
            except Exception:
                raise RuntimeError(
                    "Cannot import governance modules. "
                    "Run from project root with core/ on PYTHONPATH."
                )

        self.cryst_mem = CrystallizationMemory(window_size=20)
        self.pre_gate  = PreInferenceGate(threshold=args.gate_thr)
        self.tc        = TransferController(fisher_threshold=args.fisher_thr,
                                            geodesic_distance_max=args.geodesic,
                                            thermal_multiplier=args.thermal_mult)
        self._prompts  = [
            "What is the nature of identity in sovereign systems?",
            "How should AI systems govern their own state updates?",
            "Describe topological safety for low-rank manifolds.",
            "Explain Fisher information in confidence estimation.",
        ]

    def step(self) -> Dict[str, Any]:
        self._cycle += 1
        prompt = self._prompts[self._cycle % len(self._prompts)]
        report = {"status": "unknown", "steps": {}}

        gate_pass, risk, factors = self.pre_gate.evaluate(
            prompt, self, self.cryst_mem)
        report["steps"]["04_pre_gate"] = {"gate_pass": gate_pass,
                                          "risk_score": risk}
        if not gate_pass:
            self.cryst_mem.record_deferral("pre_gate")
            report["status"] = "deferred_by_pre_gate"
            return report

        gguf_result = self.gguf.generate(prompt, logprobs=True)
        lv = gguf_result.get("logit_variance", 0.0)
        report["steps"]["05_gguf"] = {"success": gguf_result.get("success"),
                                      "logit_variance": lv,
                                      "latency_ms": gguf_result.get("latency_ms", 0)}
        if not gguf_result.get("success"):
            self.cryst_mem.record_deferral("gguf_fail")
            report["status"] = "deferred_by_gguf"
            return report

        delta = self.tc.draft_delta(gguf_result, self)
        if delta is None:
            report["status"] = "deferred_delta"
            return report

        audit = self.tc.commit_gate_audit(delta, self, self.cryst_mem)
        report["steps"]["06_commit_gate"] = {
            "passed": audit.passed,
            "all_checks": audit.all_checks,
            "rejection_reason": audit.rejection_reason,
        }
        if not audit.passed:
            self.cryst_mem.record_crystallization(lv, delta.topology_strain, True)
            report["status"] = "rejected_by_commit_gate"
            return report

        # SIC update (inline)
        import numpy as np
        seed = hash(gguf_result.get("text", "")) % (2**31)
        x = np.random.RandomState(seed).randn(self._d).astype(np.float32)
        x /= np.linalg.norm(x) + 1e-10
        residual = x - (self.U @ (self.V.T @ x))
        if np.linalg.norm(residual) >= 1e-4:
            du = (residual / np.linalg.norm(residual)).reshape(-1, 1)
            self.U = self.U + 0.01 * (du @ (self.V.T @ x).reshape(1, -1))
            Q, R = np.linalg.qr(self.U)
            self.U = Q
            self.V = self.V @ R.T
            self._scars += 1
            self.cryst_mem.record_crystallization(lv, delta.topology_strain, False)
            report["status"] = "success"
        else:
            report["status"] = "sic_update_noop"
        return report

    def state_summary(self):
        return {
            "cycle_count": self._cycle,
            "cryst_memory": self.cryst_mem.state_summary(),
            "sic": {"scars_admitted": self._scars},
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2C Smoke Test — Sovereign Logic Core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke (mock engine):
  python3 tools/smoke_test_2c.py --cycles 150

  # Real model, Adreno GPU:
  python3 tools/smoke_test_2c.py --cycles 300 \\
      --model ~/models/phi3-mini-q4.gguf --gpu-layers 33

  # CPU-only, subprocess backend:
  python3 tools/smoke_test_2c.py --cycles 200 \\
      --model ~/models/mistral-7b-q4_k_m.gguf --backend subprocess

  # Full diagnostics + per-cycle log:
  python3 tools/smoke_test_2c.py --cycles 500 --verbose \\
      --model ~/models/phi3-mini-q4.gguf --log-file /tmp/slc_cycles.jsonl
""",
    )
    parser.add_argument("--cycles",      type=int,   default=150)
    parser.add_argument("--model",       type=str,   default=None,
                        help="Path to .gguf model (omit for mock)")
    parser.add_argument("--backend",     choices=["auto","llama_cpp_python",
                                                   "subprocess","mock"],
                        default="auto")
    parser.add_argument("--gpu-layers",  type=int,   default=0,
                        dest="gpu_layers")
    parser.add_argument("--threads",     type=int,   default=4)
    parser.add_argument("--ctx",         type=int,   default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-warmup",   action="store_false", dest="warmup")
    parser.add_argument("--sic-dim",     type=int,   default=512, dest="sic_dim")
    parser.add_argument("--sic-rank",    type=int,   default=64,  dest="sic_rank")
    parser.add_argument("--gate-thr",    type=float, default=0.65, dest="gate_thr")
    parser.add_argument("--fisher-thr",  type=float, default=0.85, dest="fisher_thr")
    parser.add_argument("--geodesic",    type=float, default=0.15)
    parser.add_argument("--thermal-mult",type=float, default=1.0, dest="thermal_mult")
    parser.add_argument("--out",         type=str,   default=None)
    parser.add_argument("--log-file",    type=str,   default=None, dest="log_file",
                        help="Optional per-cycle JSONL log path")
    parser.add_argument("--verbose",     action="store_true",
                        help="Store full per-cycle reports in memory")
    parser.add_argument("--selftest", action="store_true",
                        help="Prove the verdict gate discriminates, then exit")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    sys.exit(run_smoke_test(args))


if __name__ == "__main__":
    main()
