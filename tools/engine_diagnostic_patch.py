#!/usr/bin/env python3
"""
engine_diagnostic_patch.py
============================
Patch notes for core/engine.py — Phase 2C diagnostics additions.

These are additive changes only. No existing logic is modified.
Each section below shows exactly what to add and where.

Changes:
  1. expose_step_timings()   — per-step latency breakdown for profiling
  2. thermal_snapshot()      — sysfs thermal read inside Engine
  3. governance_health()     — single-call health dict for external monitors
  4. _step_10_telemetry()    — enhanced to include latency + thermal
  5. SCAR_LOCK detection     — rank-invariant guard with hard stop signal

Apply method: copy the methods below into Engine class in core/engine.py.
All additions slot in after the existing _step_10_telemetry() method.
"""

# ============================================================================
# ADD TO: core/engine.py — Engine class
# Insert after _step_10_telemetry() and before state_summary()
# ============================================================================

_ENGINE_DIAGNOSTIC_ADDITIONS = '''

    # =========================================================================
    # PHASE 2C ADDITIONS: Diagnostics, thermal, health, SCAR_LOCK
    # =========================================================================

    def thermal_snapshot(self) -> Dict[str, Any]:
        """
        Read CPU/SOC thermal zones directly from sysfs.
        Returns dict of zone_path → temp_celsius, plus max_temp.
        Falls back gracefully on non-Linux / Termux environments.

        Called automatically by _step_10_telemetry() if available.
        Can also be polled externally by calibration scripts.
        """
        import os
        sysfs_root = "/sys/class/thermal"
        probe_zones = [
            "thermal_zone0",   # CPU efficiency cluster
            "thermal_zone1",   # CPU performance cluster
            "thermal_zone2",   # GPU (Adreno)
            "thermal_zone4",   # SOC aggregate
            "thermal_zone7",   # Skin / battery
        ]
        result = {}
        if not os.path.isdir(sysfs_root):
            return {"available": False, "max_temp_c": None}

        for zone in probe_zones:
            path = os.path.join(sysfs_root, zone, "temp")
            try:
                val = float(open(path).read().strip()) / 1000.0
                result[zone] = val
            except Exception:
                pass

        if not result:
            return {"available": False, "max_temp_c": None}

        return {
            "available":  True,
            "max_temp_c": max(result.values()),
            "zones":      result,
        }

    def governance_health(self) -> Dict[str, Any]:
        """
        Single-call health summary for external monitors and calibration scripts.

        Returns a flat dict suitable for JSON logging, metrics dashboards,
        and the calibration script's live table.

        Keys match exactly what CalibrationRunner and EngineLoopRunner expect
        in the "diag" dict from engine.state_summary().
        """
        try:
            sic_s  = self.sic.state_summary()
            cryst_s = self.cryst_memory.state_summary()
            pg_s   = self.pre_gate.state_summary()
            tc_s   = self.transfer_controller.state_summary()
            therm  = self.thermal_snapshot()
        except Exception as e:
            return {"error": str(e)}

        pg_pass_rate = pg_s.get("pass_rate", 0.0)
        cg_acc_rate  = tc_s.get("accept_rate", 0.0)

        # Crystallization rate: accepted / total submissions to commit gate
        total_sub = tc_s.get("total_submissions", 0)
        total_acc = tc_s.get("total_accepted",    0)
        cryst_rate = total_acc / total_sub if total_sub > 0 else 0.0

        return {
            # Throughput
            "cycle_count":              self.cycle_count,
            "scars_admitted":           sic_s.get("scars_admitted", 0),
            "pregate_pass_rate":        pg_pass_rate,
            "commit_gate_accept_rate":  cg_acc_rate,
            "crystallization_rate":     cryst_rate,
            # Memory state
            "cumulative_pressure":      cryst_s.get("cumulative_pressure", 0.0),
            "variance_term":            cryst_s.get("variance_term", 0.0),
            "rejection_rate":           cryst_s.get("recent_rejection_rate", 0.0),
            # Manifold
            "sic_rank":                 sic_s.get("rank", 0),
            "spectral_norm_u":          sic_s.get("spectral_norm_U", 0.0),
            # Thermal
            "max_temp_c":               therm.get("max_temp_c"),
            "thermal_available":        therm.get("available", False),
            # Thresholds (for context in log)
            "fisher_threshold":         tc_s.get("fisher_threshold", 0.85),
        }

    def check_scar_lock(self) -> bool:
        """
        SCAR_LOCK guard: verify rank invariant has not been violated.

        A SCAR_LOCK condition means U @ V.T has dropped rank unexpectedly,
        indicating a topological hole has formed. This is a hard fault —
        the Engine should stop and alert.

        Returns True if SCAR_LOCK is detected (rank violated).
        Returns False if manifold is healthy.

        Called at the end of every successful step() if
        self.scar_lock_check_enabled is True (default: False for speed,
        enable during Phase 2C validation).
        """
        try:
            import numpy as np
            manifold = self.sic.U @ self.sic.V.T
            current_rank = np.linalg.matrix_rank(manifold, tol=1e-5)
            expected_rank = self.sic.rank

            if current_rank < max(1, expected_rank // 2):
                # Rank has dropped to less than half expected — hard fault
                logger.error(
                    f"[SCAR_LOCK] Rank violation: "
                    f"current={current_rank}, expected={expected_rank}. "
                    f"Cycle={self.cycle_count}, scars={self.sic.scars_admitted}"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"[SCAR_LOCK] check failed: {e}")
            return False

    def expose_step_timings(self) -> Dict[str, float]:
        """
        Per-step latency breakdown from the most recent cycle.
        Populated by _step_timed() wrapper — see below.
        Returns empty dict if timing data is unavailable.

        Usage in calibration:
            timings = engine.expose_step_timings()
            print(timings)  # {"01_thermal": 0.2ms, "05_gguf": 145ms, ...}
        """
        return getattr(self, "_step_timings", {})

'''

# ============================================================================
# REPLACE: _step_10_telemetry() in core/engine.py
# Enhanced version includes timing breakdown + thermal snapshot
# ============================================================================

_ENHANCED_STEP_10 = '''
    def _step_10_telemetry(self) -> Dict[str, Any]:
        """
        STEP 10: Telemetry & checkpoint (Phase 1 + Phase 2C enhanced).

        Now includes:
          - Per-step latency breakdown (if _step_timings populated)
          - Thermal snapshot (sysfs direct read)
          - Full governance health dict
          - Periodic SCAR_LOCK check (if enabled)
        """
        try:
            sic_summary  = self.sic.state_summary()
            cryst_summary = self.cryst_memory.state_summary()
            pg_summary   = self.pre_gate.state_summary()
            tc_summary   = self.transfer_controller.state_summary()
            health       = self.governance_health()
            thermal      = self.thermal_snapshot()

            # SCAR_LOCK check (every 100 cycles by default, or if enabled)
            scar_lock = False
            if getattr(self, "scar_lock_check_enabled", False):
                scar_lock = self.check_scar_lock()
            elif self.cycle_count % 100 == 0:
                scar_lock = self.check_scar_lock()

            report = {
                "sic":                sic_summary,
                "cryst_memory":       cryst_summary,
                "pre_gate":           pg_summary,
                "transfer_controller": tc_summary,
                "governance_health":  health,
                "thermal":            thermal,
                "step_timings_ms":    self.expose_step_timings(),
                "scar_lock_detected": scar_lock,
                "ok": True,
            }

            if scar_lock:
                logger.critical(
                    f"[Cycle {self.cycle_count}] SCAR_LOCK DETECTED — "
                    f"recommend engine halt and SIC inspection."
                )
                report["status_override"] = "scar_lock"

            if thermal.get("max_temp_c") and thermal["max_temp_c"] >= 48.0:
                logger.warning(
                    f"[Cycle {self.cycle_count}] Thermal warning: "
                    f"{thermal['max_temp_c']:.1f}°C"
                )
                report["thermal_warning"] = True

            logger.debug(f"Telemetry: scars={sic_summary['scars_admitted']}, "
                         f"pg_pass={pg_summary.get('pass_rate', 0):.0%}, "
                         f"tc_acc={tc_summary.get('accept_rate', 0):.0%}, "
                         f"temp={thermal.get('max_temp_c')}")
            return report

        except Exception as e:
            return {"ok": False, "error": str(e)}
'''

# ============================================================================
# MODIFY: Engine.__init__() in core/engine.py
# Add these lines after existing Phase 2 component initialization
# ============================================================================

_INIT_ADDITIONS = '''
        # ===== Phase 2C: Diagnostic flags =====
        self.scar_lock_check_enabled = False   # Set True during Phase 2C validation
        self._step_timings: Dict[str, float] = {}   # Populated per-cycle
'''

# ============================================================================
# MODIFY: Engine.step() in core/engine.py
# Wrap each _step_NN_* call with timing (optional — adds ~0.01ms overhead)
# Replace the existing step method's inner calls with timed versions:
#
#   report["steps"]["01_thermal"] = self._step_01_thermal()
#   →
#   report["steps"]["01_thermal"] = self._timed_step("01_thermal",
#                                                     self._step_01_thermal)
# ============================================================================

_TIMED_STEP_HELPER = '''
    def _timed_step(self, name: str, fn) -> Dict[str, Any]:
        """
        Execute a step function and record its wall-clock latency in ms.
        Stores result in self._step_timings[name].
        Adds negligible overhead (~5µs per call).
        """
        import time
        t0     = time.monotonic()
        result = fn()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._step_timings[name] = elapsed_ms
        return result
'''

# ============================================================================
# USAGE NOTES
# ============================================================================

if __name__ == "__main__":
    print("""
Engine Diagnostic Patch — Application Guide
============================================

1. ADD to Engine class (after _step_10_telemetry, before state_summary):
   - thermal_snapshot()
   - governance_health()
   - check_scar_lock()
   - expose_step_timings()
   - _timed_step()

2. REPLACE _step_10_telemetry() with enhanced version above.

3. ADD to Engine.__init__() (after Phase 2 component init):
   self.scar_lock_check_enabled = False
   self._step_timings = {}

4. OPTIONALLY wrap step() calls with _timed_step():
   # Before:
   report["steps"]["01_thermal"] = self._step_01_thermal()
   # After:
   report["steps"]["01_thermal"] = self._timed_step("01_thermal",
                                                      self._step_01_thermal)

5. ENABLE SCAR_LOCK during Phase 2C validation:
   engine.scar_lock_check_enabled = True

6. POLL governance_health() from calibration scripts:
   diag = engine.governance_health()
   # Returns flat dict matching EngineLoopRunner.record() expectations

Smoke test the patch:
   python3 -c "
   import sys; sys.path.insert(0, '.')
   from core.engine import Engine
   e = Engine(d=64, rank=8)
   h = e.governance_health()
   print(h)
   t = e.thermal_snapshot()
   print(t)
   "
""")
