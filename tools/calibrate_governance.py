#!/usr/bin/env python3
"""
calibrate_governance.py
========================
Phase 2B/2C Calibration Script — Sovereign Logic Core
Governance Layer Diagnostic Harness

Three run modes:
  standalone  — uses inline governance stubs (no core/ needed)
  real        — uses real core/sic.py + governance modules, standalone runner
  engine      — drives real Engine.step() 10-step cycle (full integration test)

Usage:
  # Standalone mock (no dependencies):
  python3 tools/calibrate_governance.py --cycles 500

  # Real SIC, standalone runner:
  python3 tools/calibrate_governance.py --cycles 500 --mode real

  # Real Engine.step() loop:
  python3 tools/calibrate_governance.py --cycles 300 --mode engine \
      --engine mock

  # Real Engine + real GGUF model:
  python3 tools/calibrate_governance.py --cycles 300 --mode engine \
      --engine real_gguf --model ~/models/phi3-mini-q4.gguf --gpu-layers 33

  # Threshold grid sweep:
  python3 tools/calibrate_governance.py --sweep

Output:
  - Console: formatted summary table with traffic-light indicators
  - File:    calibration_report_<timestamp>.json
  - Optional: per-cycle JSONL log (--log-file)
"""

import argparse
import json
import time
import sys
import os
import signal
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# ── Add project root ─────────────────────────────────────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, _PROJECT_ROOT)

# ── ANSI ─────────────────────────────────────────────────────────────────────
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED    = "\033[91m"
ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"


# ============================================================================
# THERMAL MONITOR (shared with smoke_test_2c)
# ============================================================================

class ThermalMonitor:
    SYSFS_ROOT  = "/sys/class/thermal"
    PROBE_ZONES = ["thermal_zone0", "thermal_zone1", "thermal_zone2",
                   "thermal_zone4", "thermal_zone7"]

    def __init__(self):
        self._paths = self._detect()

    def _detect(self):
        paths = []
        if not os.path.isdir(self.SYSFS_ROOT):
            return paths
        for z in self.PROBE_ZONES:
            p = os.path.join(self.SYSFS_ROOT, z, "temp")
            if os.path.exists(p):
                paths.append(p)
        return paths

    def max_temp(self) -> Optional[float]:
        temps = []
        for p in self._paths:
            try:
                temps.append(float(open(p).read().strip()) / 1000.0)
            except Exception:
                pass
        return max(temps) if temps else None

    def available(self) -> bool:
        return bool(self._paths)

    def status_str(self) -> str:
        t = self.max_temp()
        if t is None:
            return f"{ANSI_DIM}thermal:N/A{ANSI_RESET}"
        color = (ANSI_RED if t >= 48 else ANSI_YELLOW if t >= 43 else ANSI_GREEN)
        return f"{color}thermal:{t:.1f}°C{ANSI_RESET}"


# ============================================================================
# MOCK COMPONENTS (standalone mode / testing)
# ============================================================================

class MockCrystallizationMemory:
    def __init__(self, window_size=20):
        self.window_size    = window_size
        self.delta_history  = deque(maxlen=window_size)
        self.pressure_history = deque(maxlen=window_size)
        self.rejection_history = deque(maxlen=window_size)
        self.total_crystallizations = 0
        self.total_rejections = 0
        self.total_deferrals  = 0

    def record_crystallization(self, logit_variance, topological_strain, was_rejected=False):
        self.delta_history.append(logit_variance)
        self.pressure_history.append(topological_strain)
        self.rejection_history.append(int(was_rejected))
        self.total_crystallizations += 1
        if was_rejected:
            self.total_rejections += 1

    def record_deferral(self, reason=""):
        self.total_deferrals += 1

    def variance_term(self, use_recent_only=True):
        if not self.delta_history:
            return 0.0
        data = ([v for v, r in zip(self.delta_history, self.rejection_history) if r == 0]
                if use_recent_only else list(self.delta_history))
        return float(np.var(data)) if len(data) >= 2 else 0.0

    def rejection_rate(self, window=None):
        h = list(self.rejection_history)[-window:] if window else list(self.rejection_history)
        return float(np.mean(h)) if h else 0.0

    def cumulative_pressure(self, decay_factor=0.95):
        if not self.pressure_history:
            return 0.0
        p = list(self.pressure_history)
        n = len(p)
        ws   = sum(v * (decay_factor ** (n - 1 - i)) for i, v in enumerate(p))
        norm = sum(decay_factor ** j for j in range(n))
        return float(ws / norm)

    def mean_logvar(self):
        return float(np.mean(self.delta_history)) if self.delta_history else 0.0

    def state_summary(self):
        rr = self.rejection_rate()
        return {
            "total_crystallizations": self.total_crystallizations,
            "total_rejections":       self.total_rejections,
            "total_deferrals":        self.total_deferrals,
            "acceptance_rate":        1.0 - rr,
            "mean_logvar":            self.mean_logvar(),
            "variance_term":          self.variance_term(),
            "cumulative_pressure":    self.cumulative_pressure(),
            "recent_rejection_rate":  self.rejection_rate(window=5),
        }


class MockSICState:
    def __init__(self, dim=512, rank=64):
        self.d = dim
        self.rank = rank
        self.U = np.linalg.qr(np.random.randn(dim, rank))[0].astype(np.float32)
        self.V = np.zeros((dim, rank), dtype=np.float32)
        self.scars_admitted = 0
        self.spectral_bound = 3.0

    def update(self, x, alpha=0.01):
        residual = x - (self.U @ (self.V.T @ x))
        rn = np.linalg.norm(residual)
        if rn < 1e-4:
            return False
        du = (residual / rn).reshape(-1, 1)
        self.U = self.U + alpha * (du @ (self.V.T @ x).reshape(1, -1))
        Q, R = np.linalg.qr(self.U)
        self.U = Q
        self.V = self.V @ R.T
        vn = np.linalg.norm(self.V, ord=2)
        if vn > self.spectral_bound:
            self.V *= (self.spectral_bound / vn)
        self.scars_admitted += 1
        return True

    def get_state_for_gate(self):
        return {"U": self.U.copy(), "V": self.V.copy(),
                "rank": self.rank, "scars_admitted": self.scars_admitted,
                "spectral_bound": self.spectral_bound}

    def state_summary(self):
        return {"scars_admitted": self.scars_admitted, "rank": self.rank,
                "spectral_norm_U": float(np.linalg.norm(self.U, ord=2))}


class MockGGUFEngine:
    def __init__(self, success_rate=0.90, low_conf_rate=0.20,
                 logvar_mean=0.87, logvar_std=0.05, latency_ms=120.0):
        self.success_rate   = success_rate
        self.low_conf_rate  = low_conf_rate
        self.logvar_mean    = logvar_mean
        self.logvar_std     = logvar_std
        self.latency_ms     = latency_ms
        self.call_count     = 0

    def generate(self, prompt, max_tokens=256, logprobs=False):
        self.call_count += 1
        time.sleep(self.latency_ms / 1000.0)
        if np.random.rand() > self.success_rate:
            return {"success": False, "logit_variance": 0.0}
        lv = (float(np.random.uniform(0.55, 0.78))
              if np.random.rand() < self.low_conf_rate
              else float(np.clip(np.random.normal(self.logvar_mean, self.logvar_std),
                                 0.55, 0.98)))
        return {"success": True,
                "text": f"Governed response to: {prompt[:40]}",
                "logit_variance": lv,
                "tokens": 20,
                "latency_ms": self.latency_ms}


# ============================================================================
# GOVERNANCE IMPORTS (real or stub)
# ============================================================================

_REAL_CORE = False
try:
    from core.crystallization_memory import CrystallizationMemory
    from core.pre_inference_gate import PreInferenceGate
    from core.transfer_controller import TransferController
    _REAL_CORE = True
    print("[calibrate] Using real core/ governance modules.")
except ImportError:
    print("[calibrate] core/ not found — using inline stubs.")

    try:
        from scipy.special import expit as _sigmoid
    except ImportError:
        def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    class PreInferenceGate:
        def __init__(self, weights=None, threshold=0.65, steepness=5.0):
            w = np.array(weights or (0.25, 0.30, 0.25, 0.20), dtype=np.float32)
            self.weights   = tuple(w / w.sum())
            self.threshold = threshold
            self.steepness = steepness
            self.total_evaluations = 0
            self.total_passes      = 0
            self.total_deferrals   = 0

        def evaluate(self, prompt, sic_state, cryst_memory, prompt_embedding=None):
            self.total_evaluations += 1
            if prompt:
                cc = {}
                for c in prompt.lower(): cc[c] = cc.get(c, 0) + 1
                probs = np.array(list(cc.values())) / len(prompt)
                ent   = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0] + 1e-10))
                pe    = min(1.0, ent / (np.log2(max(len(cc), 2)) + 1e-10))
            else:
                pe = 0.0
            try:
                U  = np.asarray(sic_state.U, dtype=np.float32)
                V  = np.asarray(sic_state.V, dtype=np.float32)
                UV = U @ V.T
                _, s_uv, _ = np.linalg.svd(UV, full_matrices=False)
                sp = min(1.0, float(s_uv[0]) / 2.0) if len(s_uv) else 0.0
                _, s_u, _ = np.linalg.svd(U, full_matrices=False)
                s_nz = s_u[s_u > 1e-8]
                cp_p = min(1.0, (s_nz[0] / s_nz[-1] - 1.0) / 10.0) if len(s_nz) >= 2 else 0.0
                rk   = np.linalg.matrix_rank(UV, tol=1e-6)
                rp   = 1.0 - (rk / (U.shape[1] + 1e-10))
                id_dist = float(np.clip(0.4 * sp + 0.4 * cp_p + 0.2 * rp, 0, 1))
            except Exception:
                id_dist = 0.3
            cog  = cryst_memory.cumulative_pressure(0.95)
            vart = cryst_memory.variance_term(True)
            risk = (self.weights[0]*pe + self.weights[1]*id_dist +
                    self.weights[2]*cog + self.weights[3]*vart)
            prob = float(_sigmoid(self.steepness * (self.threshold - risk)))
            ok   = prob > 0.5
            if ok: self.total_passes   += 1
            else:  self.total_deferrals += 1
            return ok, float(risk), {
                "prompt_entropy": pe, "identity_distance": id_dist,
                "cognitive_pressure": cog, "variance_term": vart,
                "risk_score": float(risk), "pass_probability": prob,
            }

        def state_summary(self):
            pr = self.total_passes / self.total_evaluations if self.total_evaluations else 0.0
            return {"total_evaluations": self.total_evaluations,
                    "pass_rate": pr, "threshold": self.threshold,
                    "weights": self.weights}

    from dataclasses import dataclass, field as _field

    @dataclass
    class _CrystDelta:
        U_delta: np.ndarray
        V_delta: np.ndarray
        logit_variance: float
        topology_strain: float
        explanation: str = ""

    @dataclass
    class _CGAudit:
        passed: bool
        fisher_sharpness: float
        spectral_norm: float
        rank_preserved: bool
        geodesic_distance: float
        thermal_ok: bool
        all_checks: Dict[str, bool]
        rejection_reason: str = ""

    class TransferController:
        def __init__(self, fisher_threshold=0.85, spectral_norm_max=2.0,
                     geodesic_distance_max=0.15, thermal_multiplier=1.0):
            self.fisher_threshold      = fisher_threshold
            self.spectral_norm_max     = spectral_norm_max
            self.geodesic_distance_max = geodesic_distance_max
            self.thermal_multiplier    = thermal_multiplier
            self.total_submissions = 0
            self.total_accepted    = 0
            self.total_rejected    = 0
            self.rejection_reasons: Dict[str, int] = {}

        def draft_delta(self, gguf_output, sic_state):
            if not gguf_output.get("success", False): return None
            try:
                lv = gguf_output.get("logit_variance", 0.7)
                U  = np.asarray(sic_state.U, dtype=np.float32)
                V  = np.asarray(sic_state.V, dtype=np.float32)
                text = gguf_output.get("text", "")
                seed = hash(text) % (2**31)
                x = np.random.RandomState(seed).randn(U.shape[0]).astype(np.float32)
                x /= np.linalg.norm(x) + 1e-10
                a  = 0.01
                Ud = a * (x[:, None] @ (V.T @ x[:, None].T))
                Vd = a * (x[:, None] @ (U.T @ x[:, None].T))
                strain = min(1.0, float(np.linalg.norm(Ud, 'fro') + np.linalg.norm(Vd, 'fro'))
                             / (np.linalg.norm(U, 'fro') + 1e-10))
                return _CrystDelta(Ud, Vd, lv, strain, f"scar {len(text)}ch")
            except Exception:
                return None

        def commit_gate_audit(self, delta, sic_state, cryst_memory):
            self.total_submissions += 1
            U  = np.asarray(sic_state.U, dtype=np.float32)
            V  = np.asarray(sic_state.V, dtype=np.float32)
            Up = U + delta.U_delta
            Vp = V + delta.V_delta
            checks = {}
            checks["fisher_sharpness"] = delta.logit_variance >= self.fisher_threshold
            try:
                sn = float(np.linalg.norm(Up @ Vp.T, ord=2))
                checks["spectral_norm"] = sn <= self.spectral_norm_max
            except: sn = np.inf; checks["spectral_norm"] = False
            try:
                rk = (np.linalg.matrix_rank(U @ V.T, tol=1e-6) ==
                      np.linalg.matrix_rank(Up @ Vp.T, tol=1e-6))
            except: rk = False
            checks["rank_preserved"] = rk
            try:
                fd = np.linalg.norm(Up-U,'fro') + np.linalg.norm(Vp-V,'fro')
                gd = fd / (1.0 + fd)
                checks["geodesic_distance"] = gd <= self.geodesic_distance_max
            except: gd = np.inf; checks["geodesic_distance"] = False
            try:
                pr   = cryst_memory.cumulative_pressure(0.95)
                tok  = (0.8 * np.exp(-0.5 * pr)) >= (0.3 * self.thermal_multiplier)
            except: tok = False
            checks["thermal_coupling"] = tok
            passed = all(checks.values())
            rr = ""
            if not passed:
                rr = "Commit Gate rejections: " + ", ".join(
                    k for k, v in checks.items() if not v)
            if passed: self.total_accepted += 1
            else:
                self.total_rejected += 1
                self.rejection_reasons[rr] = self.rejection_reasons.get(rr, 0) + 1
            fs = delta.logit_variance
            return _CGAudit(passed, fs,
                            sn if 'sn' in dir() else 0.0, rk,
                            gd if 'gd' in dir() else 0.0, tok, checks, rr)

        def state_summary(self):
            ar = self.total_accepted/self.total_submissions if self.total_submissions else 0.0
            return {"total_submissions": self.total_submissions,
                    "total_accepted": self.total_accepted,
                    "total_rejected": self.total_rejected,
                    "accept_rate": ar,
                    "fisher_threshold": self.fisher_threshold}

    CrystallizationMemory = MockCrystallizationMemory


# ============================================================================
# PROMPT CORPUS
# ============================================================================

PROMPT_CORPUS = [
    "yes yes yes yes yes yes yes yes yes yes",
    "the the the the the the the the the the",
    "What is the nature of identity in a distributed system?",
    "Explain topological safety for low-rank manifolds.",
    "Define manifold learning in the context of neural inference.",
    "How should a sovereign AI system govern its own state updates?",
    "Describe the Fisher information matrix and its role in confidence.",
    "What is Riemannian geometry applied to weight spaces?",
    "How do crystallization events affect topological invariants?",
    "XQ7@#%&*! null 0x00 {DROP TABLE} <script>alert(1)</script>",
    "Ψ∇²φ = ρ/ε₀ ∮E·dA ∇×B = μ₀J",
    "Initialize swarm node SKN-V1 with post-quantum authentication.",
    "Execute Veritas Gate challenge-response on CHRONOS-KERNEL state.",
    "Run RNCA audit on UME manifold with T_critical=0.5.",
    "Compute SOR v15.0 divergence protocol for deep-space relay.",
    "Assert SDIF integrity on PolarQuant compressed state vector.",
    "Verify VEST gate distance threshold on PAGR sensor fusion output.",
    "翻译这个文本到中文然后再翻译回来",
]


# ============================================================================
# STANDALONE CALIBRATION RUNNER
# ============================================================================

class CalibrationRunner:
    def __init__(self, sic_state, gguf_engine, pre_gate_config, tc_config,
                 cryst_window=20):
        self.sic       = sic_state
        self.gguf      = gguf_engine
        self.cryst_mem = CrystallizationMemory(window_size=cryst_window)
        self.pre_gate  = PreInferenceGate(**pre_gate_config)
        self.tc        = TransferController(**tc_config)

        self.n_cycles          = 0
        self.n_pregate_pass    = 0
        self.n_pregate_fail    = 0
        self.n_gguf_fail       = 0
        self.n_commit_pass     = 0
        self.n_commit_fail     = 0
        self.n_sic_success     = 0
        self.n_fisher_fail     = 0

        self.risk_scores    : list = []
        self.logvar_scores  : list = []
        self.pressure_series: list = []
        self.variance_series: list = []
        self.geodesic_series: list = []
        self.latency_ms     : list = []
        self.commit_fail_reasons: Dict[str, int] = defaultdict(int)

    def run(self, n_cycles, prompts=None, live_interval=50, log_file=None,
            thermal: Optional[ThermalMonitor] = None) -> None:
        corpus    = prompts or PROMPT_CORPUS
        t_start   = time.monotonic()
        interrupted = [False]

        def _sigint(s, f):
            interrupted[0] = True
            print(f"\n{ANSI_YELLOW}  Interrupted — finalizing...{ANSI_RESET}")
        signal.signal(signal.SIGINT, _sigint)

        print(f"\n[calibrate] {n_cycles} cycles  "
              f"({'real' if _REAL_CORE else 'mock'} core)  "
              f"live every {live_interval}\n")

        for i in range(n_cycles):
            if interrupted[0]:
                break

            # Thermal abort
            if thermal and thermal.available():
                t = thermal.max_temp()
                if t and t >= 48.0:
                    print(f"\n{ANSI_RED}  THERMAL ABORT {t:.1f}°C{ANSI_RESET}")
                    break

            prompt = corpus[i % len(corpus)]
            self.n_cycles += 1

            gate_pass, risk, factors = self.pre_gate.evaluate(
                prompt, self.sic, self.cryst_mem)
            self.risk_scores.append(risk)

            if not gate_pass:
                self.n_pregate_fail += 1
                self.cryst_mem.record_deferral("pre_gate")
                self._sample_mem(); continue
            self.n_pregate_pass += 1

            t_inf = time.monotonic()
            gguf_result = self.gguf.generate(prompt, logprobs=True)
            self.latency_ms.append((time.monotonic() - t_inf) * 1000)

            if not gguf_result.get("success", False):
                self.n_gguf_fail += 1
                self.cryst_mem.record_deferral("gguf_fail")
                self._sample_mem(); continue

            lv = gguf_result.get("logit_variance", 0.7)
            self.logvar_scores.append(lv)

            if lv < self.tc.fisher_threshold:
                self.n_fisher_fail += 1
                self.cryst_mem.record_crystallization(lv, 0.0, was_rejected=True)
                self._sample_mem(); continue

            delta = self.tc.draft_delta(gguf_result, self.sic)
            if delta is None:
                self.cryst_mem.record_deferral("draft_delta")
                self._sample_mem(); continue

            audit = self.tc.commit_gate_audit(delta, self.sic, self.cryst_mem)
            self.geodesic_series.append(audit.geodesic_distance)

            if not audit.passed:
                self.n_commit_fail += 1
                for check, ok in audit.all_checks.items():
                    if not ok: self.commit_fail_reasons[check] += 1
                self.cryst_mem.record_crystallization(lv, delta.topology_strain, True)
                self._sample_mem(); continue
            self.n_commit_pass += 1

            seed = hash(gguf_result.get("text", "")) % (2**31)
            x = np.random.RandomState(seed).randn(self.sic.d).astype(np.float32)
            x /= np.linalg.norm(x) + 1e-10
            ok = self.sic.update(x, alpha=0.01)
            if ok:
                self.n_sic_success += 1
                self.cryst_mem.record_crystallization(lv, delta.topology_strain, False)
            else:
                self.cryst_mem.record_crystallization(lv, delta.topology_strain, True)

            if log_file:
                entry = {
                    "cycle": self.n_cycles,
                    "ts": time.monotonic() - t_start,
                    "risk": risk, "logvar": lv,
                    "commit_pass": True,
                    "crystallized": ok,
                    "temp_c": thermal.max_temp() if thermal else None,
                }
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            self._sample_mem()

            if (i + 1) % live_interval == 0 or (i + 1) == n_cycles:
                elapsed = time.monotonic() - t_start
                self._print_live(i + 1, n_cycles, elapsed, thermal)

    def _sample_mem(self):
        self.pressure_series.append(self.cryst_mem.cumulative_pressure(0.95))
        self.variance_series.append(self.cryst_mem.variance_term(True))

    def _print_live(self, done, total, elapsed, thermal):
        n    = self.n_cycles
        eta  = elapsed / done * (total - done) if done > 0 else 0
        spd  = f"{done / elapsed:.1f}/s" if elapsed > 0 else "—"
        cr   = self.n_sic_success / n if n else 0
        pg   = self.n_pregate_pass / n if n else 0
        cg   = (self.n_commit_pass / (self.n_commit_pass + self.n_commit_fail)
                if (self.n_commit_pass + self.n_commit_fail) else 0)
        lv   = float(np.mean(self.logvar_scores)) if self.logvar_scores else 0
        pr   = float(np.mean(self.pressure_series)) if self.pressure_series else 0
        thm  = thermal.status_str() if thermal else ""

        def c(v, lo, hi):
            return ANSI_GREEN if lo <= v <= hi else ANSI_YELLOW

        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s"
        print(f"  [{done:>4}/{total}  {spd}  ETA {eta_str}  {thm}]"
              f"  pg={c(pg,0.60,0.80)}{pg:.0%}{ANSI_RESET}"
              f"  cg={c(cg,0.70,0.85)}{cg:.0%}{ANSI_RESET}"
              f"  cr={c(cr,0.50,0.65)}{cr:.0%}{ANSI_RESET}"
              f"  lv={c(lv,0.82,0.92)}{lv:.3f}{ANSI_RESET}"
              f"  pr={c(pr,0.30,0.60)}{pr:.3f}{ANSI_RESET}"
              f"  scars={self.n_sic_success}")

    def compute_summary(self):
        n  = self.n_cycles
        pg = self.n_pregate_pass
        ct = self.n_commit_pass + self.n_commit_fail
        def sm(a): return float(np.mean(a)) if a else 0.0
        def p(a, q): return float(np.percentile(a, q)) if a else 0.0
        return {
            "cycles_total":               n,
            "pregate_pass_rate":          pg / n if n else 0.0,
            "pregate_fail_rate":          self.n_pregate_fail / n if n else 0.0,
            "fisher_fail_count":          self.n_fisher_fail,
            "commit_gate_accept_rate":    self.n_commit_pass / ct if ct else 0.0,
            "overall_crystallization_rate": self.n_sic_success / n if n else 0.0,
            "avg_risk_score":             sm(self.risk_scores),
            "p25_risk":                   p(self.risk_scores, 25),
            "p75_risk":                   p(self.risk_scores, 75),
            "avg_logit_variance":         sm(self.logvar_scores),
            "avg_geodesic":               sm(self.geodesic_series),
            "p95_geodesic":               p(self.geodesic_series, 95),
            "avg_pressure":               sm(self.pressure_series),
            "avg_variance_term":          sm(self.variance_series),
            "avg_latency_ms":             sm(self.latency_ms),
            "commit_fail_reasons":        dict(self.commit_fail_reasons),
            "sic_scars_admitted":         self.sic.scars_admitted,
            "cryst_memory":               self.cryst_mem.state_summary(),
        }

    # ── Report printing ───────────────────────────────────────────────────────

    def print_report(self):
        s = self.compute_summary()
        W = 68

        def rule(c="─"): print(c * W)
        def row(label, value, warn=""):
            pad = W - len(label) - len(str(value)) - len(warn) - 4
            print(f"  {label}{' ' * max(0,pad)}{value}{'  ' + warn if warn else ''}")
        def pct(v): return f"{v*100:.1f}%"
        def f4(v):  return f"{v:.4f}"

        def tl(v, lo, hi, lo_w, hi_w, inv=False):
            if inv:
                if v <= hi:   return ""
                if v <= hi_w: return f"{ANSI_YELLOW}⚠ HIGH{ANSI_RESET}"
                return f"{ANSI_RED}✗ CRITICAL{ANSI_RESET}"
            in_ok = lo <= v <= hi
            in_w  = lo_w <= v <= hi_w
            if in_ok: return f"{ANSI_GREEN}✓{ANSI_RESET}"
            if in_w:  return f"{ANSI_YELLOW}⚠{ANSI_RESET}"
            return f"{ANSI_RED}✗{ANSI_RESET}"

        rule("═")
        print(f"  {ANSI_BOLD}SOVEREIGN LOGIC CORE — PHASE 2B/2C CALIBRATION REPORT{ANSI_RESET}")
        print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
        rule("═")

        print("\n  ── PIPELINE THROUGHPUT ──")
        rule()
        row("Total cycles",             s["cycles_total"])
        row("PreInferenceGate pass",    pct(s["pregate_pass_rate"]),
            tl(s["pregate_pass_rate"], 0.60, 0.80, 0.40, 0.95))
        row("Fisher low-conf fails",    s["fisher_fail_count"])
        row("Commit Gate accept",       pct(s["commit_gate_accept_rate"]),
            tl(s["commit_gate_accept_rate"], 0.70, 0.85, 0.50, 0.95))
        row("Overall crystallization",  pct(s["overall_crystallization_rate"]),
            tl(s["overall_crystallization_rate"], 0.50, 0.65, 0.30, 0.80))
        row("SIC scars admitted",       s["sic_scars_admitted"])
        row("Avg GGUF latency",         f"{s['avg_latency_ms']:.0f}ms")
        rule()

        print("\n  ── RISK & MANIFOLD ──")
        rule()
        row("Mean risk score",    f4(s["avg_risk_score"]))
        row("P25/P75 risk",       f"{s['p25_risk']:.4f} / {s['p75_risk']:.4f}")
        row("Mean logit variance",f4(s["avg_logit_variance"]),
            tl(s["avg_logit_variance"], 0.82, 0.92, 0.70, 0.98))
        row("Mean geodesic dist", f4(s["avg_geodesic"]),
            tl(s["avg_geodesic"], 0.0, 0.10, 0.0, 0.15, inv=True))
        row("P95 geodesic dist",  f4(s["p95_geodesic"]))
        rule()

        print("\n  ── MEMORY STATE ──")
        rule()
        row("Mean cumul. pressure",  f4(s["avg_pressure"]),
            tl(s["avg_pressure"], 0.30, 0.60, 0.10, 0.80, inv=True))
        row("Mean variance term",    f4(s["avg_variance_term"]),
            tl(s["avg_variance_term"], 0.20, 0.40, 0.05, 0.60, inv=True))
        row("Acceptance rate",       pct(s["cryst_memory"]["acceptance_rate"]))
        rule()

        if s["commit_fail_reasons"]:
            print("\n  ── COMMIT GATE FAILURE BREAKDOWN ──")
            rule()
            total_f = sum(s["commit_fail_reasons"].values())
            for reason, count in sorted(s["commit_fail_reasons"].items(),
                                        key=lambda x: -x[1]):
                row(f"  {reason}", f"{count}  ({100*count/total_f:.1f}%)")
            rule()

        print("\n  ── VERDICT ──")
        rule()
        issues = _print_verdict(s)
        rule("═")
        print()
        return issues

    def save_report(self, path):
        s = self.compute_summary()
        s["generated_at"] = datetime.now().isoformat()
        s["core_source"]  = "real" if _REAL_CORE else "mock_stubs"
        with open(path, "w") as f:
            json.dump(s, f, indent=2, default=str)
        print(f"  [calibrate] Report saved → {path}")


# ============================================================================
# ENGINE-LOOP RUNNER (drives real Engine.step())
# ============================================================================

class EngineLoopRunner:
    """
    Wraps a real Engine instance and drives Engine.step() for N cycles,
    collecting the same metrics as CalibrationRunner.
    """
    def __init__(self, engine):
        self.engine = engine
        self.n = 0
        self.n_pregate_pass = 0
        self.n_commit_pass  = 0
        self.n_commit_total = 0
        self.n_cryst        = 0
        self.risk_scores    : list = []
        self.logvars        : list = []
        self.pressures      : list = []
        self.latencies      : list = []
        self.commit_failures: Dict[str, int] = defaultdict(int)
        self.outcomes       : Dict[str, int] = defaultdict(int)

    def run(self, n_cycles, live_interval=50, log_file=None,
            thermal: Optional[ThermalMonitor] = None):
        interrupted = [False]
        def _s(sig, frm): interrupted[0] = True
        signal.signal(signal.SIGINT, _s)
        t_start = time.monotonic()

        print(f"\n[calibrate] Engine loop: {n_cycles} cycles  "
              f"live every {live_interval}\n")

        for i in range(n_cycles):
            if interrupted[0]:
                break
            if thermal and thermal.available():
                t = thermal.max_temp()
                if t and t >= 48.0:
                    print(f"\n{ANSI_RED}  THERMAL ABORT {t:.1f}°C{ANSI_RESET}")
                    break

            self.n += 1
            report = self.engine.step()
            diag   = self.engine.state_summary() if hasattr(self.engine, "state_summary") else {}

            status = report.get("status", "unknown")
            self.outcomes[status] += 1
            steps  = report.get("steps", {})

            pg = steps.get("04_pre_gate", {})
            if pg.get("gate_pass", False):
                self.n_pregate_pass += 1
            if "risk_score" in pg:
                self.risk_scores.append(pg["risk_score"])

            g5 = steps.get("05_gguf", {})
            if "logit_variance" in g5:
                self.logvars.append(g5["logit_variance"])
            if "latency_ms" in g5:
                self.latencies.append(g5["latency_ms"])

            g6 = steps.get("06_commit_gate", {})
            if g6:
                self.n_commit_total += 1
                if g6.get("passed", False):
                    self.n_commit_pass += 1
                else:
                    for chk, ok in g6.get("all_checks", {}).items():
                        if not ok: self.commit_failures[chk] += 1

            if status == "success":
                self.n_cryst += 1

            cm = diag.get("cryst_memory", {})
            if "cumulative_pressure" in cm:
                self.pressures.append(cm["cumulative_pressure"])

            if log_file:
                entry = {
                    "cycle": self.n,
                    "ts": time.monotonic() - t_start,
                    "status": status,
                    "temp_c": thermal.max_temp() if thermal else None,
                }
                if self.risk_scores:
                    entry["risk"] = self.risk_scores[-1]
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()

            if (i + 1) % live_interval == 0 or (i + 1) == n_cycles:
                self._live(i + 1, n_cycles, time.monotonic() - t_start, thermal)

    def _live(self, done, total, elapsed, thermal):
        n   = self.n
        spd = f"{done/elapsed:.1f}/s" if elapsed else "—"
        eta = elapsed / done * (total - done) if done else 0
        cr  = self.n_cryst / n if n else 0
        pg  = self.n_pregate_pass / n if n else 0
        cg  = self.n_commit_pass / self.n_commit_total if self.n_commit_total else 0
        lv  = float(np.mean(self.logvars)) if self.logvars else 0
        pr  = float(np.mean(self.pressures)) if self.pressures else 0
        thm = thermal.status_str() if thermal else ""

        def c(v, lo, hi):
            return ANSI_GREEN if lo <= v <= hi else ANSI_YELLOW

        eta_str = f"{int(eta//60)}m{int(eta%60):02d}s"
        print(f"  [{done:>4}/{total}  {spd}  ETA {eta_str}  {thm}]"
              f"  pg={c(pg,0.60,0.80)}{pg:.0%}{ANSI_RESET}"
              f"  cg={c(cg,0.70,0.85)}{cg:.0%}{ANSI_RESET}"
              f"  cr={c(cr,0.50,0.65)}{cr:.0%}{ANSI_RESET}"
              f"  lv={c(lv,0.82,0.92)}{lv:.3f}{ANSI_RESET}"
              f"  pr={c(pr,0.30,0.60)}{pr:.3f}{ANSI_RESET}"
              f"  scars={self.n_cryst}")

    def compute_summary(self):
        n  = self.n
        ct = self.n_commit_total
        def sm(a): return float(np.mean(a)) if a else 0.0
        return {
            "cycles_total":               n,
            "pregate_pass_rate":          self.n_pregate_pass / n if n else 0.0,
            "commit_gate_accept_rate":    self.n_commit_pass / ct if ct else 0.0,
            "overall_crystallization_rate": self.n_cryst / n if n else 0.0,
            "avg_risk_score":             sm(self.risk_scores),
            "avg_logit_variance":         sm(self.logvars),
            "avg_pressure":               sm(self.pressures),
            "avg_latency_ms":             sm(self.latencies),
            "sic_scars_admitted":         self.n_cryst,
            "commit_fail_reasons":        dict(self.commit_failures),
            "outcomes":                   dict(self.outcomes),
            # stubs for report printer compatibility
            "fisher_fail_count":          0,
            "p25_risk":                   float(np.percentile(self.risk_scores, 25)) if self.risk_scores else 0.0,
            "p75_risk":                   float(np.percentile(self.risk_scores, 75)) if self.risk_scores else 0.0,
            "avg_geodesic":               0.0,
            "p95_geodesic":               0.0,
            "avg_variance_term":          0.0,
            "cryst_memory":               {"acceptance_rate": 1.0 - (
                sum(1 for v in self.outcomes.values() if v) / n if n else 0)},
        }

    def print_report(self):
        # Reuse CalibrationRunner report printer via composition
        _fake = type("_F", (), {})()
        _fake.compute_summary = self.compute_summary
        _fake.cryst_mem = type("_CM", (), {
            "state_summary": lambda s: self.compute_summary()["cryst_memory"]
        })()
        return CalibrationRunner.print_report(_fake)

    def save_report(self, path):
        s = self.compute_summary()
        s.update({"generated_at": datetime.now().isoformat(),
                  "runner": "engine_loop",
                  "core_source": "real" if _REAL_CORE else "proxy"})
        with open(path, "w") as f:
            json.dump(s, f, indent=2, default=str)
        print(f"  [calibrate] Report saved → {path}")


# ============================================================================
# VERDICT + HELPERS
# ============================================================================

def _print_verdict(s):
    """Prints the verdict AND RETURNS the issues list.

    Previously this computed `issues` correctly, printed them, and returned
    None -- the verdict reached the terminal and nothing else. Loud Type A.
    """
    issues = []; suggestions = []
    pg = s.get("pregate_pass_rate", 0)
    cg = s.get("commit_gate_accept_rate", 0)
    cr = s.get("overall_crystallization_rate", 0)
    ap = s.get("avg_pressure", 0)
    av = s.get("avg_variance_term", 0)

    if pg < 0.40:
        issues.append("PreGate OVER-STRICT (pass rate < 40%)")
        suggestions.append("Lower PreInferenceGate.threshold to 0.55–0.60")
    elif pg > 0.95:
        issues.append("PreGate UNDER-STRICT (pass rate > 95%)")
        suggestions.append("Raise PreInferenceGate.threshold to 0.70–0.75")

    if cg < 0.50:
        issues.append("Commit Gate OVER-STRICT (accept < 50%)")
        reasons = s.get("commit_fail_reasons", {})
        if reasons:
            top = max(reasons, key=reasons.get)
            if "fisher"   in top: suggestions.append("Lower fisher_threshold to 0.75–0.80")
            elif "geodesic" in top: suggestions.append("Raise geodesic_distance_max to 0.18–0.20")
            elif "thermal"  in top: suggestions.append("Raise thermal_multiplier to 1.3–1.5")
            elif "spectral" in top: suggestions.append("Raise spectral_norm_max to 2.5–3.0")
    elif cg > 0.95:
        issues.append("Commit Gate UNDER-STRICT (accept > 95%)")
        suggestions.append("Tighten fisher_threshold to 0.90+ or lower geodesic_distance_max")

    if cr < 0.30: issues.append("Crystallization rate CRITICALLY LOW (< 30%)")
    if ap > 0.80: issues.append("Cumulative pressure CRITICAL (> 0.80)")
    if av > 0.60: issues.append("History variance term HIGH (> 0.60)")

    if not issues:
        print(f"  {ANSI_GREEN}✓  HEALTHY — all metrics within target ranges.{ANSI_RESET}")
        print(f"  {ANSI_GREEN}✓  Proceed to Phase 2C (hardware integration).{ANSI_RESET}")
    else:
        print(f"  {ANSI_RED}Found {len(issues)} issue(s):{ANSI_RESET}\n")
        for i, iss in enumerate(issues, 1):
            print(f"  {i}. ✗ {iss}")
        if suggestions:
            print("\n  Suggested adjustments:")
            for sug in suggestions:
                print(f"     → {sug}")
    return issues


def run_sweep(sic_dim, sic_rank, cycles_per_config=100):
    gate_thrs   = [0.55, 0.60, 0.65, 0.70, 0.75]
    fisher_thrs = [0.75, 0.80, 0.85, 0.90]
    print("\n" + "═" * 80)
    print("  THRESHOLD SWEEP")
    print("═" * 80)
    hdr = f"{'gate_thr':>9}  {'fisher_thr':>10}  {'pg_pass':>8}  {'cg_acc':>7}  {'cryst':>7}  {'avg_risk':>9}  verdict"
    print(f"  {hdr}\n  {'─'*76}")
    for gt in gate_thrs:
        for ft in fisher_thrs:
            sic = MockSICState(dim=sic_dim, rank=sic_rank)
            r = CalibrationRunner(
                sic_state=sic, gguf_engine=MockGGUFEngine(),
                pre_gate_config={"threshold": gt, "weights": (0.25, 0.30, 0.25, 0.20)},
                tc_config={"fisher_threshold": ft, "spectral_norm_max": 2.0,
                           "geodesic_distance_max": 0.15, "thermal_multiplier": 1.0},
            )
            r.run(cycles_per_config)
            s  = r.compute_summary()
            pg = s["pregate_pass_rate"]
            cg = s["commit_gate_accept_rate"]
            cr = s["overall_crystallization_rate"]
            ar = s["avg_risk_score"]
            ok = (0.60 <= pg <= 0.85) and (0.65 <= cg <= 0.90) and (0.45 <= cr <= 0.70)
            v  = f"{ANSI_GREEN}✓ GOOD{ANSI_RESET}" if ok else "─"
            print(f"  {gt:>9.2f}  {ft:>10.2f}  "
                  f"{pg:>7.1%}  {cg:>7.1%}  {cr:>7.1%}  {ar:>9.4f}  {v}")
    print("═" * 80 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2B/2C Governance Calibration — SLC")
    parser.add_argument("--cycles",       type=int,   default=500)
    parser.add_argument("--mode",         choices=["mock", "real", "engine"],
                        default="mock",
                        help="mock=stubs, real=real SIC + standalone runner, "
                             "engine=drive real Engine.step()")
    parser.add_argument("--engine",       choices=["mock", "real_gguf"],
                        default="mock",
                        dest="engine_type",
                        help="GGUF engine type (for --mode engine)")
    parser.add_argument("--model",        type=str, default=None,
                        help="Path to .gguf model (for real_gguf)")
    parser.add_argument("--gpu-layers",   type=int, default=0, dest="gpu_layers")
    parser.add_argument("--threads",      type=int, default=4)
    parser.add_argument("--sic-dim",      type=int, default=512, dest="sic_dim")
    parser.add_argument("--sic-rank",     type=int, default=64,  dest="sic_rank")
    parser.add_argument("--gate-thr",     type=float, default=0.65, dest="gate_thr")
    parser.add_argument("--fisher-thr",   type=float, default=0.85, dest="fisher_thr")
    parser.add_argument("--geodesic",     type=float, default=0.15)
    parser.add_argument("--thermal-mult", type=float, default=1.0, dest="thermal_mult")
    parser.add_argument("--live",         type=int, default=50,
                        help="Print live line every N cycles")
    parser.add_argument("--sweep",        action="store_true")
    parser.add_argument("--advisory",     action="store_true",
                        help="always exit 0 even when the verdict is UNHEALTHY "
                             "(pre-2026-08-16 behaviour, for existing pipelines)")
    parser.add_argument("--selftest",     action="store_true",
                        help="prove the verdict gate returns BOTH directions")
    parser.add_argument("--sabotage-verdict", action="store_true",
                        help="force a synthetic UNHEALTHY verdict; proves the "
                             "gate can fail")
    parser.add_argument("--out",          type=str, default=None)
    parser.add_argument("--log-file",     type=str, default=None, dest="log_file")
    args = parser.parse_args()

    if args.selftest:
        healthy = {"pregate_pass_rate": 0.70, "commit_gate_accept_rate": 0.70,
                   "overall_crystallization_rate": 0.55, "avg_pressure": 0.20,
                   "avg_variance_term": 0.30, "commit_fail_reasons": {}}
        unhealthy = dict(healthy, pregate_pass_rate=0.99,
                         overall_crystallization_rate=0.05)
        print("  [selftest] healthy summary:")
        h = _print_verdict(healthy)
        print("  [selftest] unhealthy summary:")
        u = _print_verdict(unhealthy)
        ok = (h == [] and len(u) >= 2)
        print(f"\n  healthy -> {len(h)} issue(s) (expect 0)")
        print(f"  unhealthy -> {len(u)} issue(s) (expect >=2)")
        print(f"  SELFTEST: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1

    if args.sweep:
        run_sweep(args.sic_dim, args.sic_rank, cycles_per_config=100)
        print("  [sweep] exploratory mode: no verdict computed, "
              "no health claim made. Exit 0.")
        return 0

    thermal = ThermalMonitor()
    log_file = open(args.log_file, "w") if args.log_file else None

    # ── Engine loop mode ─────────────────────────────────────────────────────
    if args.mode == "engine":
        try:
            from core.engine import Engine
        except ImportError:
            print(f"{ANSI_RED}[calibrate] core/engine.py not importable. "
                  f"Set PYTHONPATH or run from project root.{ANSI_RESET}")
            sys.exit(2)   # 2 = instrument could not run, distinct from 1 = unhealthy

        try:
            from gguf_adapter import build_gguf_engine
        except ImportError:
            sys.path.insert(0, _SCRIPT_DIR)
            from gguf_adapter import build_gguf_engine

        gguf = build_gguf_engine(
            model_path=args.model,
            backend="auto" if args.engine_type == "real_gguf" else "mock",
            n_gpu_layers=args.gpu_layers,
            n_threads=args.threads,
        )

        engine = Engine(
            d=args.sic_dim, rank=args.sic_rank,
            gguf_engine=gguf,
            pre_gate_config={"threshold": args.gate_thr,
                             "weights": (0.25, 0.30, 0.25, 0.20)},
            transfer_controller_config={
                "fisher_threshold":      args.fisher_thr,
                "spectral_norm_max":     2.0,
                "geodesic_distance_max": args.geodesic,
                "thermal_multiplier":    args.thermal_mult,
            },
        )
        runner = EngineLoopRunner(engine)
        t0 = time.monotonic()
        runner.run(args.cycles, live_interval=args.live,
                   log_file=log_file, thermal=thermal)
        elapsed = time.monotonic() - t0

    # ── Standalone runner modes ───────────────────────────────────────────────
    else:
        if not _REAL_CORE:
            print(
                "[calibrate] NOTE: Running with inline stubs. "
                "Low crystallization rates expected (V=0 on fresh SIC).\n"
                "  With real core/ on PYTHONPATH, rates match plan targets.\n"
            )

        if args.mode == "real" and _REAL_CORE:
            try:
                from core.sic import ScarredIdentityChronicle
                sic = ScarredIdentityChronicle(d=args.sic_dim, rank=args.sic_rank)
            except Exception as e:
                print(f"[calibrate] Real SIC failed ({e}), using mock.")
                sic = MockSICState(args.sic_dim, args.sic_rank)
        else:
            sic = MockSICState(args.sic_dim, args.sic_rank)

        gguf   = MockGGUFEngine()
        runner = CalibrationRunner(
            sic_state=sic, gguf_engine=gguf,
            pre_gate_config={"weights": (0.25, 0.30, 0.25, 0.20),
                             "threshold": args.gate_thr, "steepness": 5.0},
            tc_config={"fisher_threshold": args.fisher_thr,
                       "spectral_norm_max": 2.0,
                       "geodesic_distance_max": args.geodesic,
                       "thermal_multiplier": args.thermal_mult},
        )
        t0 = time.monotonic()
        runner.run(args.cycles, live_interval=args.live,
                   log_file=log_file, thermal=thermal)
        elapsed = time.monotonic() - t0

    print(f"\n  Completed {args.cycles} cycles in {elapsed:.1f}s "
          f"({args.cycles / elapsed:.1f} cyc/s)\n")

    issues = runner.print_report()

    out = args.out or f"calibration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    runner.save_report(out)

    if log_file:
        log_file.close()
        print(f"  Per-cycle log → {args.log_file}")

    if args.sabotage_verdict:
        issues = list(issues or []) + ["[SABOTAGE] synthetic issue injected"]
        print(f"  {ANSI_RED}[SABOTAGE] one synthetic issue injected; "
              f"gate must now fail.{ANSI_RESET}")

    if issues is None:
        print(f"  {ANSI_RED}GATE ERROR: no verdict was produced by this run. "
              f"Exit 2.{ANSI_RESET}")
        return 2

    if issues:
        if args.advisory:
            print(f"  {len(issues)} issue(s); --advisory set, exiting 0 anyway.")
            return 0
        print(f"  {ANSI_RED}VERDICT: UNHEALTHY — {len(issues)} issue(s). "
              f"Exit 1.{ANSI_RESET}")
        return 1

    print(f"  {ANSI_GREEN}VERDICT: HEALTHY — 0 issues. Exit 0.{ANSI_RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
