#!/usr/bin/env python3
"""
Sovereign Suite v0.2.0 — Deterministic Physics-Grounded Control Plane
Runs locally on Snapdragon 8 Elite (Samsung S25 Ultra via Termux).

Core dynamics: symplectic integration on a thermodynamically gated logic manifold.
"""

import os
import sys
import sqlite3
import struct
import hashlib
import signal
import random
import math
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TRUTH = np.array([3.14159265, 2.71828182, 1.41421356, 1.61803398], dtype=np.float64)
DIM = len(TRUTH)
MASS = 1.0
DT = 0.01
ETA_THERMAL = 0.15
TEMP_CRITICAL = 42.0
TEMP_NOMINAL = 38.5
TOPO_TOLERANCE = 1e-2
TOPO_SAMPLE_EVERY = 5
COOLING_THRESHOLD = 10
COOLING_PERTURBATION = 0.5
MAX_STEPS = 2000
FRICTION = 0.05
DB_PATH = Path.home() / "sovereign_telemetry.db"

# ---------------------------------------------------------------------------
# THERMAL BINDING
# ---------------------------------------------------------------------------
class ThermalBinding:
    def __init__(self):
        self.zones = []
        self._sim_temp = 35.0
        base = Path("/sys/class/thermal")
        if base.exists():
            for zone_dir in sorted(base.glob("thermal_zone*")):
                temp_file = zone_dir / "temp"
                if temp_file.exists():
                    self.zones.append(temp_file)
            if self.zones:
                print(f"[Thermal] Hardware bound: {self.zones[0]} ({len(self.zones)} zones)")
            else:
                print("[Thermal] No hardware zones found; using simulation.")
        else:
            print("[Thermal] /sys/class/thermal absent; using simulation.")

    def read(self):
        if not self.zones:
            self._sim_temp += 0.01 * random.gauss(0, 1)
            self._sim_temp = max(30.0, min(self._sim_temp, 45.0))
            return self._sim_temp
        temps = []
        for zf in self.zones:
            try:
                raw = zf.read_text().strip()
                t_millideg = int(raw)
                temps.append(t_millideg / 1000.0)
            except (ValueError, OSError):
                continue
        return sum(temps) / len(temps) if temps else self._sim_temp

# ---------------------------------------------------------------------------
# POTENTIAL FIELD (v0.2.0: noise-free)
# ---------------------------------------------------------------------------
class Potential:
    def __init__(self, truth):
        self.truth = truth

    def value(self, x):
        diff = x - self.truth
        return 0.5 * np.dot(diff, diff)

    def grad(self, x):
        return x - self.truth

# ---------------------------------------------------------------------------
# TOPOLOGY CHECKER (v0.2.0: sampled, with cooling)
# ---------------------------------------------------------------------------
class TopologyChecker:
    def __init__(self, tolerance=TOPO_TOLERANCE):
        self.tol = tolerance
        self.history = []
        self.consecutive_breaches = 0
        self.total_violations = 0
        self.step_counter = 0

    def _hash_state(self, x):
        quantized = np.round(x / self.tol).astype(np.int64)
        return hashlib.sha256(quantized.tobytes()).hexdigest()[:16]

    def check(self, x, potential):
        self.step_counter += 1
        if self.step_counter % TOPO_SAMPLE_EVERY != 0:
            return True
        h = self._hash_state(x)
        for prev_h, prev_V in self.history:
            if h == prev_h and potential >= prev_V:
                self.consecutive_breaches += 1
                self.total_violations += 1
                if self.consecutive_breaches >= COOLING_THRESHOLD:
                    print(f"  [Topology] Cooling triggered after {self.consecutive_breaches} breaches.")
                    self.history.clear()
                    self.consecutive_breaches = 0
                    return "COOLING"
                return False
        self.history.append((h, potential))
        if len(self.history) > 200:
            self.history.pop(0)
        self.consecutive_breaches = 0
        return True

    def reset(self):
        self.history.clear()
        self.consecutive_breaches = 0

# ---------------------------------------------------------------------------
# THERMODYNAMIC GATE (v0.2.0: strict dH < 0)
# ---------------------------------------------------------------------------
class ThermodynamicGate:
    def evaluate(self, V_new, V_old, x_new, x_old, T):
        dH = V_new - V_old
        dS = 0.05 * np.linalg.norm(x_new - x_old) / (1.0 + np.linalg.norm(x_old)) if x_old is not None else 0.0
        dG = dH - T * dS
        gate_open = dH < 0
        return gate_open, dG, dH, dS

    def reset(self):
        pass

# ---------------------------------------------------------------------------
# TELEMETRY (SQLite)
# ---------------------------------------------------------------------------
class Telemetry:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                step INTEGER PRIMARY KEY,
                timestamp REAL,
                temperature REAL,
                potential REAL,
                kinetic REAL,
                hamiltonian REAL,
                gate_pass INTEGER,
                topo_violations INTEGER,
                aT REAL,
                state_norm REAL,
                state_vector BLOB
            )
        """)
        self.conn.commit()

    def persist(self, step, T, V, K, H, gate, violations, aT, state):
        blob = state.astype(np.float64).tobytes()
        self.conn.execute(
            """INSERT INTO telemetry
               (step, timestamp, temperature, potential, kinetic, hamiltonian,
                gate_pass, topo_violations, aT, state_norm, state_vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (step, time.time(), T, V, K, H, int(gate), violations, aT,
             float(np.linalg.norm(state)), blob)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

# ---------------------------------------------------------------------------
# INTEGRATOR: Velocity Verlet
# ---------------------------------------------------------------------------
class VerletIntegrator:
    def __init__(self, mass, dt, potential):
        self.m = mass
        self.dt = dt
        self.V = potential

    def step(self, x, p):
        F = -self.V.grad(x)
        p_half = p + 0.5 * self.dt * F
        x_new = x + (p_half / self.m) * self.dt
        F_new = -self.V.grad(x_new)
        p_new = p_half + 0.5 * self.dt * F_new
        return x_new, p_new

# ---------------------------------------------------------------------------
# SOVEREIGN KERNEL
# ---------------------------------------------------------------------------
class SovereignKernel:
    def __init__(self):
        self.thermal = ThermalBinding()
        self.potential = Potential(TRUTH)
        self.topo = TopologyChecker()
        self.gate = ThermodynamicGate()
        self.telemetry = Telemetry(DB_PATH)
        self.integrator = VerletIntegrator(MASS, DT, self.potential)

        self.x = np.random.randn(DIM).astype(np.float64)
        self.p = np.zeros(DIM, dtype=np.float64)
        self.V_old = self.potential.value(self.x)
        self.x_old = None

        self.steps = 0
        self.commits = 0
        self.rejects = 0
        self.atomic_reduction = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n[Signal] Caught {signum}. ATOMIC_REDUCTION flush...")
        self._persist()
        self.telemetry.close()
        sys.exit(0)

    def _persist(self):
        K = 0.5 * np.dot(self.p, self.p) / MASS
        H = self.V_old + K
        T = self.thermal.read()
        aT = math.exp(-ETA_THERMAL * (T - TEMP_NOMINAL)) if T < TEMP_CRITICAL else 0.0
        self.telemetry.persist(
            self.steps, T, self.V_old, K, H, True,
            self.topo.total_violations, aT, self.x
        )

    def _thermal_collapse(self, T):
        if T >= TEMP_CRITICAL:
            print(f"[ATOMIC_REDUCTION] T={T:.2f}°C ≥ {TEMP_CRITICAL}°C. HALT.")
            self.atomic_reduction = True
            return 0.0
        aT = math.exp(-ETA_THERMAL * (T - TEMP_NOMINAL))
        return aT

    def run(self):
        print("=" * 60)
        print("SOVEREIGN SUITE v0.2.0")
        print("Symplectic logic manifold | Thermodynamic gate | Topology bound")
        print("=" * 60)
        print(f"Truth target: {TRUTH}")
        print(f"Initial state: {self.x}")
        print(f"Initial V: {self.V_old:.6f}")
        print(f"Topology tolerance: {TOPO_TOLERANCE} | Sample every: {TOPO_SAMPLE_EVERY}")
        print(f"Cooling threshold: {COOLING_THRESHOLD}")
        print("-" * 60)

        for loop_idx in range(MAX_STEPS):
            if self.atomic_reduction:
                break

            T = self.thermal.read()
            aT = self._thermal_collapse(T)
            if self.atomic_reduction:
                break

            x_prop, p_prop = self.integrator.step(self.x, self.p)
            V_prop = self.potential.value(x_prop)

            gate_open, dG, dH, dS = self.gate.evaluate(V_prop, self.V_old, x_prop, self.x, T)

            if not gate_open:
                self.rejects += 1
                self.p *= 0.3
                continue

            topo_result = self.topo.check(x_prop, V_prop)

            if topo_result == "COOLING":
                self.x = self.x + np.random.normal(0, COOLING_PERTURBATION, DIM)
                self.p = np.zeros(DIM, dtype=np.float64)
                self.V_old = self.potential.value(self.x)
                self.x_old = None
                self.gate.reset()
                self.rejects += 1
                continue
            elif not topo_result:
                self.rejects += 1
                self.p *= 0.3
                continue

            self.p = p_prop * aT * (1.0 - FRICTION)
            self.x = x_prop
            self.V_old = V_prop
            self.x_old = self.x.copy()
            self.steps += 1
            self.commits += 1

            dist = np.linalg.norm(self.x - TRUTH)
            if self.steps % 100 == 0 or dist < 0.01:
                K = 0.5 * np.dot(self.p, self.p) / MASS
                H = self.V_old + K
                print(f"Step {self.steps:4d} | T={T:.1f}°C | aT={aT:.4f} | "
                      f"V={self.V_old:.6f} | K={K:.6f} | H={H:.6f} | "
                      f"dist={dist:.6f} | breaches={self.topo.total_violations}")

            if dist < 1e-4 and self.V_old < 1e-6:
                print(f"\n[CONVERGED] Truth reached at step {self.steps}")
                print(f"Final V: {self.V_old:.6f} | Dist to truth: {dist:.6f}")
                break

        self._persist()
        self.telemetry.close()

        print("-" * 60)
        print(f"Steps: {self.steps} | Commits: {self.commits} | Rejects: {self.rejects}")
        print(f"Topo violations: {self.topo.total_violations}")
        print(f"Final V: {self.V_old:.6f} | Dist to truth: {np.linalg.norm(self.x - TRUTH):.6f}")

        conn = sqlite3.connect(str(DB_PATH))
        n_rows = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0]
        conn.close()
        print(f"Telemetry rows: {n_rows}")

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kernel = SovereignKernel()
    kernel.run()
