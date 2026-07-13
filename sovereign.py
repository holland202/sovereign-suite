#!/usr/bin/env python3
"""
sovereign.py — Sovereign Control Plane v0.1.2
Fixed: topology breach storm, clean harmonic convergence
Run: python3 sovereign.py
Requires: numpy (pip install numpy)
"""

import os, sys, math, sqlite3, signal, time
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

# ─── CONFIG ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    dim: int = 64
    mass: float = 1.0
    dt: float = 0.01
    eta: float = 2.0
    temp_threshold: float = 38.5
    temp_critical: float = 42.0
    dH: float = -0.1
    dS: float = 0.02
    max_steps: int = 2000
    log_interval: int = 100
    db_path: str = "sovereign_telemetry.db"
    persist_every: int = 10

# ─── THERMAL MONITOR ────────────────────────────────────────────────────

class ThermalMonitor:
    ZONES = [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/thermal/thermal_zone1/temp",
        "/sys/class/thermal/thermal_zone2/temp",
        "/sys/class/power_supply/battery/temp",
    ]
    
    def __init__(self):
        self.sim = not self._is_android()
        self._sim_temp = 35.0
        self._path = self._find_zone()
        if self._path:
            ttype = self._read_type()
            print(f"[Thermal] Hardware bound: {self._path} ({ttype})")
        else:
            print("[Thermal] Simulation mode")
    
    def _is_android(self) -> bool:
        return ("ANDROID_ROOT" in os.environ or
                os.path.exists("/system/bin/app_process") or
                "termux" in os.environ.get("PREFIX", "").lower())
    
    def _find_zone(self) -> Optional[str]:
        for p in self.ZONES:
            if os.path.exists(p):
                return p
        return None
    
    def _read_type(self) -> str:
        tpath = self._path.replace("/temp", "/type")
        try:
            with open(tpath) as f:
                return f.read().strip()
        except:
            return "unknown"
    
    def read(self) -> float:
        if self.sim or not self._path:
            self._sim_temp += 0.005 * (1 + 0.1 * np.random.randn())
            self._sim_temp = max(30.0, min(self._sim_temp, 45.0))
            return self._sim_temp
        try:
            with open(self._path) as f:
                v = int(f.read().strip())
            return v / 10.0 if "battery" in self._path else v / 1000.0
        except:
            self.sim = True
            return self.read()

# ─── THERMODYNAMIC FUNCTOR ──────────────────────────────────────────────

class ThermoFunctor:
    def __init__(self, cfg: Config, monitor: ThermalMonitor):
        self.cfg = cfg
        self.mon = monitor
        self.commits = 0
        self.rejects = 0
    
    def compute(self) -> Tuple[float, float, bool, float]:
        T = self.mon.read()
        if T <= self.cfg.temp_threshold:
            aT = 1.0
        else:
            aT = math.exp(-self.cfg.eta * (T - self.cfg.temp_threshold))
        dG = self.cfg.dH - T * self.cfg.dS
        gate = dG < 0
        if gate:
            self.commits += 1
        else:
            self.rejects += 1
        return aT, dG, gate, T

# ─── EPISTEMIC POTENTIAL ────────────────────────────────────────────────

class Potential:
    """Clean harmonic well. Guaranteed convex, simply-connected."""
    def __init__(self, dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.truth = rng.normal(0, 1, dim)
        self.truth /= np.linalg.norm(self.truth)
    
    def grad(self, X: np.ndarray) -> np.ndarray:
        return X - self.truth
    
    def V(self, X: np.ndarray) -> float:
        return 0.5 * np.linalg.norm(X - self.truth) ** 2

# ─── MANIFOLD STATE ─────────────────────────────────────────────────────

class State:
    def __init__(self, cfg: Config, pot: Potential):
        self.cfg = cfg
        self.pot = pot
        rng = np.random.default_rng(42)
        self.q = rng.normal(0, 0.5, cfg.dim)
        self.p = np.zeros(cfg.dim)
        self.history = deque(maxlen=1000)
        self.violations = 0
        self.consecutive_breaches = 0
        self.step = 0
    
    def check_topology(self) -> bool:
        if len(self.history) < 10:
            self.history.append(self.q.copy())
            return True
        
        for past in self.history:
            if np.linalg.norm(self.q - past) < 1e-3:
                if self.pot.V(self.q) >= self.pot.V(past) - 1e-6:
                    self.violations += 1
                    self.consecutive_breaches += 1
                    if self.consecutive_breaches >= 3:
                        self._recover()
                    return False
        
        self.consecutive_breaches = 0
        self.history.append(self.q.copy())
        return True
    
    def _recover(self):
        """Break a non-contractible loop by perturbing state and clearing history."""
        rng = np.random.default_rng()
        perturb = rng.normal(0, 0.05, self.cfg.dim)
        self.q += perturb
        self.p *= 0.1
        self.history.clear()
        self.consecutive_breaches = 0
    
    def K(self) -> float:
        return 0.5 * np.dot(self.p, self.p) / self.cfg.mass

# ─── SYMPLECTIC INTEGRATOR ──────────────────────────────────────────────

class Integrator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
    
    def step(self, state: State, force: np.ndarray, gate: bool, aT: float):
        dt = self.cfg.dt
        m = self.cfg.mass
        
        if not gate or aT < 1e-12:
            F = np.zeros_like(force)
            state.p *= 0.99
        else:
            F = force * aT
        
        p_half = state.p + 0.5 * dt * F
        state.q = state.q + dt * p_half / m
        state.p = p_half + 0.5 * dt * F
        state.step += 1

# ─── SOVEREIGN CORE ─────────────────────────────────────────────────────

class SovereignCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.thermal = ThermalMonitor()
        self.thermo = ThermoFunctor(cfg, self.thermal)
        self.pot = Potential(cfg.dim)
        self.state = State(cfg, self.pot)
        self.integrator = Integrator(cfg)
        
        self.conn = sqlite3.connect(cfg.db_path)
        self.cursor = self.conn.cursor()
        self._init_db()
        
        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)
        
        self.Vs = []
        self.Ks = []
        self.Ts = []
        self.gates = []
    
    def _init_db(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                step INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                temperature REAL NOT NULL,
                potential REAL NOT NULL,
                kinetic REAL NOT NULL,
                hamiltonian REAL NOT NULL,
                gate INTEGER NOT NULL,
                violations INTEGER NOT NULL,
                aT REAL NOT NULL,
                state_norm REAL NOT NULL,
                state_vector BLOB
            )
        """)
        self.conn.commit()
    
    def _persist(self, step: int, T: float, aT: float, gate: bool):
        V = self.pot.V(self.state.q)
        K = self.state.K()
        H = K + V
        state_blob = self.state.q.astype(np.float64).tobytes()
        
        self.cursor.execute(
            """INSERT INTO telemetry
               (step, timestamp, temperature, potential, kinetic, hamiltonian,
                gate, violations, aT, state_norm, state_vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                step, time.time(), T, float(V), float(K), float(H),
                int(gate), self.state.violations, float(aT),
                float(np.linalg.norm(self.state.q)), state_blob,
            )
        )
        self.conn.commit()
    
    def _shutdown(self, signum, frame):
        print("\n[SHUTDOWN] Signal received. Closing SQLite.")
        self.conn.close()
        sys.exit(0)
    
    def run(self):
        print("=" * 50)
        print("SOVEREIGN CONTROL PLANE v0.1.2")
        print(f"DB: {self.cfg.db_path}")
        print("=" * 50)
        
        for i in range(self.cfg.max_steps):
            aT, dG, gate, T = self.thermo.compute()
            
            if T >= self.cfg.temp_critical:
                print(f"\n[ATOMIC_REDUCTION] T={T:.2f}°C — logic frozen")
                self._persist(i, T, aT, gate)
                break
            
            force = -self.pot.grad(self.state.q)
            
            if not self.state.check_topology():
                print(f"\n[VERITAS] Topology breach at step {i} — recovering")
                self.state.p *= 0.5
                continue
            
            self.integrator.step(self.state, force, gate, aT)
            
            self.Vs.append(self.pot.V(self.state.q))
            self.Ks.append(self.state.K())
            self.Ts.append(T)
            self.gates.append(gate)
            
            if i % self.cfg.persist_every == 0:
                self._persist(i, T, aT, gate)
            
            if i % self.cfg.log_interval == 0:
                H = self.Ks[-1] + self.Vs[-1]
                s = "COMMIT" if gate else "REJECT"
                print(f"Step {i:04d} | T={T:5.2f}°C | a={aT:.4f} | "
                      f"H={H:8.4f} | V={self.Vs[-1]:8.4f} | {s}")
            
            if np.linalg.norm(self.state.q - self.pot.truth) < 1e-3:
                print(f"\n[CONVERGED] Truth reached at step {i}")
                self._persist(i, T, aT, gate)
                break
        
        self._persist(self.state.step, T, aT, gate)
        
        print("=" * 50)
        print(f"Steps: {self.state.step} | Commits: {self.thermo.commits} | "
              f"Rejects: {self.thermo.rejects} | Topo violations: {self.state.violations}")
        print(f"Final V: {self.Vs[-1]:.6f} | Dist to truth: "
              f"{np.linalg.norm(self.state.q - self.pot.truth):.6f}")
        print(f"Telemetry rows: {self.cursor.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]}")
        print("=" * 50)
        
        self.conn.close()
        
        return {
            "V": np.array(self.Vs),
            "K": np.array(self.Ks),
            "T": np.array(self.Ts),
            "gates": np.array(self.gates),
        }

# ─── ASCII PLOTTER ────────────────────────────────────────────────────────

def plot(y, title, w=50, h=8):
    if len(y) == 0:
        return
    mn, mx = y.min(), y.max()
    if mx == mn:
        mx += 1
    print(f"\n{title}")
    print("-" * w)
    for row in range(h):
        thr = mx - (row + 0.5) * (mx - mn) / h
        line = "".join(
            "*" if y[i] >= thr else " "
            for i in range(0, len(y), max(1, len(y) // w))
        )
        print(line)
    print("-" * w)

# ─── MAIN ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    cfg = Config(dim=dim, max_steps=2000, log_interval=100)
    core = SovereignCore(cfg)
    m = core.run()
    
    plot(m["V"][:500], "Potential V(X)")
    plot(m["T"], "Temperature T(t)")
    plot(m["K"][:500], "Kinetic Energy K(t)")
