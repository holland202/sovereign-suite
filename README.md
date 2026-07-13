# Sovereign Suite

A deterministic, physics-grounded AI control plane that runs locally on Snapdragon 8 Elite (Samsung Galaxy S25 Ultra via Termux). The system does not predict tokens; it computes minimum-energy trajectories through a continuous logic manifold bounded by thermodynamics and topology.

## Core Equation

```
d/dt(m Ẋ) = -∇V(X) · 1_{ΔG < 0} · a(T)
```

Where:
- **-∇V(X)** pulls logic toward verified truth (harmonic well)
- **1_{ΔG < 0}** is the thermodynamic gate — state only commits if Gibbs free energy decreases
- **a(T)** is the thermal collapse factor — compute velocity scales with hardware temperature

## Quick Start (Termux)

```bash
# Install numpy
pip install numpy

# Run the sovereign control plane
python3 sovereign.py

# Or with custom dimensionality
python3 sovereign.py 128
```

## Architecture

| Component | Role | File |
|-----------|------|------|
| **Symplectic Integrator** | Velocity Verlet (2nd order), prevents energy drift | `sovereign.py` |
| **Thermal Monitor** | Reads Android thermal sysfs; falls back to simulation | `sovereign.py` |
| **Thermodynamic Functor** | Computes ΔG = ΔH - TΔS; gates state commits | `sovereign.py` |
| **Epistemic Potential** | Harmonic well V(X) = ½‖X - truth‖² | `sovereign.py` |
| **Manifold State** | Tracks X(t), enforces β₁ = β₂ = 0 (no logic holes) | `sovereign.py` |
| **SQLite Persistence** | File-first memory; BLOB state vectors every 10 steps | `sovereign.py` |

## Telemetry

After running, query the SQLite database:

```bash
sqlite3 sovereign_telemetry.db "SELECT step, temperature, potential, violations FROM telemetry LIMIT 5;"
```

Reconstruct a state vector:

```bash
python3 -c "
import sqlite3, numpy as np
conn = sqlite3.connect("sovereign_telemetry.db")
c = conn.cursor()
c.execute('SELECT step, state_vector FROM telemetry WHERE step=100')
row = c.fetchone()
if row:
    vec = np.frombuffer(row[1], dtype=np.float64)
    print(f"Step {row[0]}: norm={np.linalg.norm(vec):.4f}, dim={len(vec)}")
"
```

## Thermal Binding

The system reads hardware temperature from `/sys/class/thermal/thermal_zone*/temp`. On the S25 Ultra, this binds to the `aoss-0` sensor. If temperature exceeds 42°C, the **ATOMIC_REDUCTION** protocol triggers: logic freezes, SQLite flushes, and the process protects the silicon.

## Topology Enforcement

The **Veritas Constraint** enforces β₁ = β₂ = 0 (simply connected manifold). The system detects if the state trajectory forms a non-contractible loop — a mathematical definition of hallucination. After 3 consecutive breaches, the state auto-recovers via orthogonal perturbation.

## Files

| File | Description |
|------|-------------|
| `sovereign.py` | Main control plane (v0.1.2) |
| `sovereign_dashboard.py` | ANSI terminal dashboard (real-time telemetry) |
| `dashboard.html` | Browser-based Canvas dashboard with live charts |
| `export_telemetry.py` | SQLite-to-JSON exporter for HTML dashboard |
| `kernel_verify.py` | SHA-256 intrinsic verification gate for binary auditing |
| `polytope_explorer.html` | 4D quantum polytope explorer (Three.js) |
| `verify_polytopes.py` | Coordinate verification for 24-cell, 600-cell, 120-cell |
| `verify_quantum_claims.py` | Quantum group annotation verification |

## Hardware

- **Device:** Samsung Galaxy S25 Ultra
- **SoC:** Snapdragon 8 Elite
- **RAM:** 12GB LPDDR5X
- **OS:** Android 15 + Termux

## License

MIT
