# Sovereign Suite v0.2.1

Deterministic, physics-grounded AI control plane running locally on Snapdragon 8 Elite (Samsung Galaxy S25 Ultra via Termux).

> **Core principle:** The system does not "predict" tokens; it computes minimum-energy trajectories through a continuous logic manifold bounded by thermodynamics and topology.

---

## Core Equation

```
d/dt(m Ẋ) = -∇V(X) · 1_{ΔH < 0} · a(T)
```

Where:
- **-∇V(X)** pulls logic toward verified truth (harmonic well)
- **1_{ΔH < 0}** is the thermodynamic gate — state only commits if potential energy strictly decreases
- **a(T)** is the thermal telemetry factor — displays thermal stress, does not scale momentum

---

## Quick Start (Termux)

```bash
pip install numpy
python3 sovereign.py
```

Expected clean convergence:
```
[CONVERGED] Truth reached at step ~320
Final V: 0.000000 | Dist to truth: 0.000031
Telemetry rows: 2
```

---

## Architecture

| Component | Role | File |
|-----------|------|------|
| **Damped Verlet Integrator** | 2nd-order symplectic with physical damping γ=0.05 | `sovereign.py` |
| **Thermal Monitor** | Reads Android thermal sysfs (68 zones on S25 Ultra); falls back to simulation | `sovereign.py` |
| **Thermodynamic Gate** | Strict ΔH < 0 criterion; ΔG computed for telemetry only | `sovereign.py` |
| **Epistemic Potential** | Harmonic well V(X) = ½‖X - truth‖², noise-free | `sovereign.py` |
| **Topology Checker** | Sampled every 5 steps; cooling reset after 10 breaches | `sovereign.py` |
| **SQLite Persistence** | File-first memory; BLOB state vectors | `sovereign.py` |

---

## Bug Fix History

### v0.1.1 → v0.2.0
- **Topology breach storm**: Removed `sin(4πr)` noise from gradient
- **Gate accepted uphill steps**: Enforced strict `ΔH < 0` criterion
- **Loop counter inflation**: Steps only increment on commits
- **Topology checker storm**: Sample every 5 steps + cooling reset

### v0.2.0 → v0.2.1
- **Thermal cold-amplification**: `exp(-η(T-38.5))` gave `aT=35×` at `T=15°C`, causing kinetic energy to explode to ~8000. Fixed by removing thermal scaling from momentum entirely. `aT` is now telemetry-only.
- **Slow convergence at cold temps**: Replaced post-step friction with physical damping `γ=0.05` in the Verlet force term. Converges in ~320 steps at all temperatures below 42°C.

---

## Verified Behavior

| Temperature | aT (telemetry) | Convergence | Breaches |
|-------------|----------------|-------------|----------|
| 15°C | 1.000 | ~320 steps | 0 |
| 25°C | 1.000 | ~320 steps | 0 |
| 38.5°C | 1.000 | ~320 steps | 0 |
| 41.5°C | 0.638 | ~320 steps | 0 |
| ≥42°C | 0.000 | ATOMIC_REDUCTION | — |

---

## Telemetry

```bash
sqlite3 sovereign_telemetry.db "SELECT step, temperature, potential, aT, violations FROM telemetry LIMIT 5;"
```

---

## Files

| File | Description |
|------|-------------|
| `sovereign.py` | Main control plane (v0.2.1) |
| `sovereign_dashboard.py` | ANSI terminal dashboard |
| `dashboard.html` | Browser-based Canvas dashboard |
| `export_telemetry.py` | SQLite-to-JSON exporter |
| `kernel_verify.py` | SHA-256 intrinsic verification |
| `polytope_explorer.html` | 4D quantum polytope explorer (Three.js) |

---

## Hardware

- **Device:** Samsung Galaxy S25 Ultra
- **SoC:** Snapdragon 8 Elite
- **RAM:** 12GB LPDDR5X
- **OS:** Android 15 + Termux

---

## License

MIT
