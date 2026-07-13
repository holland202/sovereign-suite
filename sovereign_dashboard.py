#!/usr/bin/env python3
"""
sovereign_dashboard.py — ANSI Terminal Dashboard for Sovereign Control Plane
Run: python3 sovereign_dashboard.py [db_path] [refresh_rate]
Requires: numpy (pip install numpy)
"""

import os, sys, time, sqlite3
import numpy as np

CLEAR = "\033[2J\033[H"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"

def sparkline(values, width=48):
    if len(values) == 0:
        return " " * width
    mn, mx = min(values), max(values)
    if mx == mn:
        mx = mn + 1
    blocks = " ▁▂▃▄▅▆▇█"
    result = []
    step = max(1, len(values) // width)
    for i in range(0, min(len(values), width * step), step):
        idx = int((values[i] - mn) / (mx - mn) * 8)
        result.append(blocks[max(0, min(8, idx))])
    return "".join(result).ljust(width)

def gauge(value, max_val, width=40, label="", color_fn=None):
    ratio = min(1.0, max(0.0, value / max_val))
    filled = int(width * ratio)
    bar = "█" * filled + "░" * (width - filled)
    if color_fn:
        bar = color_fn(bar, ratio)
    return f"{label:12s} │{bar}│ {ratio*100:5.1f}%"

def thermal_color(bar, ratio):
    if ratio < 0.5:
        return GREEN + bar + RESET
    elif ratio < 0.8:
        return YELLOW + bar + RESET
    else:
        return RED + bar + RESET

class Dashboard:
    def __init__(self, db_path="sovereign_telemetry.db", rate=0.5):
        self.db_path = db_path
        self.rate = rate
        self.conn = None

    def _connect(self):
        if os.path.exists(self.db_path):
            try:
                self.conn = sqlite3.connect(self.db_path)
                return True
            except:
                pass
        return False

    def _fetch(self, limit=200):
        if not self.conn:
            return None
        c = self.conn.cursor()
        c.execute("""SELECT step, temperature, potential, kinetic, hamiltonian,
                     gate, violations, aT, state_norm FROM telemetry
                     ORDER BY step DESC LIMIT ?""", (limit,))
        rows = c.fetchall()
        if not rows:
            return None
        rows.reverse()
        return rows

    def _stats(self):
        if not self.conn:
            return None
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*), AVG(temperature), MIN(potential), MAX(violations) FROM telemetry")
        return c.fetchone()

    def _draw(self, data, stats):
        if not data:
            print(f"{CLEAR}{CYAN}{BOLD}SOVEREIGN DASHBOARD{RESET}\n{DIM}Waiting for telemetry...{RESET}")
            return

        steps = [r[0] for r in data]
        temps = [r[1] for r in data]
        pots = [r[2] for r in data]
        kins = [r[3] for r in data]
        gates = [r[5] for r in data]
        vios = [r[6] for r in data]
        aTs = [r[7] for r in data]
        norms = [r[8] for r in data]

        s, T, V, K, H, g, v, aT, n = data[-1]

        t_state = GREEN + "NOMINAL" + RESET if T < 38.5 else (YELLOW + "WARM" + RESET if T < 42.0 else RED + BG_RED + WHITE + "CRITICAL" + RESET)
        g_state = GREEN + "COMMIT" + RESET if g else RED + "REJECT" + RESET

        lines = [
            f"{CLEAR}{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════╗{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {WHITE}{BOLD}SOVEREIGN CONTROL PLANE{RESET}  {DIM}v0.1.2{RESET}                    {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════╣{RESET}",
            f"{CYAN}{BOLD}║{RESET}  Step {BOLD}{s:05d}{RESET}  │  T={T:5.2f}°C  │  a(T)={aT:.4f}  │  {g_state}      {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  Thermal: {t_state}  │  Violations: {v:4d}  │  Norm: {n:.4f}      {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════╣{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {gauge(T, 48.0, 40, 'Temperature', thermal_color)}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {gauge(aT, 1.0, 40, 'a(T) Collapse', lambda b,r: CYAN+b+RESET)}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {gauge(abs(V), 5.0, 40, 'Potential V', lambda b,r: MAGENTA+b+RESET)}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {gauge(K, 2.0, 40, 'Kinetic K', lambda b,r: BLUE+b+RESET)}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════╣{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {DIM}V(X){RESET}  {sparkline(pots)}  {RESET}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {DIM}T(t){RESET}  {sparkline(temps)}  {RESET}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {DIM}K(t){RESET}  {sparkline(kins)}  {RESET}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {DIM}a(T){RESET}  {sparkline(aTs)}  {RESET}        {CYAN}{BOLD}║{RESET}",
            f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════╣{RESET}",
            f"{CYAN}{BOLD}║{RESET}  {DIM}Gate{RESET}  {''.join(GREEN+'●'+RESET if g else RED+'○'+RESET for g in gates[-48:]).ljust(48)}  {RESET}        {CYAN}{BOLD}║{RESET}",
        ]

        if stats:
            lines.append(f"{CYAN}{BOLD}╠══════════════════════════════════════════════════════════════╣{RESET}")
            lines.append(f"{CYAN}{BOLD}║{RESET}  {DIM}Rows: {stats[0]}  │  Avg T: {stats[1]:.2f}°C  │  Min V: {stats[2]:.4f}  │  Max Vio: {stats[3]}{RESET}  {CYAN}{BOLD}║{RESET}")

        lines.extend([
            f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════╝{RESET}",
            f"{DIM}Press Ctrl+C to exit. Refresh: {self.rate}s{RESET}"
        ])

        print("\n".join(lines))

    def run(self):
        print(HIDE_CURSOR, end="")
        try:
            while True:
                if not self.conn:
                    self._connect()
                self._draw(self._fetch(), self._stats())
                time.sleep(self.rate)
        except KeyboardInterrupt:
            pass
        finally:
            print(SHOW_CURSOR, end="")
            if self.conn:
                self.conn.close()
            print(f"\n{DIM}Dashboard closed.{RESET}")

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "sovereign_telemetry.db"
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    Dashboard(db, rate).run()
