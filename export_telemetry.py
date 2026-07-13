#!/usr/bin/env python3
"""
export_telemetry.py — Exports SQLite telemetry to JSON for dashboard.html
Run: python3 export_telemetry.py
Serves: telemetry.json (polled by dashboard.html)
"""
import sqlite3, json, os, time

DB = "sovereign_telemetry.db"
OUT = "telemetry.json"

def export():
    if not os.path.exists(DB):
        return
    try:
        conn = sqlite3.connect(DB)
        c = conn.cursor()
        c.execute("SELECT step, temperature, potential, kinetic, hamiltonian, gate, violations, aT, state_norm FROM telemetry ORDER BY step")
        rows = c.fetchall()
        data = []
        for r in rows:
            data.append({
                "step": r[0], "temperature": r[1], "potential": r[2],
                "kinetic": r[3], "hamiltonian": r[4], "gate": bool(r[5]),
                "violations": r[6], "aT": r[7], "state_norm": r[8]
            })
        with open(OUT, "w") as f:
            json.dump(data, f)
        conn.close()
    except:
        pass

if __name__ == "__main__":
    while True:
        export()
        time.sleep(1)
