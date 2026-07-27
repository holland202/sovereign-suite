# Phase 2C tools

Real-hardware tooling for the Sovereign Logic Core, Galaxy S25 Ultra under Termux.

| File | Gate |
|---|---|
| `smoke_test_2c.py` | Yes. `--selftest` proves the verdict discriminates (12/12). Exits 1 on out-of-band rates or abort, 130 on Ctrl+C. |
| `gguf_adapter.py` | None. Raises on failure; no self-test. |
| `calibrate_governance.py` | **None.** No verdict logic; always exits 0 except on ImportError. |
| `engine_diagnostic_patch.py` | None. Prints patch instructions only. |

Health bands used by the smoke test: crystallization 0.45-0.70,
pregate 0.60-0.85, commit-accept 0.65-0.90.

Run `python3 tools/smoke_test_2c.py --selftest` before trusting a result.
