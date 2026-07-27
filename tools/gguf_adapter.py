#!/usr/bin/env python3
"""
gguf_adapter.py
================
Universal GGUF Engine Adapter — Sovereign Logic Core
Wraps llama-cpp-python, raw llama.cpp subprocess, or any compatible
inference backend into the single interface Engine and CalibrationRunner
expect:

    engine.generate(prompt, max_tokens, logprobs) -> {
        "success":        bool,
        "text":           str,
        "logit_variance": float,   # Fisher-estimate of output confidence
        "tokens":         int,     # token count
        "latency_ms":     float,
    }

Usage:
    # Auto-detect best available backend:
    engine = build_gguf_engine(model_path="models/mistral-7b-q4.gguf")

    # Explicit backend:
    engine = LlamaCppPythonEngine("models/phi3-mini.gguf", n_gpu_layers=99)
    engine = LlamaCppSubprocessEngine("/data/data/com.termux/llama.cpp/main",
                                      "models/mistral-7b-q4.gguf")
    engine = MockGGUFEngine(success_rate=0.90)  # for tests without a model

All engines implement BaseSLCEngine and return the same dict schema.
The LogitVarianceEstimator converts raw logprob arrays → scalar logit_variance.
"""

import os
import re
import sys
import json
import time
import logging
import subprocess
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

logger = logging.getLogger("slc.gguf_adapter")


# ============================================================================
# BASE INTERFACE
# ============================================================================

class BaseSLCEngine(ABC):
    """
    Minimal contract every GGUF backend must satisfy.

    The critical field is `logit_variance`:
      - Ideally: empirical variance of per-token logit distributions
        (Var[softmax(z_t)] averaged over generated tokens).
      - Acceptable: top-token probability mean (proxy; less precise).
      - Fallback: fixed value (0.80) — disables Fisher discrimination.

    Fisher-sharpness in the Commit Gate is:
        F_t = logit_variance  (already scaled to [0,1])
    The gate passes when F_t >= fisher_threshold (default 0.85).
    A miscalibrated engine that always returns 0.80 will cause ~100%
    Fisher failures. Use LogitVarianceEstimator to calibrate the proxy.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        logprobs: bool = True,
    ) -> Dict[str, Any]:
        """
        Run inference and return structured result.

        Returns dict with keys:
            success        (bool)   — False if engine error
            text           (str)    — generated text (empty on failure)
            logit_variance (float)  — Fisher estimate in [0,1]
            tokens         (int)    — tokens generated
            latency_ms     (float)  — wall-clock time in ms
        """

    def warmup(self, n: int = 3) -> None:
        """Run n throwaway prompts to warm JIT / KV cache."""
        for i in range(n):
            self.generate("Hello", max_tokens=8, logprobs=False)
        logger.info(f"[{self.__class__.__name__}] Warmup complete ({n} prompts)")

    def health_check(self) -> bool:
        """Return True if engine can produce a response."""
        try:
            r = self.generate("Test", max_tokens=4)
            return r.get("success", False)
        except Exception:
            return False


# ============================================================================
# LOGIT VARIANCE ESTIMATOR
# ============================================================================

class LogitVarianceEstimator:
    """
    Converts raw logprob data from llama.cpp into a scalar logit_variance.

    llama-cpp-python can return logprobs as a list of dicts like:
        [{"token": "▁Hello", "logprob": -0.12, "top_logprobs": {...}}, ...]

    Strategy (in order of preference):
        1. If top_logprobs present: compute variance of top-k softmax dist
        2. If only token logprob: use exp(logprob) as token confidence,
           accumulate variance across tokens
        3. Fallback: fixed 0.80

    Output is always clipped to [0.55, 0.98] to keep it in a physically
    meaningful range (pure noise = 0.55, near-certain = 0.98).
    """

    CLIP_LO = 0.55
    CLIP_HI = 0.98

    @classmethod
    def from_logprob_list(
        cls,
        logprob_data: Optional[List[Dict]],
        fallback: float = 0.80,
    ) -> float:
        """
        Args:
            logprob_data: List of per-token logprob dicts from llama-cpp-python
            fallback: Value to return if logprob_data is None/empty
        Returns:
            float in [CLIP_LO, CLIP_HI]
        """
        if not logprob_data:
            return fallback

        token_vars = []
        token_probs = []

        for tok in logprob_data:
            if tok is None:
                continue

            top = tok.get("top_logprobs") or tok.get("top_log_probs")
            lp  = tok.get("logprob")

            if top and isinstance(top, dict) and len(top) >= 2:
                # Strategy 1: variance of top-k softmax distribution
                log_vals = np.array(list(top.values()), dtype=np.float32)
                log_vals = log_vals - log_vals.max()          # numerical stability
                probs    = np.exp(log_vals)
                probs   /= probs.sum() + 1e-10
                token_vars.append(float(np.var(probs)))
            elif lp is not None:
                # Strategy 2: treat exp(logprob) as "certainty"
                token_probs.append(float(np.exp(np.clip(lp, -20, 0))))

        if token_vars:
            # Higher variance of softmax dist → more confident distribution
            # Map: avg_var ∈ [0, 0.25] → logit_variance ∈ [0.55, 0.98]
            raw = float(np.mean(token_vars))
            lv  = 0.55 + min(raw / 0.20, 1.0) * 0.43
        elif token_probs:
            # Mean token probability ∈ [0, 1] → rescale to [0.55, 0.98]
            raw = float(np.mean(token_probs))
            lv  = 0.55 + raw * 0.43
        else:
            lv = fallback

        return float(np.clip(lv, cls.CLIP_LO, cls.CLIP_HI))

    @classmethod
    def from_token_probs(cls, probs: np.ndarray, fallback: float = 0.80) -> float:
        """
        From a flat array of per-token top-1 probabilities.
        Useful when logprob_data isn't available but token probs are.
        """
        if probs is None or len(probs) == 0:
            return fallback
        raw = float(np.mean(np.clip(probs, 0.0, 1.0)))
        lv  = 0.55 + raw * 0.43
        return float(np.clip(lv, cls.CLIP_LO, cls.CLIP_HI))


# ============================================================================
# BACKEND 1: llama-cpp-python
# ============================================================================

class LlamaCppPythonEngine(BaseSLCEngine):
    """
    Wraps llama-cpp-python (pip install llama-cpp-python).

    Recommended install for Snapdragon 8 Elite (Adreno GPU offload):
        CMAKE_ARGS="-DLLAMA_VULKAN=on" pip install llama-cpp-python --upgrade

    Or CPU-only (slower but stable):
        pip install llama-cpp-python

    Args:
        model_path:    Absolute path to .gguf model file
        n_gpu_layers:  Number of layers to offload to Adreno GPU (0 = CPU only)
        n_ctx:         Context length (default 2048)
        n_threads:     CPU threads (default 4, Snapdragon 8 Elite has 8 perf cores)
        temperature:   Sampling temperature (default 0.7)
        verbose:       llama.cpp verbosity (False = quiet)
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.7,
        verbose: bool = False,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed.\n"
                "Install: pip install llama-cpp-python --break-system-packages\n"
                "GPU:     CMAKE_ARGS='-DLLAMA_VULKAN=on' "
                "pip install llama-cpp-python --break-system-packages"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model: {model_path} (gpu_layers={n_gpu_layers})")
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=verbose,
            logits_all=False,      # we use logprobs param in generate()
        )
        self.temperature  = temperature
        self.model_path   = model_path
        self._call_count  = 0
        logger.info("Model loaded.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        logprobs: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        self._call_count += 1

        try:
            resp = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
                logprobs=5 if logprobs else None,  # top-5 logprobs per token
                echo=False,
            )

            choices = resp.get("choices", [])
            if not choices:
                return self._fail(t0)

            choice  = choices[0]
            text    = choice.get("text", "").strip()
            n_toks  = resp.get("usage", {}).get("completion_tokens", len(text.split()))

            # Extract logit_variance
            lp_data = choice.get("logprobs")
            if lp_data and isinstance(lp_data, dict):
                raw_list = lp_data.get("token_logprobs") or []
                # Build list of dicts for estimator
                top_lps  = lp_data.get("top_logprobs") or [None] * len(raw_list)
                lp_dicts = [
                    {"logprob": lp, "top_logprobs": top}
                    for lp, top in zip(raw_list, top_lps)
                    if lp is not None
                ]
                lv = LogitVarianceEstimator.from_logprob_list(lp_dicts)
            else:
                lv = 0.80  # fallback if logprobs not available

            return {
                "success":        True,
                "text":           text,
                "logit_variance": lv,
                "tokens":         n_toks,
                "latency_ms":     (time.monotonic() - t0) * 1000,
            }

        except Exception as e:
            logger.error(f"LlamaCppPythonEngine.generate() error: {e}")
            return self._fail(t0)

    @staticmethod
    def _fail(t0: float) -> Dict[str, Any]:
        return {
            "success": False, "text": "", "logit_variance": 0.0,
            "tokens": 0, "latency_ms": (time.monotonic() - t0) * 1000,
        }


# ============================================================================
# BACKEND 2: llama.cpp subprocess (direct binary)
# ============================================================================

class LlamaCppSubprocessEngine(BaseSLCEngine):
    """
    Calls the llama.cpp `main` (or `llama-cli`) binary via subprocess.
    No Python bindings required — useful when only the binary is available
    (e.g. Termux pre-compiled package).

    Termux install:
        pkg install llama-cpp

    Args:
        binary_path:   Path to llama.cpp binary (e.g. /data/data/com.termux/files/usr/bin/llama-cli)
        model_path:    Path to .gguf model
        n_gpu_layers:  GPU offload layers (0 = CPU)
        n_ctx:         Context window
        n_threads:     CPU threads
        temperature:   Sampling temperature
        timeout_s:     Per-call timeout in seconds (default 60)

    NOTE: Subprocess mode cannot expose token-level logprobs easily.
    logit_variance is estimated from output token count and response length
    as a rough proxy. For production use, prefer LlamaCppPythonEngine.
    """

    def __init__(
        self,
        binary_path: str,
        model_path: str,
        n_gpu_layers: int = 0,
        n_ctx: int = 2048,
        n_threads: int = 4,
        temperature: float = 0.7,
        timeout_s: float = 60.0,
    ):
        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"llama.cpp binary not found: {binary_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.binary_path  = binary_path
        self.model_path   = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx        = n_ctx
        self.n_threads    = n_threads
        self.temperature  = temperature
        self.timeout_s    = timeout_s
        self._call_count  = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        logprobs: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        self._call_count += 1

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(self.temperature),
            "-t", str(self.n_threads),
            "--ctx-size", str(self.n_ctx),
            "--no-mmap",
            "--silent-prompt",
            "-ngl", str(self.n_gpu_layers),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )

            text = result.stdout.strip()
            if result.returncode != 0 or not text:
                logger.warning(f"subprocess exited {result.returncode}: {result.stderr[:200]}")
                return self._fail(t0)

            # Estimate logit_variance from output characteristics
            # (crude proxy: longer coherent output → higher confidence)
            n_words = len(text.split())
            # Map token count to logit_variance: 5–50 words → 0.72–0.88
            lv = float(np.clip(0.72 + (n_words - 5) / 200.0, 0.72, 0.88))

            return {
                "success":        True,
                "text":           text,
                "logit_variance": lv,
                "tokens":         n_words,
                "latency_ms":     (time.monotonic() - t0) * 1000,
            }

        except subprocess.TimeoutExpired:
            logger.warning("Subprocess timeout")
            return self._fail(t0)
        except Exception as e:
            logger.error(f"Subprocess error: {e}")
            return self._fail(t0)

    @staticmethod
    def _fail(t0: float) -> Dict[str, Any]:
        return {
            "success": False, "text": "", "logit_variance": 0.0,
            "tokens": 0, "latency_ms": (time.monotonic() - t0) * 1000,
        }


# ============================================================================
# BACKEND 3: Mock (testing / calibration without a model)
# ============================================================================

class MockGGUFEngine(BaseSLCEngine):
    """
    Stochastic mock engine for testing without a real model.
    Models realistic failure distribution:
      - success_rate:   fraction of calls that succeed
      - low_conf_rate:  fraction of successes with logit_variance below typical threshold
    """

    PROMPTS_RESPONSES = [
        ("identity",    "Identity is defined by continuity of state across perturbation."),
        ("governance",  "Self-governance requires a membrane between inference and state."),
        ("manifold",    "A Riemannian manifold admits a metric tensor at each tangent space."),
        ("topological", "Topological invariants are preserved under continuous deformation."),
        ("default",     "The system maintains coherence through constrained crystallization."),
    ]

    def __init__(
        self,
        success_rate: float = 0.90,
        low_conf_rate: float = 0.20,
        logvar_mean: float = 0.87,
        logvar_std: float = 0.05,
        latency_ms_mean: float = 150.0,
    ):
        self.success_rate    = success_rate
        self.low_conf_rate   = low_conf_rate
        self.logvar_mean     = logvar_mean
        self.logvar_std      = logvar_std
        self.latency_ms_mean = latency_ms_mean
        self._call_count     = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        logprobs: bool = True,
    ) -> Dict[str, Any]:
        t0 = time.monotonic()
        self._call_count += 1

        # Simulate latency
        lat = float(np.abs(np.random.normal(self.latency_ms_mean, 30)))
        time.sleep(lat / 1000.0)

        if np.random.rand() > self.success_rate:
            return {
                "success": False, "text": "", "logit_variance": 0.0,
                "tokens": 0, "latency_ms": lat,
            }

        # Choose response
        prompt_lower = prompt.lower()
        text = next(
            (r for k, r in self.PROMPTS_RESPONSES if k in prompt_lower),
            self.PROMPTS_RESPONSES[-1][1],
        )

        lv = (float(np.random.uniform(0.55, 0.78))
              if np.random.rand() < self.low_conf_rate
              else float(np.clip(np.random.normal(self.logvar_mean, self.logvar_std), 0.55, 0.98)))

        return {
            "success":        True,
            "text":           text,
            "logit_variance": lv,
            "tokens":         len(text.split()),
            "latency_ms":     lat,
        }


# ============================================================================
# AUTO-DETECTION FACTORY
# ============================================================================

def build_gguf_engine(
    model_path: Optional[str] = None,
    backend: str = "auto",
    n_gpu_layers: int = 0,
    n_threads: int = 4,
    n_ctx: int = 2048,
    temperature: float = 0.7,
    llama_binary: Optional[str] = None,
) -> BaseSLCEngine:
    """
    Build the best available GGUF engine.

    Args:
        model_path:    Path to .gguf model. If None, returns MockGGUFEngine.
        backend:       "auto" | "llama_cpp_python" | "subprocess" | "mock"
        n_gpu_layers:  Adreno GPU offload layers (0 = CPU)
        n_threads:     CPU threads for inference
        n_ctx:         Context window size
        temperature:   Sampling temperature
        llama_binary:  Path to llama.cpp binary (for subprocess backend)

    Returns:
        Configured BaseSLCEngine instance.

    Auto-detection order:
        1. llama-cpp-python (if importable and model_path given)
        2. llama.cpp binary (if binary found and model_path given)
        3. MockGGUFEngine (fallback)
    """
    if backend == "mock" or model_path is None:
        logger.info("Using MockGGUFEngine (no model path provided)")
        return MockGGUFEngine()

    if backend in ("auto", "llama_cpp_python"):
        try:
            import llama_cpp  # noqa: F401
            logger.info(f"Auto-selected: LlamaCppPythonEngine (gpu_layers={n_gpu_layers})")
            return LlamaCppPythonEngine(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=n_threads,
                temperature=temperature,
            )
        except ImportError:
            if backend == "llama_cpp_python":
                raise
            logger.info("llama-cpp-python not available, trying subprocess...")

    if backend in ("auto", "subprocess"):
        # Common Termux paths for llama.cpp binary
        candidates = [
            llama_binary,
            "/data/data/com.termux/files/usr/bin/llama-cli",
            "/data/data/com.termux/files/usr/bin/llama-cpp",
            "/data/data/com.termux/files/usr/bin/main",
            os.path.expanduser("~/llama.cpp/main"),
            os.path.expanduser("~/llama.cpp/build/bin/main"),
        ]
        binary = next((p for p in candidates if p and os.path.exists(p)), None)
        if binary:
            logger.info(f"Auto-selected: LlamaCppSubprocessEngine ({binary})")
            return LlamaCppSubprocessEngine(
                binary_path=binary,
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                temperature=temperature,
            )
        if backend == "subprocess":
            raise FileNotFoundError(
                "No llama.cpp binary found. Checked:\n" +
                "\n".join(f"  {p}" for p in candidates if p)
            )

    logger.warning("No GGUF backend available — falling back to MockGGUFEngine")
    return MockGGUFEngine()
