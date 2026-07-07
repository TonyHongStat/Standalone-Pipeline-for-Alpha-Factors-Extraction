#!/usr/bin/env python3
"""
GTJA191 end-to-end demo script.

Runnable from terminal or JupyterLab via:
    %run notebooks/gtja191_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from gtja191 import GTJA191Engine, N_FACTORS, load_sample_universe  # noqa: E402
from gtja191.backtest import FactorAnalyzer, forward_return, load_backtest_config  # noqa: E402
from gtja191.data import AkShareLoader  # noqa: E402
from gtja191.data.synthetic import make_synthetic_panel  # noqa: E402
from gtja191.factors import list_implemented  # noqa: E402
from gtja191.pipeline import FactorRegenerator  # noqa: E402

START = "2023-01-01"
END = "2024-12-31"
USE_AKSHARE = True
SYMBOLS = load_sample_universe()

DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
FACTOR_DIR = DATA_DIR / "factors"


def load_panel():
    if USE_AKSHARE:
        try:
            loader = AkShareLoader(cache_dir=DATA_DIR / "akshare_cache")
            panel = loader.load(SYMBOLS, START, END, adjust="qfq")
            print("Loaded AkShare panel:", {k: v.shape for k, v in panel.items()})
            return panel
        except Exception as exc:
            print(f"AkShare failed: {exc}; using synthetic data.")
    panel = make_synthetic_panel(SYMBOLS, START, END)
    print("Synthetic panel:", {k: v.shape for k, v in panel.items()})
    return panel


def main():
    print(f"GTJA191 target={N_FACTORS}, implemented={list_implemented()}")
    panel = load_panel()

    regen = FactorRegenerator(output_dir=FACTOR_DIR, cache_dir=MARKET_DIR)
    manifest = regen.run(universe=SYMBOLS, start=START, end=END, factors="all", panel=panel)
    print("Regeneration manifest:", manifest)

    engine = GTJA191Engine(panel)
    preview = engine.compute([1, 2, 171, 191], return_format="dict")
    for fid, df in preview.items():
        print(f"\nalpha_{fid:03d} tail:\n", df.iloc[-3:].dropna(how="all"))

    cfg = load_backtest_config()
    cfg["n_quintiles"] = 3
    analyzer = FactorAnalyzer(FACTOR_DIR, MARKET_DIR, config=cfg)
    for fid in [1, 2, 171, 191]:
        rep = analyzer.evaluate(fid, start=START, end=END)
        print(
            f"alpha_{fid:03d}: IC={rep.ic.ic_mean:.4f} IR={rep.ic.ir:.4f} "
            f"spread={rep.quintile_spread:.4f}"
        )

    h = int(cfg.get("forward_return_horizon", 5))
    fwd = forward_return(panel["open"], panel["close"], horizon=h)
    print(f"\nForward return (H={h}) last 5 days mean:", fwd.iloc[-5:].mean(axis=1).to_dict())


if __name__ == "__main__":
    main()
