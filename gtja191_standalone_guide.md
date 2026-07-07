# Guide to `gtja191_standalone.ipynb`

Self-contained GP factor discovery on **~64 liquid A-shares** via OpenBB/Yahoo Finance.

**Run:** Kernel → Restart & Run All (~15–25 min)

---

## Key settings (Cell 2)

| Setting | Value | Purpose |
|---------|-------|---------|
| `SYMBOLS` | 64 liquid A-shares | Meaningful cross-sectional ranks |
| `TRAIN_END` / `TEST_START` | `2024-01-01` | GP on train; OOS on test |
| `MIN_LOADED` | 40 | Min tickers required from OpenBB |
| `POP_SIZE` / `GENERATIONS` | 60 / 12 | GP size tuned for 60+ stocks |
| `MIN_CS_NAMES` | 20 | Min stocks per day for IC |
| `MAX_ZERO_FRAC` | 0.25 | Reject degenerate (mostly-zero) factors |
| `CORR_THRESH` | 0.65 | Stricter decorrelation |
| `OOS_MIN_IR` / `OOS_IR_DECAY_MIN` | 0.10 / 0.40 | OOS effectiveness filters |

---

## Steps

| Step | Cell | ~Time |
|------|------|-------|
| 1 | Load OpenBB + train/test split | 2–5 min |
| 2 | GP evolve (**train only**) | 1–15 min |
| 3 | Select 191 diverse alphas | 1–20 min |
| 4 | In-sample catalog + deduped ranking | < 10 s |
| 5 | In-sample charts | < 10 s |
| 6 | **OOS evaluation** (`oos_df`, `effective_oos`) | < 1 min |
| 7 | OOS charts (cum IC, IS vs OOS IR) | < 10 s |

---

## Ranking effective factors

1. **`effective_is`** (Step 4) — dedupe by correlation, rank by composite score on **train** stats.
2. **`effective_oos`** (Step 6) — filter on OOS IR, positive OOS IC, and `ir_decay = oos_ir / is_ir`.

Tune thresholds in Cell 2 if too few/many factors pass OOS filters.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `< MIN_LOADED` tickers | Some Yahoo symbols missing; notebook falls back to synthetic |
| Slow Step 3 | `select_diverse` compares every candidate pairwise — normal for 191 target |
| No OOS factors pass | Loosen `OOS_MIN_IR` / `OOS_IR_DECAY_MIN`, or extend test window |
| Offline | `DATA_PROVIDER = "synthetic"` |
