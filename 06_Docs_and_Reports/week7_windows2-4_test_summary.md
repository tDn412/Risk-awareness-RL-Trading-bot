# Walk-forward windows 2–4: PPO A0–A3 out-of-sample results

Protocol: same as window 1 (`ppo_baseline.yaml`, 500k steps, best-on-validation checkpoint), but **one seed (42)** per
reward, so the ± columns below are 0 and single-run noise is not averaged out. Seeds 101 and 2024 are still to run.
All strategies except the index go through `VN30PortfolioEnv` (T+2, ±7% band, 100-share lots, 0.15% fee, 0.1% tax).

Reproduce:
```bash
python 03_Model_Development/run_experiments.py --config ppo_baseline.yaml --rewards A0 A1 A2 A3 \
    --windows window_2 window_3 window_4 --seeds 42 --parallel 1
python 04_Backtesting_Evaluation/backtester.py --window window_2   # and window_3, window_4
```

## Headline (test-year total return %, max drawdown % in brackets)

| Strategy | W2 test 2023 | W3 test 2024 | W4 test 2025 |
|---|---|---|---|
| PPO A0 | -12.1 (25.4) | -14.9 (22.1) | -8.2 (23.3) |
| PPO A1 | -22.9 (35.9) | -2.1 (12.9) | 9.0 (19.7) |
| PPO A2 | -8.0 (21.1) | 5.4 (9.1) | 21.7 (17.9) |
| PPO A3 | -3.2 (19.5) | 7.6 (8.8) | 22.1 (18.4) |
| VN30 Index (B&H) | 8.0 (17.5) | 18.8 (8.4) | 51.2 (16.2) |
| Equal Weight 1/N | 8.7 (17.6) | 15.7 (9.1) | 39.6 (17.7) |
| Static 60/40 | 5.4 (10.8) | 9.4 (5.5) | 23.6 (10.6) |

## Findings

1. **No PPO run beats any baseline out of sample**, in any window. Every run also loses to its own Equivalent Equity
   Benchmark (EEB) with p < 0.05 (paired t-test), so H3 (skill beyond de-risking) is not supported.
2. **Transaction costs explain most of the gap.** Agents trade 10–34% of NAV per day, costing 9–33% of NAV per year in
   fees and tax, against ~1% for their EEB. The return gap to EEB is roughly the friction figure.
3. **Risk-aware rewards help, consistently.** A2 (drawdown barrier) and A3 (combined) beat A0 in every window, mainly
   because they learned to trade less (turnover ~0.10–0.16 vs ~0.26–0.32 for A0). A3 was the best agent in all three
   windows, and in window 3 its drawdown (8.8%) matched the index (8.4%).
4. **Agents stay almost fully invested** (cash weight 1–5%), so drawdowns are not lower than the index; the
   "MDD ≤ 50% of VN30" KPI is not met.

## Suggested next steps

- Penalise turnover directly in the reward (or add a no-trade band / rebalance only every k days) before spending
  compute on more seeds; friction is the dominant term.
- Then run seeds 101 and 2024 for windows 2–4 so results are mean ± std across 3 seeds like window 1.

Full tables: `outputs/backtest_results/window_{2,3,4}_test_summary.md`; validation-set metrics:
`outputs/validation_summary_w234.md`; figures: `outputs/figures/window_{2,3,4}_test_nav.png`.
