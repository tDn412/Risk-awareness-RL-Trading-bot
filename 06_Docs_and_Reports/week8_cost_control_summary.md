# Cutting trading costs: turnover penalty and partial rebalancing

Follow-up to `week7_windows2-4_test_summary.md`, where every PPO run lost to its baselines mainly because it
traded 10–34% of NAV per day (9–33% of NAV per year in fees and tax).

## What was added (both off by default, so earlier results are unchanged)

- **Turnover penalty** `reward.lambda_turnover` (λ): every reward mode subtracts `λ · turnover` per step.
- **Partial rebalancing** `env.rebalance_speed` (κ): each day the portfolio moves only a fraction κ of the way from its
  current weights to the agent's target. Baselines and EEB in the backtester still rebalance fully (κ = 1).

CLI: `train.py --lambda-turnover 0.03 --rebalance-speed 0.1`, or `run_experiments.py --lambda-turnovers ... --rebalance-speed ...`.

## Tuning (validation set only: window 2, validation year 2022, A3, seed 42)

| Setting | Val. return % | Val. Sharpe | Val. MDD % | Avg daily turnover |
|---|---|---|---|---|
| no penalty (κ = 1, λ = 0) | -37.0 | -1.47 | 45.3 | 0.134 |
| λ = 0.03 | -34.9 | -1.39 | 43.7 | 0.116 |
| λ = 0.3 | -32.7 | -1.31 | 41.6 | 0.082 |
| κ = 0.25 | -28.8 | -1.24 | 36.9 | 0.078 |
| κ = 0.25, λ = 0.03 | -20.1 | -0.83 | 33.4 | 0.034 |
| κ = 0.1 | -18.4 | -0.95 | 29.7 | 0.038 |
| **κ = 0.1, λ = 0.03 (chosen)** | **-21.0** | **-0.70** | 32.6 | 0.040 |

The penalty alone barely changes behaviour: final-checkpoint turnover stays at 0.2–0.3 even at λ = 0.3, because most
churn comes from the policy's day-to-day noise, which the reward cannot train away. Partial rebalancing removes it
mechanically. The test years were not used for this choice. The sweep checkpoints were moved to `outputs/models/sweep/`
so they don't appear in the backtest tables.

## Out-of-sample results (seed 42, 500k steps, best-on-validation checkpoint)

Total return % (Sharpe) per test year; friction is fees + tax as % of NAV per year.

| Strategy | W2 test 2023 | W3 test 2024 | W4 test 2025 | Friction %/yr |
|---|---|---|---|---|
| PPO A2 (before) | -8.0 (-0.39) | 5.4 (0.46) | 21.7 (1.06) | 10–15 |
| PPO A3 (before) | -3.2 (-0.11) | 7.6 (0.61) | 22.1 (1.07) | 9–12 |
| **PPO A2, κ 0.1, λ 0.03** | -1.4 (0.00) | 9.7 (0.76) | 38.7 (1.65) | 1.2–3.3 |
| **PPO A3, κ 0.1, λ 0.03** | -1.1 (0.01) | **16.6 (1.15)** | 36.1 (1.58) | 1.6–4.3 |
| Equal Weight 1/N | 8.7 (0.58) | 15.7 (1.11) | 39.6 (1.68) | 0.3 |
| VN30 Index (B&H) | 8.0 (0.53) | 18.8 (1.29) | 51.2 (1.99) | – |

## Findings

1. Costs fell 3–8x (2.8x for A3 in 2023) and returns rose by 2–17 percentage points in every window, for both A2 and A3.
2. Cost-aware A3 in 2024 beat Equal Weight (16.6% vs 15.7%) and its own EEB (16.6% vs 11.8%, p = 0.28, not
   significant). This is the first agent run in the project to beat an investable baseline out of sample.
3. The agents still trail the VN30 index in every year and Equal Weight in 2023 and 2025, and none of the remaining gaps
   to EEB is significant except A2 in 2024, which trails its EEB (p = 0.03). With one seed, the 2024 result could be luck.

## Next steps

- Run seeds 101 and 2024 for the cost-aware A2/A3 on windows 2–4 to see whether the 2024 result holds.
- Re-run window 1 (test 2022) with the cost-aware setting so all four windows are comparable.
