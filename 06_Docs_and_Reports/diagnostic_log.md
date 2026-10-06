# Diagnostic Log — Risk-Aware RL Trading Bot

All agent/pipeline failures are logged here following the triage template in `AGENTS.md §7`.

---

## Incident #1

```
DATE: 2025-10
TASK: Fetch VN30 price data via vnstock
LAYER FAILED: Data Collector
SYMPTOM: ImportError — vnstock top-level API (e.g. `from vnstock import stock_historical_data`) deprecated
ROOT CAUSE: vnstock library underwent breaking API refactor; old top-level functions removed
FIX APPLIED: Migrated all imports to vnstock.api.quote.Quote and set VNSTOCK_TELEMETRY=off
TEST ADDED: test_collector_import() — verifies vnstock.api.quote.Quote is importable
PREVENTION: Added vnstock import pattern to AGENTS.md §1 "Known vnstock Gotchas"
```

---

## Incident #2

```
DATE: 2025-10
TASK: Download all 30 VN30 tickers
LAYER FAILED: Data Collector
SYMPTOM: HTTP 429 Too Many Requests after ~5-6 tickers
ROOT CAUSE: vnstock rate limit is ~20 requests/minute; no delay between requests
FIX APPLIED: Added time.sleep(3.5) between ticker downloads + skip_existing=True guard
TEST ADDED: test_skip_existing() — mocks API, verifies no second call when file exists
PREVENTION: Rate limit documented in AGENTS.md §1 and Collector DoD §6.1
```

---

<!-- Add new incidents below this line -->
