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

## Incident #3

```
DATE: 2026-10-14
TASK: Align 30 VN30 tickers to common trading calendar (Phase 1 clean_alignment)
LAYER FAILED: Data Processor
SYMPTOM: ValueError: Ticker BCM exceeds max gap limit (5 days): detected gap of 7 days between 2020-08-20 and 2020-08-28.
ROOT CAUSE: Structural UBCKNN exchange-transfer halts (UPCoM to HOSE) created valid trading halts of 6-9 days (POW 9d, BCM 7d, VIB 7d, GVR 6d) during official listing transitions.
FIX APPLIED: Configured MAX_ALLOWED_GAP_DAYS = 10 to accommodate legitimate exchange transfers while guarding against gaps > 10 days; implemented is_listed mask for late-listed stocks (SSB, TCB).
TEST ADDED: test_pre_listing_not_ffilled(), test_gap_ffill_short(), test_gap_raises_long() in tests/test_clean_alignment.py.
PREVENTION: Documented exchange-transfer transition threshold in clean_alignment.py and roadmap.
```

---

<!-- Add new incidents below this line -->

## Incident #4

```
DATE: 2026-10-22
TASK: Set up Python environment and run VN30 data pipeline
LAYER FAILED: Infrastructure / Data Collector
SYMPTOM: pip: "No matching distribution found for vnstock>=4.0.0"; installing vnstock from GitHub then fails on "No matching distribution found for vnai>=2.6.0" (both return 404 on PyPI).
ROOT CAUSE: vnstock and its mandatory dependency vnai are no longer published on PyPI; vnstock/__init__.py imports vnai unconditionally.
FIX APPLIED: Added 02_Data_Pipeline/collectors/vci_client.py, a direct client for the same Vietcap endpoint vnstock uses (trading.vietcap.com.vn/api/chart/OHLCChart/gap-chart). fetch_vn30_prices.py and fetch_benchmarks.py fall back to it when vnstock cannot be imported. Corporate events (vnstock-only) skipped; VCI prices are already split/dividend adjusted.
TEST ADDED: none (network client); pipeline run verified 30 tickers x 1,999 days written to data/processed/.
PREVENTION: Remove vnstock from requirements.txt or pin a GitHub install once vnai is available again; keep VCI fallback.
```

## Incident #5

```
DATE: 2026-10-22
TASK: Run full pytest suite after adding training tests
LAYER FAILED: Infrastructure (Windows)
SYMPTOM: OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed — while importing torch during test collection.
ROOT CAUSE: On Windows, torch DLLs fail to initialise if pandas/pyarrow native libs are loaded first in the same process.
FIX APPLIED: tests/conftest.py imports torch before any other module.
TEST ADDED: tests/test_train.py (exercises torch import after env/pandas tests).
PREVENTION: Keep the torch import at the top of conftest.py; train.py imports torch before pandas-dependent modules.
```

## Incident #6

```
DATE: 2026-10-22
TASK: Run existing test suite on fresh environment
LAYER FAILED: Processor (test)
SYMPTOM: tests/test_clean_alignment.py::test_output_schema — assert StringDtype == object (pandas 3.0 default 'str' dtype).
ROOT CAUSE: requirements.txt allows pandas>=2.0; pandas 3 infers a dedicated string dtype for text columns.
FIX APPLIED: Assertion uses pd.api.types.is_string_dtype(), valid on pandas 2 and 3.
TEST ADDED: existing test fixed.
PREVENTION: Prefer pandas type predicates over dtype equality in tests.
```
