# SpecVel — Spectral Velocity Engine

Apply the spectral velocity formula to any financial or macro time series.
Plug in a data source, get LONG/SHORT/NEUTRAL signals and anomaly flags out.

---

## What it does

Spectral velocity measures the **rate of change of a smoothed signal** —
originally developed for satellite NDVI crop monitoring, now applied to
prices, yields, spreads, macro indicators, and geopolitical fundamentals.

For any time series it produces:

- **Velocity score** — how fast is this series moving vs its own history
- **LONG/SHORT/NEUTRAL signal** — conviction-scored directional signal
- **Phase label** — green_up / peak / senescence / dormant
- **Anomaly flag** — is the current pattern outside the historical envelope
- **Leaderboard** — ranked scan across all assets in a universe

---

## Validated Results (5/5 tests passed)

| Test | Metric | Result |
|------|--------|--------|
| Signal Returns | Avg L/S spread | +3–13% across asset classes |
| Information Coefficient | Avg IC | **0.18** (threshold: 0.04) |
| Phase Transition Accuracy | Hit rate | **66%** |
| v2 vs Naive Baseline | Beat rate | **14/15 assets** |
| Stability Across Regimes | Positive periods | **5/5** regimes |

**Verdict: 🟢 SIGNAL VALIDATED**

---

## Quick Start

```python
import sys
sys.path.insert(0, "specvel")

# ── Equities — no API key needed ──────────────────────────────────────
from adapters.equities import EquitiesAdapter
from leaderboard import run_leaderboard, print_leaderboard

adapter = EquitiesAdapter()
df = run_leaderboard(adapter, start="2023-01-01", end="2026-03-13",
                     asset_class="equities", top_n=15)
print_leaderboard(df)

# ── Commodities — no API key needed ───────────────────────────────────
from adapters.commodities import CommoditiesAdapter
adapter = CommoditiesAdapter()
df = run_leaderboard(adapter, start="2023-01-01", end="2026-03-13",
                     asset_class="commodities", top_n=10)
print_leaderboard(df)

# ── Fixed income + macro — free FRED key needed ───────────────────────
# Get yours free at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_KEY = "your_key_here"

from adapters.fixed_income import FixedIncomeAdapter
adapter = FixedIncomeAdapter(api_key=FRED_KEY)
df = run_leaderboard(adapter, start="2020-01-01", end="2026-03-13",
                     asset_class="fixed_income", top_n=10)
print_leaderboard(df)

from adapters.macro import MacroAdapter
adapter = MacroAdapter(api_key=FRED_KEY)
df = run_leaderboard(adapter, start="2015-01-01", end="2026-03-13",
                     asset_class="macro", top_n=10)
print_leaderboard(df)

# ── FX — USD vs 6 major currencies, FRED key needed ──────────────────
from adapters.fx import FXAdapter
adapter = FXAdapter(api_key=FRED_KEY)
series = adapter.fetch("EURUSD", "2015-01-01", "2026-03-13")

# ── MENA Geopolitical Regime Filter — no API key needed ──────────────
from geopolitical import GeopoliticalRegimeFilter

grf    = GeopoliticalRegimeFilter()
report = grf.run_report(start="2000-01-01", end="2026-03-13")

index_df   = report["index_df"]
multiplier = grf.get_multiplier(report["current_regime"], "ZW=F")
print(f"Wheat multiplier: {multiplier}x  (regime: {report['current_regime']})")
```

---

## Run the Full Backtest

```bash
# Fast mode — equities + commodities, tests 1+2+4 only (~5 min)
python specvel/backtest.py --fast

# Full mode — all 5 tests (~20 min)
python specvel/backtest.py

# With fixed income + macro (FRED key required)
python specvel/backtest.py --fred_key YOUR_KEY --include_macro

# With FX module (FRED key required)
python specvel/backtest.py --fred_key YOUR_KEY --include_fx

# With MENA geopolitical regime filter (no key needed)
python specvel/backtest.py --include_geo

# Full
python specvel/backtest.py --fred_key YOUR_KEY --include_macro --include_fx --include_geo --save results/
```

The backtest returns `T1, T2, T3, T4, T5, T6, T7, T7_index`:

| Return | Contents |
|--------|----------|
| T1 | Signal return L/S spreads per ticker and horizon |
| T2 | Information coefficients (Spearman IC + ICIR) |
| T3 | Phase transition hit rates |
| T4 | v2 vs naive vs v1 comparison |
| T5 | Stability across 5 market regimes |
| T6 | FX dual-direction results (USD + local currency) |
| T7 | Geo regime history (annual) |
| T7_index | Full daily geopolitical stress index DataFrame |

---

## Installation

```bash
pip install -r requirements.txt
```

No accounts needed for equities, commodities, or geopolitical modules.
For fixed income, macro, and FX — get a **free** FRED API key:
→ https://fred.stlouisfed.org/docs/api/api_key.html

---

## Repo Structure

```
specvel/
├── specvel/
│   ├── core.py              ← spectral velocity formula (Savitzky-Golay + gradient)
│   ├── features.py          ← full feature vector per series
│   ├── signals.py           ← LONG/SHORT/NEUTRAL classifier
│   ├── anomaly.py           ← IsolationForest anomaly + changepoint detection
│   ├── leaderboard.py       ← ranked scan across all assets in a universe
│   ├── geopolitical.py      ← MENA geopolitical regime filter (IMF + World Bank)
│   ├── backtest.py          ← 5-test validation suite + FX + geo (T1–T7)
│   ├── cycle.py             ← cycle detection and phase analysis
│   ├── signal_runner.py     ← live signal runner
│   └── adapters/
│       ├── base.py          ← abstract base — implement to add any data source
│       ├── equities.py      ← yfinance stocks/ETFs (free, no key)
│       ├── commodities.py   ← yfinance futures (free, no key)
│       ├── fixed_income.py  ← FRED yields/spreads (free key)
│       ├── macro.py         ← FRED GDP/CPI/PMI (free key)
│       ├── fx.py            ← FRED FX rates — USD vs 6 majors (free key)
│       ├── imf.py           ← IMF PCPS commodity prices (free, no key)
│       └── world_bank.py    ← World Bank WDI MENA indicators (free, no key)
│
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_equities_velocity.ipynb
│   ├── 03_rates_regime.ipynb
│   ├── 04_commodities_signal.ipynb
│   ├── 05_macro_dashboard.ipynb
│   ├── 06_cycle_analysis.ipynb
│   └── 07_mena_geopolitical_filter.ipynb
│
├── configs/
│   ├── equities.yaml
│   ├── fixed_income.yaml
│   ├── commodities.yaml
│   └── macro.yaml
│
├── results/                 ← auto-created, gitignored
├── requirements.txt
└── README.md
```

---

## Data Sources

| Adapter | Source | Key Required | Cost |
|---------|--------|-------------|------|
| `EquitiesAdapter` | Yahoo Finance (yfinance) | None | Free |
| `CommoditiesAdapter` | Yahoo Finance (yfinance) | None | Free |
| `FixedIncomeAdapter` | FRED (St. Louis Fed) | Free registration | Free |
| `MacroAdapter` | FRED (St. Louis Fed) | Free registration | Free |
| `FXAdapter` | FRED (St. Louis Fed) | Free registration | Free |
| `IMFAdapter` | IMF PCPS database | None | Free |
| `WorldBankAdapter` | World Bank WDI | None | Free |

FRED key: https://fred.stlouisfed.org/docs/api/api_key.html

---

## Optimal Holding Periods (from backtest)

| Asset | Optimal Hold | Notes |
|-------|-------------|-------|
| SPY / QQQ / XLK / XLF | 10–20 days | Signal fades quickly |
| IWM | 20–45 days | Small cap slower |
| XLE | 30–45 days | Energy sector |
| WTI Crude | 25–35 days | |
| Silver | 45–90 days | Slow momentum |
| Corn | 20–45 days | |
| Wheat | 5–20 days | Fast decay |
| Gold | 20–45 days | Noisy |
| 2Y Treasury | 60–90 days | Rate cycle |
| 10Y Treasury | 60–90 days | Rate cycle |
| HY Credit | 30–45 days | |
| T10Y2Y Spread | **Regime filter only** | Do not trade directly |

---

## Signal Interpretation

| Conviction | Signal | Action |
|-----------|--------|--------|
| +4 | STRONG LONG 🟢 | Maximum bullish conviction |
| +3 | LONG 🟢 | Strong bullish |
| +2 | LEAN LONG 🟡 | Mild bullish lean |
| +1 | WEAK LONG | Slight positive tilt |
| 0 | NEUTRAL ⚪ | No edge |
| -1 | WEAK SHORT | Slight negative tilt |
| -2 | LEAN SHORT 🟡 | Mild bearish lean |
| -3 | SHORT 🔴 | Strong bearish |
| -4 | STRONG SHORT 🔴 | Maximum bearish conviction |

Signals ≥ +3 or ≤ -3 are high conviction.
Signals of ±1 or ±2 are directional leans — useful for sizing, not standalone signals.

---

## MENA Geopolitical Regime Filter

Adds a **slow fundamental layer** on top of the fast financial velocity signal.
Pulls annual data from IMF PCPS and World Bank WDI, combines them into a
composite stress index, and applies a conviction multiplier to existing signals.

| Regime | Z-score | Wheat | Crude | Gas | Equities |
|--------|---------|-------|-------|-----|----------|
| HIGH_STRESS | > +1.0 | 1.30× | 1.20× | 1.20× | 0.80× |
| ELEVATED | +0.3 → +1.0 | 1.10× | 1.10× | 1.10× | 0.90× |
| NEUTRAL | ±0.3 | 1.00× | 1.00× | 1.00× | 1.00× |
| SUPPRESSED | −0.3 → −1.0 | 0.90× | 0.90× | 0.90× | 1.10× |
| LOW_STRESS | < −1.0 | 0.80× | 0.80× | 0.80× | 1.20× |

```bash
# Standalone regime report + CSV export
python specvel/geopolitical.py --save results/geo_index.csv
```

No API key required — both IMF and World Bank APIs are open.

---

## Two-Layer Signal Framework

```
Layer 1 — FAST  (daily, yfinance / FRED)
    What markets are pricing RIGHT NOW.
    → SpecVel velocity z-score → LONG/SHORT/NEUTRAL

Layer 2 — SLOW  (annual → daily ffill, IMF + World Bank)
    What physical and geopolitical reality IS.
    → Regime classification → multiplier applied to Layer 1

Both align    → amplify conviction
Both diverge  → regime-change warning, reduce conviction
```

---

## FX Module

| Internal ID | FRED Series | Convention |
|-------------|-------------|-----------|
| EURUSD | DEXUSEU | USD per EUR |
| GBPUSD | DEXUSUK | USD per GBP |
| USDJPY | DEXJPUS | inverted → USD per JPY |
| USDCHF | DEXSZUS | inverted → USD per CHF |
| USDCAD | DEXCAUS | inverted → USD per CAD |
| AUDUSD | DEXAUS | USD per AUD |

All series normalised to USD-per-foreign convention. Reports both
USD-strength and local-currency-strength directions in backtest T6 output.

---

## Adding a New Data Source (10 minutes)

```python
# specvel/adapters/my_data.py
from adapters.base import BaseAdapter
import pandas as pd

class MyDataAdapter(BaseAdapter):
    source_name = "my_data"

    def fetch(self, series_id: str, start: str, end: str) -> pd.Series:
        df = pd.read_csv(f"data/{series_id}.csv",
                         index_col=0, parse_dates=True)
        return df["value"][start:end].rename(series_id)

    def list_series(self) -> list:
        return ["series_a", "series_b", "series_c"]

# Immediately usable:
from leaderboard import run_leaderboard, print_leaderboard
adapter = MyDataAdapter()
df = run_leaderboard(adapter, "2020-01-01", "2026-03-13")
print_leaderboard(df)
```

---

## GitHub Actions

Weekly validation every Sunday midnight UTC.

```
.github/workflows/specvel_backtest.yml

Inputs (workflow_dispatch):
  mode:           fast | full
  include_macro:  true | false   (requires FRED_API_KEY secret)
  include_fx:     true | false   (requires FRED_API_KEY secret)
  include_geo:    true | false   (no key required)
```

Add `FRED_API_KEY` as a repository secret to enable FRED modules.
Results are uploaded as workflow artifacts and auto-committed on scheduled runs.

---

## License

Proprietary. See LICENSE.txt.
