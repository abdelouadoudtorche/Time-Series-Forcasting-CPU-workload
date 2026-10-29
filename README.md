# Cloud Workload Forecasting: Azure VM trace

Forecasts the final 2 hours (24 five-minute steps) of two aggregate targets built from the `data-v1` subset of the Microsoft Azure Public Dataset V2: `active_vm_count` and `aggregate_cpu_utilization`.

**Result:** a persistence (last-value) forecast was the best method under leakage-safe rolling-origin validation. Full write-up: [`reports/final_report.md`](reports/final_report.md).

## Installation

Python 3.11 or newer is required.

Windows PowerShell:

```powershell
py -3.13 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Dataset

Download `data-v1.tar.gz` from the `data-v1` release of <https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1> and extract it into `challenge_data/` (the data is not committed to this repository):

```bash
tar -xzf data-v1.tar.gz -C challenge_data
```

Expected files: `vm_cpu_readings.csv.gz`, `vm_metadata.csv.gz`, `DATA_CARD.md`, `SHA256SUMS`. Verify with `sha256sum -c SHA256SUMS` (Linux/macOS) or `Get-FileHash` (PowerShell).

## Generate the predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This builds the full target series, splits it chronologically (everything before the last 24 timestamps is the training window), forecasts the final 24 timestamps, validates the schema (24 rows, chronological, finite, count >= 0, CPU in [0, 100]) and writes `predictions.csv`. The holdout *timestamps* are used to label rows; the holdout *target values* are never used.

Options: `--holdout-steps N` (default 24). The pipeline derives timestamps from the data and does not hard-code the dataset size.

## Run the tests

```bash
python -m pytest -q
```

## Analysis notebook

`notebooks/01_explore.ipynb` holds the exploration, baseline comparison and error analysis. It needs two extra packages and must be run from inside `notebooks/`:

```bash
python -m pip install matplotlib ipykernel
```

## Layout

| Path | Purpose |
|---|---|
| `src/cwfc/data.py` | load readings, build the two targets, chronological split |
| `src/cwfc/baselines.py` | persistence, moving average, seasonal-naive |
| `src/cwfc/models.py` | simple exponential smoothing (candidate, rejected) |
| `src/cwfc/evaluate.py` | rolling-origin evaluation (by row position) |
| `src/cwfc/metrics.py` | MAE, RMSE, sMAPE, normalized MAE |
| `src/cwfc/predict.py` | command-line entry point that writes `predictions.csv` |
| `tests/` | unit tests, including an end-to-end run on a tiny dataset |
| `reports/final_report.md` | final report |

## Main modeling decisions

- **Targets:** `active_vm_count` = distinct VMs with a valid reading per timestamp; `aggregate_cpu_utilization` = mean `avg_cpu` over valid readings. A reading is valid if `avg_cpu` is within [0, 100]. Invalid readings are dropped, never imputed.
- **Split:** the full series is built first, then the last 24 timestamps are held out.
- **Validation:** expanding-window rolling-origin on the 110 training rows only (horizon 24, step 12, minimum 50 training rows, 4 folds).
- **Models compared:** persistence, moving average (6, 12), seasonal-naive (12), simple exponential smoothing (alpha 0.3 to 0.9).
- **Final forecast:** persistence for both targets (best on validation; for `active_vm_count` the series is constant).
- **Leakage controls:** VM metadata (whole-trace statistics, deletion timestamps) is not used as a feature; no scaling or selection uses holdout values; the holdout is never evaluated.