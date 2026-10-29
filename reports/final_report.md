# Cloud Workload Forecasting: Final Report

## 1. Executive summary

We forecast the last 2 hours (24 five-minute steps) of two aggregate targets from 10,000 Azure VMs.

- `active_vm_count` is **constant at 10,000** at every timestamp, because the dataset keeps only VMs with complete traces. There is nothing to model; the forecast is 10,000.
- `aggregate_cpu_utilization` is very stable: training mean 9.41%, std 0.19, range 9.11 to 9.91.
- Under rolling-origin validation, a **persistence forecast (last value held flat) was best**: MAE 0.206 (about 2% of the level). Moving averages, seasonal-naive and exponential smoothing were 2% to 29% worse on average.
- Recommendation: plan capacity for **forecast + 0.30 percentage points** (the 90% validation margin); use +0.41 for critical workloads.

## 2. Data understanding and preprocessing

| Check | Result |
|---|---|
| Rows / VMs / readings per VM | 1,340,000 / 10,000 / 134 |
| Timestamps | 134 unique, 0 to 39,900 s, every step exactly 300 s |
| Duplicates, NaNs | none |
| `avg_cpu` range | 0.007 to 99.49 (already in %) |
| VMs per timestamp | 10,000 at every timestamp (std 0) |

**Churn and lifetimes:** none inside the window. The data card states that only VMs with complete traces over the whole window were eligible, so no VM starts or stops. This is a property of the sampling, not of Azure workloads in general.

**Target construction:** the full series was built first, then split. A VM is counted once per timestamp; a reading is valid if `avg_cpu` is in [0, 100]; invalid readings are dropped and never imputed; a timestamp with no valid reading would get count 0 and NaN utilization. The real data needed none of these rules, but the code and tests cover them.

**Metadata** was not used as a feature: its `avg_cpu`, `max_cpu`, `p95_max_cpu` are whole-trace statistics that include the holdout period, and `timestamp_vm_deleted` reveals future churn.

**Disclosure:** the initial data-quality checks and the raw-reading distribution below were computed over the whole raw file, before the split. They were used for understanding only; no modeling decision depends on them. All forecasting analysis (plots, hourly table, validation) used the 110 training rows only.

## 3. Workload analysis

**Per-VM readings.** Mean `avg_cpu` is 9.39% but the median is 5.11% and the 75th percentile 9.62%; the maximum is 99.49%. The distribution is strongly right-skewed: most VMs are lightly used and a few are heavy. The mean of `max_cpu` is 18.3%, so individual VMs spike well above their averages.

**Aggregate series (training window).** Average over 10,000 VMs smooths individual spikes, so the aggregate moves little:

| Hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9* |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean CPU % | 9.41 | 9.41 | 9.37 | 9.36 | **9.62** | **9.71** | 9.37 | 9.30 | 9.24 | 9.14 |

\*hour 9 has only 2 readings.

- Hours 4 and 5 form a temporary plateau about 0.3 points above the rest; hour 4 has the highest variability (std 0.25).
- From hour 6 the level declines steadily.
- The mean absolute step-to-step change is 0.121. Independent noise around a constant with the same std would give about 0.22, so the series is smooth: neighbouring values are strongly related.

**Why it matters operationally:** total load is stable, so aggregate headroom needs are small. Capacity risk sits at individual-VM level, which this aggregate hides.

**Not analysed:** workload differences across segments (6,180 delay-insensitive, 3,091 interactive, 729 unknown VMs by the data card), and daily or weekly patterns (the window is only about 11 hours, so a daily cycle cannot be observed).

## 4. Forecasting approach

| Model | Idea |
|---|---|
| Persistence | all 24 steps equal the last observed value |
| Moving average (6, 12) | mean of the last 6 or 12 values, held flat |
| Seasonal-naive (12) | repeat the last 12 values (1 hour) |
| Simple exponential smoothing | weighted blend of recent and older values; alpha = 1 equals persistence |

No lag-feature or neural models were tried: with 110 points and a smooth series, the risk of overfitting outweighs the gain, and the SES results (below) already show that smoothing does not help. Daily seasonal-naive is impossible (288 steps exceeds the history).

## 5. Validation methodology

Expanding-window **rolling-origin** evaluation on the 110 training rows: fit on the rows before the origin, forecast the next 24 (the same 2-hour horizon as the real task). Origins at rows 50, 62, 74, 86, giving 4 folds. Splits are by row position, never random. The baselines and the candidate model use the identical folds. No holdout target value is read by any model, plot or metric; `predict.py` uses only the holdout timestamps to label rows.

## 6. Results versus baseline

| Model | MAE | RMSE | MAE vs persistence |
|---|---|---|---|
| **persistence** | **0.206** | **0.239** | 1.00 |
| SES alpha 0.9 | 0.209 | 0.241 | 1.02 |
| SES alpha 0.7 | 0.218 | 0.249 | 1.06 |
| SES alpha 0.5 | 0.230 | 0.261 | 1.12 |
| SES alpha 0.3 | 0.245 | 0.276 | 1.19 |
| moving average 12 | 0.254 | 0.282 | 1.24 |
| seasonal-naive 12 | 0.263 | 0.297 | 1.28 |
| moving average 6 | 0.266 | 0.297 | 1.29 |

MAE by fold:

| Fold | persistence | MA 12 | MA 6 | seasonal 12 | SES 0.9 |
|---|---|---|---|---|---|
| 0 | **0.268** | 0.324 | 0.316 | 0.316 | 0.278 |
| 1 | 0.297 | **0.220** | 0.312 | 0.281 | 0.290 |
| 2 | **0.124** | 0.355 | 0.296 | 0.355 | 0.138 |
| 3 | 0.132 | 0.119 | 0.140 | **0.102** | 0.131 |

**Decision rule** (set in advance): a candidate must beat persistence on average *and* in at least 3 of 4 folds. None did. MAE rises as SES alpha moves away from 1, so the latest value is the best summary and smoothing only adds lag.

**Honest reading:** persistence wins 2 of 4 folds, and its average lead is driven largely by fold 2. The evidence is modest, not overwhelming.

**Metrics notes.** `active_vm_count` has zero error for every method, so relative-to-baseline metrics are undefined (baseline MAE is 0). We report MAE and RMSE in CPU percentage points. sMAPE is implemented with an epsilon guard on the denominator but not relied on. No metric was computed on the holdout.

## 7. Error and failure analysis

Pooled over the 4 folds (96 forecast points; folds overlap): MAE 0.206, RMSE 0.254 (pooled RMSE differs from the per-fold average above).

| Group | n | MAE | mean signed error (actual - forecast) |
|---|---|---|---|
| actual above the 75th percentile (9.51) | 33 | 0.240 | +0.15 |
| other | 63 | 0.187 | -0.18 |

- Errors are about 28% larger when load is high.
- Caution: grouping by the *actual* value builds in this bias pattern (any forecaster looks like it under-forecasts at highs). We therefore claim larger peak errors, not a proven peak under-forecasting bias. Grouping by the forecast is not possible with persistence, which yields only 4 distinct forecasts.
- MAE by horizon (steps 1-6, 7-12, 13-18, 19-24): 0.150, 0.174, 0.279, 0.219: roughly worse further out, but 4 overlapping folds are too few to trust the dip.
- Fold 0 is hardest for every model. Fold 2 is bad for every model except persistence and SES 0.9; we did not investigate why.
- **Risk:** the series falls through hours 6-9, so the final training value (9.16) is near the training minimum (9.11). If the level moves back toward the 9.4 average, persistence under-forecasts by roughly 0.25, which is inside the 90% margin below.

## 8. Capacity-planning recommendation

Forecast for the holdout: persistence, 9.16% mean CPU, 10,000 VMs, flat over all 24 steps.

**Policy:** plan for **forecast + margin**, where the margin is a quantile of validation errors (actual - forecast):

| Coverage | Margin (points) | Planning level for a 9.16 forecast |
|---|---|---|
| 80% | +0.134 | 9.30 |
| **90% (recommended)** | **+0.296** | **9.46** |
| 95% (critical workloads) | +0.414 | 9.58 |

**Trade-off:** a smaller margin saves idle capacity but exposes more cases to overload (under-provisioning: slowdowns, outages); a larger margin protects against surges but pays for unused capacity (over-provisioning). In validation the forecast under-shot in 32% of cases, so the margin matters.

For `active_vm_count`, plan for 10,000 VMs; no scaling decision arises in this window. Because the aggregate averages 10,000 VMs, margins are small in absolute terms; real headroom decisions should also consider individual-VM peaks.

## 9. Limitations and future work

- Only about 9 hours of training data: no daily or weekly seasonality can be tested or learned.
- 4 overlapping folds: margins (especially the 95% one) and model rankings are rough. Alpha for SES was chosen on the same folds it is scored on, which is slightly optimistic.
- Persistence cannot anticipate mean reversion or level shifts such as the hour 4-5 plateau.
- A peak-risk trigger could not be validated; segment-level and VM-level analysis was not done.
- Future work: longer traces, segment forecasts (by VM category and core bucket), quantile or conformal prediction intervals, and VM-level peak-risk modelling.