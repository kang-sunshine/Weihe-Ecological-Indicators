# -*- coding: utf-8 -*-
"""
Step L5.1 - Extract Event-EcoFunction Coupled Responses from L5.0 Masks (Robust V2)

Purpose
-------
Couple L5.0 event masks with L4.3 ecological function state-space variables,
and extract event-aligned ecological responses.

This V2 script reads the validated L5.0 event-mask product directly instead
of attempting to auto-detect event masks in L3.

Inputs
------
1. L5.0 event-mask cube:
   Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc

2. L4.3 EcoFunction state-space cube:
   Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc

Required L5.0 variables
-----------------------
    CDH_event
    HEAT_event
    HR_event
    phase8

Required L4.3 variables
-----------------------
    ZSync
    ZuWUE
    ZNIRv
    A_comp
    EcoState
    phase8

Event-aligned window
--------------------
    tau = -5 ... +7 phases
    relative_days = tau * 8

Event onset definition
----------------------
For each event type and pixel:
    onset[t] = event_mask[t] == 1 and event_mask[t-1] == 0

Outputs
-------
Directory:
    .\Weihe database V3\L5_EventEcoFunction

1. L5_1_Event_Onset_Counts_ByType.csv
2. L5_1_Event_EcoFunction_Ensemble_ByTau.csv
3. L5_1_Event_EcoState_ByTau.csv
4. L5_1_EventEcoFunction_Report.txt

Important notes
---------------
- HEAT_event is Tmax-based high-temperature event, not strict CDNH.
- Means, SDs, minima and maxima are exact over all valid aligned observations.
- Quantiles are approximate sample quantiles for memory safety.
- No input NetCDF is modified.
"""

import os
import gc
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. PATHS
# =============================================================================

L5_EVENT_MASK_NC = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
    r"\Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc"
)

L4_STATE_NC = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
    r"\Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc"
)

OUT_DIR = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_EVENT_COUNTS = os.path.join(
    OUT_DIR,
    "L5_1_Event_Onset_Counts_ByType.csv"
)

OUT_ENSEMBLE = os.path.join(
    OUT_DIR,
    "L5_1_Event_EcoFunction_Ensemble_ByTau.csv"
)

OUT_STATE = os.path.join(
    OUT_DIR,
    "L5_1_Event_EcoState_ByTau.csv"
)

OUT_REPORT = os.path.join(
    OUT_DIR,
    "L5_1_EventEcoFunction_Report.txt"
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

TAU_MIN = -5
TAU_MAX = 7
TAUS = list(range(TAU_MIN, TAU_MAX + 1))
PHASE_DAYS = 8

BLOCK_Y = 24

# Approximate quantile sampling controls.
SAMPLE_LIMIT_PER_GROUP = 200000
SAMPLE_PER_BLOCK_LIMIT = 5000
RANDOM_SEED = 20260907

EVENT_MASK_VARS = {
    "CDH": "CDH_event",
    "HEAT": "HEAT_event",
    "HR": "HR_event",
}

EVENT_LABELS = {
    "CDH": "Compound dry-hot event",
    "HEAT": "Tmax-based high-temperature event",
    "HR": "Heavy rainfall event",
}

L4_FLOAT_VARS = [
    "ZSync",
    "ZuWUE",
    "ZNIRv",
    "A_comp",
]

L4_STATE_VAR = "EcoState"

STATE_LABELS = {
    0: "NoData_or_invalid",
    1: "Stable_functional_state",
    2: "Early_water_carbon_decoupling",
    3: "Water_use_impairment",
    4: "Carbon_photosynthetic_impairment",
    5: "Severe_compound_degradation",
}

# =============================================================================
# 2. HELPERS
# =============================================================================

def sep(title):
    print("\n" + "=" * 112)
    print(title)
    print("=" * 112)


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def masked_to_nan(arr, dtype=np.float32):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    arr = np.asarray(arr, dtype=dtype)

    arr = np.where(
        np.isfinite(arr)
        &
        (arr > -9000.0),
        arr,
        np.nan
    ).astype(dtype, copy=False)

    return arr


def event_mask_from_array(arr):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(0)

    arr = np.asarray(arr)

    if arr.dtype == np.bool_:
        return arr.astype(bool)

    arr = arr.astype(np.float32, copy=False)

    return (
        np.isfinite(arr)
        &
        (arr > 0.5)
    )


def onset_from_mask(mask):
    onset = np.zeros_like(mask, dtype=bool)

    onset[0, :, :] = mask[0, :, :]

    onset[1:, :, :] = (
        mask[1:, :, :]
        &
        ~mask[:-1, :, :]
    )

    return onset


def initialize_aggregators():
    sums = {}
    samples = {}

    for event_type in EVENT_MASK_VARS:
        for tau in TAUS:
            for var in L4_FLOAT_VARS:
                key = (event_type, tau, var)
                sums[key] = {
                    "n": 0,
                    "sum": 0.0,
                    "sumsq": 0.0,
                    "min": np.inf,
                    "max": -np.inf,
                }
                samples[key] = []

    state_counts = {}
    for event_type in EVENT_MASK_VARS:
        for tau in TAUS:
            for state_code in STATE_LABELS:
                state_counts[(event_type, tau, state_code)] = 0

    onset_counts = {event_type: 0 for event_type in EVENT_MASK_VARS}
    event_cell_counts = {event_type: 0 for event_type in EVENT_MASK_VARS}

    valid_response_counts = {
        (event_type, tau): 0
        for event_type in EVENT_MASK_VARS
        for tau in TAUS
    }

    return sums, samples, state_counts, onset_counts, event_cell_counts, valid_response_counts


def update_numeric_agg(agg, values):
    finite = np.isfinite(values)

    if not finite.any():
        return

    vals = values[finite].astype(np.float64, copy=False)

    agg["n"] += int(vals.size)
    agg["sum"] += float(vals.sum())
    agg["sumsq"] += float(np.square(vals).sum())
    agg["min"] = min(agg["min"], float(vals.min()))
    agg["max"] = max(agg["max"], float(vals.max()))


def update_samples(sample_list, values, rng):
    finite = np.isfinite(values)

    if not finite.any():
        return

    vals = values[finite].astype(np.float32, copy=False)

    if vals.size > SAMPLE_PER_BLOCK_LIMIT:
        idx = rng.choice(
            vals.size,
            size=SAMPLE_PER_BLOCK_LIMIT,
            replace=False
        )
        vals = vals[idx]

    sample_list.append(vals.copy())

    total = sum(len(v) for v in sample_list)

    if total > SAMPLE_LIMIT_PER_GROUP * 1.25:
        merged = np.concatenate(sample_list)

        if merged.size > SAMPLE_LIMIT_PER_GROUP:
            idx = rng.choice(
                merged.size,
                size=SAMPLE_LIMIT_PER_GROUP,
                replace=False
            )
            merged = merged[idx]

        sample_list.clear()
        sample_list.append(merged.astype(np.float32, copy=False))


def finalize_samples(sample_list, rng):
    if not sample_list:
        return np.array([], dtype=np.float32)

    merged = np.concatenate(sample_list)

    if merged.size > SAMPLE_LIMIT_PER_GROUP:
        idx = rng.choice(
            merged.size,
            size=SAMPLE_LIMIT_PER_GROUP,
            replace=False
        )
        merged = merged[idx]

    return merged.astype(np.float32, copy=False)


def tau_slices(tau, nt):
    if tau >= 0:
        return (
            slice(0, nt - tau),
            slice(tau, nt)
        )

    return (
        slice(-tau, nt),
        slice(0, nt + tau)
    )


def get_dims_ordered_block(da, y0, y1):
    sub = da.isel(y=slice(y0, y1)).transpose("time", "y", "x")
    return sub.values


# =============================================================================
# 3. OPEN AND VALIDATE INPUTS
# =============================================================================

sep("OPEN INPUT DATASETS")

require_file(L5_EVENT_MASK_NC, "L5.0 event-mask NetCDF")
require_file(L4_STATE_NC, "L4.3 EcoFunction StateSpace NetCDF")

print("L5.0 Event Masks:")
print(L5_EVENT_MASK_NC)

print("\nL4.3:")
print(L4_STATE_NC)

ds_event = xr.open_dataset(
    L5_EVENT_MASK_NC,
    decode_times=True,
    cache=False
)

ds_l4 = xr.open_dataset(
    L4_STATE_NC,
    decode_times=True,
    cache=False
)

print("\nL5.0 dimensions:")
print(dict(ds_event.sizes))

print("\nL4.3 dimensions:")
print(dict(ds_l4.sizes))

for dim in ("time", "y", "x"):
    if dim not in ds_event.dims:
        raise RuntimeError(f"L5.0 missing dimension: {dim}")

    if dim not in ds_l4.dims:
        raise RuntimeError(f"L4.3 missing dimension: {dim}")

for event_type, var in EVENT_MASK_VARS.items():
    if var not in ds_event.data_vars:
        raise KeyError(
            f"Required event mask missing for {event_type}: {var}"
        )

for var in L4_FLOAT_VARS + [L4_STATE_VAR, "phase8"]:
    if var not in ds_l4.variables:
        raise KeyError(
            f"L4.3 required variable missing: {var}"
        )

if "phase8" not in ds_event.variables:
    raise KeyError("L5.0 event-mask file missing phase8.")

time_event = pd.DatetimeIndex(pd.to_datetime(ds_event.time.values))
time_l4 = pd.DatetimeIndex(pd.to_datetime(ds_l4.time.values))

x_event = np.asarray(ds_event.x.values)
x_l4 = np.asarray(ds_l4.x.values)

y_event = np.asarray(ds_event.y.values)
y_l4 = np.asarray(ds_l4.y.values)

align_ok = (
    np.array_equal(time_event.values, time_l4.values)
    and
    np.array_equal(x_event, x_l4)
    and
    np.array_equal(y_event, y_l4)
)

phase_ok = np.array_equal(
    np.asarray(ds_event["phase8"].values, dtype=np.int8),
    np.asarray(ds_l4["phase8"].values, dtype=np.int8)
)

print("\nAlignment PASS:", align_ok)
print("phase8 PASS    :", phase_ok)

if not align_ok:
    raise RuntimeError("L5.0 event masks and L4.3 are not exactly aligned.")

if not phase_ok:
    raise RuntimeError("L5.0 and L4.3 phase8 variables are inconsistent.")

nt = len(time_l4)
ny = len(y_l4)
nx = len(x_l4)

print(
    "\nTime:",
    time_l4[0].date(),
    "->",
    time_l4[-1].date()
)

print(f"Grid: time={nt}, y={ny}, x={nx}")

print(
    f"Relative window: tau={TAU_MIN}..{TAU_MAX} "
    f"({TAU_MIN * PHASE_DAYS}..{TAU_MAX * PHASE_DAYS} days)"
)

print("\nEvent masks:")
for event_type, var in EVENT_MASK_VARS.items():
    print(f"  {event_type:5s}: {var} | {EVENT_LABELS[event_type]}")

# =============================================================================
# 4. INITIALIZE AGGREGATORS
# =============================================================================

sep("INITIALIZE AGGREGATORS")

sums, samples, state_counts, onset_counts, event_cell_counts, valid_response_counts = (
    initialize_aggregators()
)

rng = np.random.default_rng(RANDOM_SEED)

print("Aggregators initialized.")

# =============================================================================
# 5. PROCESS EVENT TYPES AND BLOCKS
# =============================================================================

sep("EXTRACT EVENT-ALIGNED ECOFUNCTION RESPONSES")

n_blocks = int(np.ceil(ny / BLOCK_Y))

for event_type, event_var_name in EVENT_MASK_VARS.items():

    sep(f"PROCESS EVENT TYPE: {event_type} | mask: {event_var_name}")

    event_da = ds_event[event_var_name]

    for ib, y0 in enumerate(range(0, ny, BLOCK_Y), start=1):

        y1 = min(y0 + BLOCK_Y, ny)

        print(f"Block {ib:02d}/{n_blocks:02d} | y={y0}:{y1}")

        event_block_raw = get_dims_ordered_block(event_da, y0, y1)
        event_mask = event_mask_from_array(event_block_raw)
        onset = onset_from_mask(event_mask)

        event_count_block = int(event_mask.sum())
        onset_count_block = int(onset.sum())

        event_cell_counts[event_type] += event_count_block
        onset_counts[event_type] += onset_count_block

        print(
            f"  event cells={event_count_block:,} | "
            f"onsets={onset_count_block:,}"
        )

        if onset_count_block == 0:
            del event_block_raw, event_mask, onset
            gc.collect()
            continue

        l4_blocks = {}

        for var in L4_FLOAT_VARS:
            l4_blocks[var] = masked_to_nan(
                get_dims_ordered_block(ds_l4[var], y0, y1),
                dtype=np.float32
            )

        state_block = np.asarray(
            get_dims_ordered_block(ds_l4[L4_STATE_VAR], y0, y1),
            dtype=np.int8
        )

        for tau in TAUS:

            onset_slice, response_slice = tau_slices(tau, nt)

            onset_tau = onset[onset_slice, :, :]

            if not onset_tau.any():
                continue

            acomp_resp = l4_blocks["A_comp"][response_slice, :, :]

            valid_response_mask = (
                onset_tau
                &
                np.isfinite(acomp_resp)
            )

            n_valid_resp = int(valid_response_mask.sum())

            valid_response_counts[(event_type, tau)] += n_valid_resp

            if n_valid_resp == 0:
                continue

            for var in L4_FLOAT_VARS:
                response_values = l4_blocks[var][response_slice, :, :]
                vals = response_values[valid_response_mask]

                key = (event_type, tau, var)

                update_numeric_agg(sums[key], vals)
                update_samples(samples[key], vals, rng)

            states = state_block[response_slice, :, :][valid_response_mask]

            for state_code in STATE_LABELS:
                state_counts[(event_type, tau, state_code)] += int(
                    (states == state_code).sum()
                )

        del event_block_raw, event_mask, onset, l4_blocks, state_block
        gc.collect()

# =============================================================================
# 6. BUILD OUTPUT TABLES
# =============================================================================

sep("BUILD OUTPUT TABLES")

count_rows = []

for event_type in EVENT_MASK_VARS:
    count_rows.append(
        {
            "event_type": event_type,
            "event_label": EVENT_LABELS[event_type],
            "mask_variable": EVENT_MASK_VARS[event_type],
            "event_cells_total": event_cell_counts[event_type],
            "onset_pixel_events": onset_counts[event_type],
            "event_cells_percent_of_all": (
                100.0 * event_cell_counts[event_type] / (nt * ny * nx)
            ),
            "onset_percent_of_all": (
                100.0 * onset_counts[event_type] / (nt * ny * nx)
            ),
            "tau_min": TAU_MIN,
            "tau_max": TAU_MAX,
            "phase_days": PHASE_DAYS,
        }
    )

counts_df = pd.DataFrame(count_rows)

counts_df.to_csv(
    OUT_EVENT_COUNTS,
    index=False,
    encoding="utf-8-sig"
)

ensemble_rows = []

for event_type in EVENT_MASK_VARS:
    for tau in TAUS:
        row = {
            "event_type": event_type,
            "event_label": EVENT_LABELS[event_type],
            "tau": tau,
            "relative_days": tau * PHASE_DAYS,
            "onset_pixel_events_total": onset_counts[event_type],
            "event_cells_total": event_cell_counts[event_type],
            "valid_response_count": valid_response_counts[(event_type, tau)],
        }

        for var in L4_FLOAT_VARS:
            key = (event_type, tau, var)
            agg = sums[key]
            n = agg["n"]

            row[f"{var}_n"] = n

            if n > 0:
                mean = agg["sum"] / n

                if n > 1:
                    numerator = agg["sumsq"] - n * mean * mean
                    sd = np.sqrt(max(numerator / (n - 1), 0.0))
                else:
                    sd = np.nan

                sample_values = finalize_samples(samples[key], rng)

                if sample_values.size > 0:
                    q05, q25, q50, q75, q95 = np.percentile(
                        sample_values,
                        [5, 25, 50, 75, 95]
                    )
                else:
                    q05 = q25 = q50 = q75 = q95 = np.nan

                row[f"{var}_mean"] = float(mean)
                row[f"{var}_sd"] = float(sd)
                row[f"{var}_min"] = float(agg["min"])
                row[f"{var}_max"] = float(agg["max"])
                row[f"{var}_p05_sample"] = float(q05)
                row[f"{var}_p25_sample"] = float(q25)
                row[f"{var}_median_sample"] = float(q50)
                row[f"{var}_p75_sample"] = float(q75)
                row[f"{var}_p95_sample"] = float(q95)
                row[f"{var}_sample_n"] = int(sample_values.size)

            else:
                for suffix in [
                    "mean",
                    "sd",
                    "min",
                    "max",
                    "p05_sample",
                    "p25_sample",
                    "median_sample",
                    "p75_sample",
                    "p95_sample",
                    "sample_n",
                ]:
                    row[f"{var}_{suffix}"] = np.nan

        ensemble_rows.append(row)

ensemble_df = pd.DataFrame(ensemble_rows)

ensemble_df.to_csv(
    OUT_ENSEMBLE,
    index=False,
    encoding="utf-8-sig"
)

state_rows = []

for event_type in EVENT_MASK_VARS:
    for tau in TAUS:

        total_valid = valid_response_counts[(event_type, tau)]

        for state_code, label in STATE_LABELS.items():

            count = state_counts[(event_type, tau, state_code)]

            state_rows.append(
                {
                    "event_type": event_type,
                    "event_label": EVENT_LABELS[event_type],
                    "tau": tau,
                    "relative_days": tau * PHASE_DAYS,
                    "state_code": state_code,
                    "state_label": label,
                    "count": count,
                    "valid_response_count": total_valid,
                    "percent_of_valid_response": (
                        100.0 * count / total_valid
                        if total_valid > 0
                        else np.nan
                    ),
                }
            )

state_df = pd.DataFrame(state_rows)

state_df.to_csv(
    OUT_STATE,
    index=False,
    encoding="utf-8-sig"
)

print("Saved:")
print(OUT_EVENT_COUNTS)
print(OUT_ENSEMBLE)
print(OUT_STATE)

# =============================================================================
# 7. QC SUMMARY
# =============================================================================

sep("QC SUMMARY")

print("\nEvent onset counts:")
print(counts_df.to_string(index=False))

print("\nEnsemble preview:")
preview_cols = [
    "event_type",
    "tau",
    "relative_days",
    "valid_response_count",
    "ZSync_mean",
    "ZuWUE_mean",
    "ZNIRv_mean",
    "A_comp_mean",
]

print(ensemble_df[preview_cols].head(39).to_string(index=False))

print("\nState preview:")
print(state_df.head(36).to_string(index=False))

event_counts_ok = all(
    counts_df["onset_pixel_events"].fillna(0) > 0
)

valid_response_ok = all(
    ensemble_df.groupby("event_type")["valid_response_count"].max() > 0
)

finite_mean_ok = np.isfinite(
    ensemble_df[
        [
            "ZSync_mean",
            "ZuWUE_mean",
            "ZNIRv_mean",
            "A_comp_mean",
        ]
    ].to_numpy(dtype=float)
).any()

state_sum_ok = True

for (event_type, tau), sub in state_df.groupby(["event_type", "tau"]):
    total_state = int(sub["count"].sum())
    expected = int(sub["valid_response_count"].iloc[0])

    if total_state != expected:
        state_sum_ok = False
        print(
            f"WARNING: state counts do not sum to valid responses "
            f"for {event_type}, tau={tau}: {total_state} vs {expected}"
        )

# =============================================================================
# 8. REPORT
# =============================================================================

sep("WRITE REPORT")

report_lines = []
report_lines.append("Step L5.1 Event-EcoFunction Coupling Report")
report_lines.append("=" * 76)
report_lines.append("")
report_lines.append(
    f"Created UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
)
report_lines.append("")
report_lines.append("Inputs:")
report_lines.append(f"  L5.0 event masks: {L5_EVENT_MASK_NC}")
report_lines.append(f"  L4.3 state space : {L4_STATE_NC}")
report_lines.append("")
report_lines.append("Event masks:")

for event_type, var in EVENT_MASK_VARS.items():
    report_lines.append(
        f"  {event_type}: {var} | {EVENT_LABELS[event_type]}"
    )

report_lines.append("")
report_lines.append(
    "Important methodological note: HEAT is Tmax-based high-temperature event, "
    "not strict CDNH/day-night heatwave because Tmin is unavailable in the "
    "current L3/L5.0 data chain."
)
report_lines.append("")
report_lines.append(
    f"Relative window: tau={TAU_MIN}..{TAU_MAX}, "
    f"approximately {TAU_MIN * PHASE_DAYS}..{TAU_MAX * PHASE_DAYS} days"
)
report_lines.append("")
report_lines.append("Event onset counts:")

for _, row in counts_df.iterrows():
    report_lines.append(
        f"  {row['event_type']}: "
        f"{int(row['onset_pixel_events']):,} onset pixel-events "
        f"from {int(row['event_cells_total']):,} event cells"
    )

report_lines.append("")
report_lines.append("QC:")
report_lines.append(f"  Event counts PASS        : {event_counts_ok}")
report_lines.append(f"  Valid response PASS      : {valid_response_ok}")
report_lines.append(f"  Finite ensemble mean PASS: {finite_mean_ok}")
report_lines.append(f"  State count sum PASS     : {state_sum_ok}")
report_lines.append("")
report_lines.append("Outputs:")
report_lines.append(f"  {OUT_EVENT_COUNTS}")
report_lines.append(f"  {OUT_ENSEMBLE}")
report_lines.append(f"  {OUT_STATE}")
report_lines.append("")
report_lines.append(
    "Interpretation note: output quantiles are approximate sample quantiles. "
    "Means, standard deviations, minima and maxima are exact over all valid "
    "event-aligned observations."
)

report_text = "\n".join(report_lines)

with open(
    OUT_REPORT,
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)

print(report_text)
print("\nSaved report:")
print(OUT_REPORT)

# =============================================================================
# 9. FINAL STATUS
# =============================================================================

sep("FINAL RESULT")

final_ok = all([
    event_counts_ok,
    valid_response_ok,
    finite_mean_ok,
    state_sum_ok,
])

if final_ok:
    print(
        "PASS: Step L5.1 event-ecofunction response extraction completed successfully."
    )
else:
    print(
        "WARNING: Step L5.1 completed, but one or more QC checks failed. "
        "Review event masks and output tables before continuing."
    )

print(f"Event counts PASS        : {event_counts_ok}")
print(f"Valid response PASS      : {valid_response_ok}")
print(f"Finite ensemble mean PASS: {finite_mean_ok}")
print(f"State count sum PASS     : {state_sum_ok}")

print("\nOUTPUT DIRECTORY:")
print(OUT_DIR)

print("\nStep L5.1 Robust V2 completed.")

ds_event.close()
ds_l4.close()
