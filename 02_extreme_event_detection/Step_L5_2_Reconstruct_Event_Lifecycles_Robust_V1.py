# -*- coding: utf-8 -*-
"""
Step L5.2
Independent Event Reconstruction and Basic EcoFunction Lifecycle Metrics
Robust V1

Purpose
-------
1. Read L5.0 event masks:
      CDH_event
      HEAT_event
      HR_event

2. Reconstruct independent pixel-level events.

3. Merge short interruptions:
      default MAX_GAP_PHASES = 1
      i.e. an interruption <= 8 days is merged into one event.

   Example:
      1 1 1 0 1 1
   becomes one event instead of two independent events.

4. Assign globally unique event IDs.

5. Calculate:
      onset
      termination
      duration
      active phases
      filled-gap phases

6. Detect contamination/isolation:
      same-event pre-window contamination
      any-event pre-window contamination
      any-event post-window contamination

7. Extract basic ecosystem-function lifecycle metrics from L4.3:
      ZSync
      ZuWUE
      ZNIRv
      A_comp

8. Estimate:
      event-specific pre-event A_comp baseline
      ecological peak timing
      peak A_comp
      preliminary recovery timing

IMPORTANT
---------
This step DOES NOT yet calculate climate-event severity.

Climate severity and final resistance / recovery / legacy metrics
will be calculated in L5.3 after the event catalogue has passed QC.

The HEAT event remains Tmax-based high-temperature event.
It must NOT be labelled CDNH / day-night heatwave unless Tmin
is added to the upstream data chain.

Author workflow:
Weihe Basin ecological resilience project
"""

import os
import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")


# =============================================================================
# 0. USER SETTINGS
# =============================================================================

BASE_DIR = r".\Weihe database V3"

L50_FILE = os.path.join(
    BASE_DIR,
    "L5_EventEcoFunction",
    "Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc"
)

L43_FILE = os.path.join(
    BASE_DIR,
    "L4_EcoFunction",
    "Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "L5_EventEcoFunction",
    "L5_2_EventLifecycle"
)

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Event definition
# -----------------------------------------------------------------------------

EVENT_CONFIG = {
    "CDH": {
        "mask": "CDH_event",
        "label": "Compound dry-hot event"
    },
    "HEAT": {
        "mask": "HEAT_event",
        "label": "Tmax-based high-temperature event"
    },
    "HR": {
        "mask": "HR_event",
        "label": "Heavy rainfall event"
    }
}


# -----------------------------------------------------------------------------
# Time / lifecycle configuration
# -----------------------------------------------------------------------------

PHASE_DAYS = 8

# Merge event fragments separated by <= 1 phase (= 8 days)
MAX_GAP_PHASES = 1

# Do NOT impose minimum event duration yet.
# This preserves the upstream event definition.
MIN_ACTIVE_PHASES = 1

# Same relative-time system as L5.1
TAU_MIN = -5
TAU_MAX = 7

# y block size.
# 24 worked well in L5.1.
Y_BLOCK_SIZE = 24

# Minimum number of finite pre-event A_comp observations
# required to define pre-event baseline.
MIN_PRE_VALID = 3

# Flush records periodically so millions of events do not stay in RAM.
FLUSH_RECORDS = 100_000


# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def print_header(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def detect_variable(ds, candidates, keywords=None):
    """
    Detect variable using:
      1. exact candidate name
      2. case-insensitive candidate name
      3. keyword search
    """

    names = list(ds.data_vars)

    # exact
    for name in candidates:
        if name in names:
            return name

    # case insensitive
    lower_map = {n.lower(): n for n in names}

    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # keyword search
    if keywords:
        for n in names:
            ln = n.lower()
            if all(k.lower() in ln for k in keywords):
                return n

    return None


def safe_float(v):
    """
    Return finite float or NaN.
    """

    try:
        v = float(v)
        if np.isfinite(v):
            return v
    except Exception:
        pass

    return np.nan


def reconstruct_segments(mask_1d, max_gap=1):
    """
    Convert one pixel's binary time series into independent event segments.

    Parameters
    ----------
    mask_1d : bool array, shape(time,)
    max_gap : maximum number of FALSE phases allowed inside one event.

    Example
    -------
    max_gap = 1:

    111011 -> one event

    1110011 -> two events

    Returns
    -------
    list of tuples:
        (start_idx, end_idx, active_phase_count, gap_filled_count)
    """

    true_idx = np.flatnonzero(mask_1d)

    if true_idx.size == 0:
        return []

    # If difference between two TRUE values exceeds max_gap + 1,
    # they belong to different reconstructed events.
    d = np.diff(true_idx)

    split_positions = np.where(d > (max_gap + 1))[0]

    group_starts = np.r_[0, split_positions + 1]
    group_ends = np.r_[split_positions, len(true_idx) - 1]

    segments = []

    for gs, ge in zip(group_starts, group_ends):

        start = int(true_idx[gs])
        end = int(true_idx[ge])

        active_count = int(ge - gs + 1)

        span_count = end - start + 1

        gap_count = span_count - active_count

        segments.append(
            (
                start,
                end,
                active_count,
                gap_count
            )
        )

    return segments


def finite_median(arr, min_valid=1):
    arr = np.asarray(arr, dtype=np.float64)

    good = np.isfinite(arr)

    if good.sum() < min_valid:
        return np.nan

    return float(np.nanmedian(arr))


def finite_max_with_index(arr):
    """
    Return:
        max_value,
        local_index

    local_index is relative to the provided array.
    """

    arr = np.asarray(arr, dtype=np.float64)

    if not np.isfinite(arr).any():
        return np.nan, None

    i = int(np.nanargmax(arr))

    return float(arr[i]), i


def get_date_string(time_value):
    try:
        return pd.Timestamp(time_value).strftime("%Y-%m-%d")
    except Exception:
        return str(time_value)


def count_raw_onsets(mask_block):
    """
    Count ordinary 0 -> 1 event onsets before gap merging.

    mask_block shape:
        time, y, x
    """

    first = mask_block[0].sum()

    transitions = (
        mask_block[1:]
        &
        (~mask_block[:-1])
    ).sum()

    return int(first + transitions)


# =============================================================================
# 2. OPEN INPUT DATASETS
# =============================================================================

print_header("OPEN INPUT DATASETS")

print("L5.0 Event Masks:")
print(L50_FILE)

print("\nL4.3 EcoFunction:")
print(L43_FILE)


ds_event = xr.open_dataset(
    L50_FILE,
    mask_and_scale=True
)

ds_eco = xr.open_dataset(
    L43_FILE,
    mask_and_scale=True
)


print("\nL5.0 dimensions:")
print(dict(ds_event.sizes))

print("\nL4.3 dimensions:")
print(dict(ds_eco.sizes))


# =============================================================================
# 3. DATA ALIGNMENT QC
# =============================================================================

print_header("ALIGNMENT QC")


required_dims = ["time", "y", "x"]

for dim in required_dims:

    if dim not in ds_event.dims:
        raise RuntimeError(
            f"L5.0 missing required dimension: {dim}"
        )

    if dim not in ds_eco.dims:
        raise RuntimeError(
            f"L4.3 missing required dimension: {dim}"
        )


same_shape = all(
    ds_event.sizes[d] == ds_eco.sizes[d]
    for d in required_dims
)

print("Dimension-size alignment:", same_shape)

if not same_shape:
    raise RuntimeError(
        "L5.0 and L4.3 dimension sizes differ."
    )


time_event = pd.DatetimeIndex(
    pd.to_datetime(ds_event["time"].values)
)

time_eco = pd.DatetimeIndex(
    pd.to_datetime(ds_eco["time"].values)
)

time_equal = np.array_equal(
    time_event.values,
    time_eco.values
)

print("Time alignment:", time_equal)

if not time_equal:
    raise RuntimeError(
        "L5.0 and L4.3 time coordinates differ."
    )


x_equal = np.allclose(
    ds_event["x"].values,
    ds_eco["x"].values,
    equal_nan=True
)

y_equal = np.allclose(
    ds_event["y"].values,
    ds_eco["y"].values,
    equal_nan=True
)

print("X alignment:", x_equal)
print("Y alignment:", y_equal)

if not (x_equal and y_equal):
    raise RuntimeError(
        "Spatial coordinates differ between L5.0 and L4.3."
    )


n_time = ds_event.sizes["time"]
n_y = ds_event.sizes["y"]
n_x = ds_event.sizes["x"]

x_coord = np.asarray(ds_event["x"].values)
y_coord = np.asarray(ds_event["y"].values)


# -----------------------------------------------------------------------------
# Check 8-day phase structure
# -----------------------------------------------------------------------------

time_diff_days = (
    np.diff(time_event.values)
    / np.timedelta64(1, "D")
)

phase8_pass = np.all(
    np.isin(
        time_diff_days,
        [6, 7, 8, 9]
    )
)

# Some year-boundary / MODIS-style calendar intervals can differ slightly,
# therefore this is diagnostic rather than a fatal condition.

print("Approximate 8-day phase PASS:", phase8_pass)

print(
    f"\nTime: "
    f"{time_event[0].date()} -> "
    f"{time_event[-1].date()}"
)

print(
    f"Grid: time={n_time}, "
    f"y={n_y}, "
    f"x={n_x}"
)

print(
    f"Lifecycle extraction window: "
    f"tau={TAU_MIN}..{TAU_MAX} "
    f"({TAU_MIN * PHASE_DAYS}.."
    f"{TAU_MAX * PHASE_DAYS} days)"
)


# =============================================================================
# 4. DETECT ECOFUNCTION VARIABLES
# =============================================================================

print_header("DETECT ECOFUNCTION VARIABLES")


VAR_ZSYNC = detect_variable(
    ds_eco,
    [
        "ZSync",
        "Z_Sync",
        "ZSynchrony",
        "Synchrony_Z",
        "Synchrony_r_Z",
        "Z_Synchrony"
    ]
)

VAR_ZUWUE = detect_variable(
    ds_eco,
    [
        "ZuWUE",
        "Z_uWUE",
        "uWUE_Z",
        "Zuwue"
    ]
)

VAR_ZNIRV = detect_variable(
    ds_eco,
    [
        "ZNIRv",
        "Z_NIRv",
        "NIRv_Z",
        "Znirv"
    ]
)

VAR_ACOMP = detect_variable(
    ds_eco,
    [
        "A_comp",
        "Acomp",
        "A_compound",
        "Composite_Anomaly"
    ]
)


detected = {
    "ZSync": VAR_ZSYNC,
    "ZuWUE": VAR_ZUWUE,
    "ZNIRv": VAR_ZNIRV,
    "A_comp": VAR_ACOMP
}


for k, v in detected.items():
    print(f"{k:<10}: {v}")


missing = [
    k
    for k, v in detected.items()
    if v is None
]

if missing:

    print("\nAvailable L4.3 variables:")

    for v in ds_eco.data_vars:
        print("  ", v)

    raise RuntimeError(
        "Unable to detect required L4.3 variables: "
        + ", ".join(missing)
    )


# =============================================================================
# 5. CHECK EVENT MASK VARIABLES
# =============================================================================

print_header("CHECK EVENT MASKS")


for event_type, cfg in EVENT_CONFIG.items():

    mask_name = cfg["mask"]

    if mask_name not in ds_event.data_vars:

        raise RuntimeError(
            f"Missing event mask variable: {mask_name}"
        )

    print(
        f"{event_type:<6}: "
        f"{mask_name:<15} | "
        f"{cfg['label']}"
    )


# =============================================================================
# 6. PREPARE OUTPUT FILES
# =============================================================================

print_header("PREPARE OUTPUT FILES")


catalog_files = {}

for event_type in EVENT_CONFIG:

    fp = os.path.join(
        OUT_DIR,
        f"L5_2_{event_type}_Independent_Event_Catalog.csv.gz"
    )

    catalog_files[event_type] = fp

    # remove old output before rerun
    if os.path.exists(fp):
        os.remove(fp)

    print(
        f"{event_type}: {fp}"
    )


summary_file = os.path.join(
    OUT_DIR,
    "L5_2_Event_Reconstruction_Summary.csv"
)

report_file = os.path.join(
    OUT_DIR,
    "L5_2_EventLifecycle_Report.txt"
)


# =============================================================================
# 7. GLOBAL QC COUNTERS
# =============================================================================

qc = {}

for event_type in EVENT_CONFIG:

    qc[event_type] = {
        "event_cells": 0,
        "raw_onsets": 0,
        "reconstructed_events": 0,
        "events_with_gap_fill": 0,
        "gap_phases_filled": 0,
        "clean_pre_same": 0,
        "clean_pre_any": 0,
        "clean_post_any": 0,
        "valid_pre_baseline": 0,
        "valid_eco_peak": 0,
        "recovery_found": 0,
        "duration_phase_sum": 0,
        "active_phase_sum": 0
    }


event_serial = {
    k: 0
    for k in EVENT_CONFIG
}


header_written = {
    k: False
    for k in EVENT_CONFIG
}


# =============================================================================
# 8. OUTPUT WRITER
# =============================================================================

def flush_records(event_type, records):

    if not records:
        return

    df = pd.DataFrame.from_records(records)

    mode = "at" if header_written[event_type] else "wt"

    with gzip.open(
        catalog_files[event_type],
        mode,
        encoding="utf-8",
        newline=""
    ) as gz:

        df.to_csv(
            gz,
            index=False,
            header=not header_written[event_type]
        )

    header_written[event_type] = True


# =============================================================================
# 9. EVENT LIFECYCLE RECONSTRUCTION
# =============================================================================

print_header("RECONSTRUCT INDEPENDENT EVENT LIFECYCLES")


n_blocks = int(
    np.ceil(
        n_y / Y_BLOCK_SIZE
    )
)


for block_no, y0 in enumerate(
    range(0, n_y, Y_BLOCK_SIZE),
    start=1
):

    y1 = min(
        y0 + Y_BLOCK_SIZE,
        n_y
    )

    print(
        f"\nBlock {block_no:02d}/{n_blocks:02d} "
        f"| y={y0}:{y1}"
    )

    # -------------------------------------------------------------------------
    # Load all 3 event masks for this spatial block
    # -------------------------------------------------------------------------

    block_masks = {}

    for event_type, cfg in EVENT_CONFIG.items():

        arr = (
            ds_event[cfg["mask"]]
            .transpose(
                "time",
                "y",
                "x"
            )
            .isel(
                y=slice(y0, y1)
            )
            .values
        )

        arr = np.asarray(arr)

        # tolerate 0/1, NaN etc.
        arr = np.isfinite(arr) & (arr > 0)

        block_masks[event_type] = arr


    # Any climate event type
    any_event_mask = np.zeros_like(
        next(iter(block_masks.values())),
        dtype=bool
    )

    for arr in block_masks.values():
        any_event_mask |= arr


    # -------------------------------------------------------------------------
    # Load EcoFunction variables once for this block
    # -------------------------------------------------------------------------

    eco_zsync = np.asarray(
        ds_eco[VAR_ZSYNC]
        .transpose(
            "time",
            "y",
            "x"
        )
        .isel(
            y=slice(y0, y1)
        )
        .values,
        dtype=np.float32
    )

    eco_zuwue = np.asarray(
        ds_eco[VAR_ZUWUE]
        .transpose(
            "time",
            "y",
            "x"
        )
        .isel(
            y=slice(y0, y1)
        )
        .values,
        dtype=np.float32
    )

    eco_znirv = np.asarray(
        ds_eco[VAR_ZNIRV]
        .transpose(
            "time",
            "y",
            "x"
        )
        .isel(
            y=slice(y0, y1)
        )
        .values,
        dtype=np.float32
    )

    eco_acomp = np.asarray(
        ds_eco[VAR_ACOMP]
        .transpose(
            "time",
            "y",
            "x"
        )
        .isel(
            y=slice(y0, y1)
        )
        .values,
        dtype=np.float32
    )


    # =========================================================================
    # Each event type
    # =========================================================================

    for event_type, cfg in EVENT_CONFIG.items():

        mask_block = block_masks[event_type]

        event_cells = int(
            mask_block.sum()
        )

        raw_onsets = count_raw_onsets(
            mask_block
        )

        qc[event_type]["event_cells"] += event_cells
        qc[event_type]["raw_onsets"] += raw_onsets

        print(
            f"  {event_type:<6} "
            f"| cells={event_cells:,} "
            f"| raw_onsets={raw_onsets:,}"
        )


        records = []


        # =====================================================================
        # Spatial pixel loop
        # =====================================================================

        for by in range(y1 - y0):

            global_y = y0 + by

            for ix in range(n_x):

                mask_1d = mask_block[
                    :,
                    by,
                    ix
                ]

                if not mask_1d.any():
                    continue


                segments = reconstruct_segments(
                    mask_1d,
                    max_gap=MAX_GAP_PHASES
                )


                for (
                    start,
                    end,
                    active_count,
                    gap_count
                ) in segments:


                    # ---------------------------------------------------------
                    # Preserve upstream event definition.
                    # ---------------------------------------------------------

                    if active_count < MIN_ACTIVE_PHASES:
                        continue


                    event_serial[event_type] += 1

                    eid = (
                        f"{event_type}_"
                        f"{event_serial[event_type]:010d}"
                    )


                    qc[event_type]["reconstructed_events"] += 1

                    span_phases = (
                        end
                        - start
                        + 1
                    )

                    qc[event_type]["duration_phase_sum"] += span_phases
                    qc[event_type]["active_phase_sum"] += active_count


                    if gap_count > 0:

                        qc[event_type][
                            "events_with_gap_fill"
                        ] += 1

                        qc[event_type][
                            "gap_phases_filled"
                        ] += gap_count


                    # =========================================================
                    # Event-isolation QC
                    # =========================================================

                    pre0 = max(
                        0,
                        start + TAU_MIN
                    )

                    pre1 = start


                    # No same event type during tau=-5..-1
                    clean_pre_same = not bool(
                        mask_block[
                            pre0:pre1,
                            by,
                            ix
                        ].any()
                    )


                    # No CDH / HEAT / HR during tau=-5..-1
                    clean_pre_any = not bool(
                        any_event_mask[
                            pre0:pre1,
                            by,
                            ix
                        ].any()
                    )


                    post0 = end + 1

                    post1 = min(
                        n_time,
                        end + TAU_MAX + 1
                    )


                    # no climate event of any type after termination
                    clean_post_any = not bool(
                        any_event_mask[
                            post0:post1,
                            by,
                            ix
                        ].any()
                    )


                    if clean_pre_same:
                        qc[event_type]["clean_pre_same"] += 1

                    if clean_pre_any:
                        qc[event_type]["clean_pre_any"] += 1

                    if clean_post_any:
                        qc[event_type]["clean_post_any"] += 1


                    # =========================================================
                    # Pre-event EcoFunction baseline
                    # =========================================================

                    acomp_pre = eco_acomp[
                        pre0:pre1,
                        by,
                        ix
                    ]


                    pre_acomp_median = finite_median(
                        acomp_pre,
                        min_valid=MIN_PRE_VALID
                    )


                    pre_valid_n = int(
                        np.isfinite(
                            acomp_pre
                        ).sum()
                    )


                    if np.isfinite(pre_acomp_median):
                        qc[event_type][
                            "valid_pre_baseline"
                        ] += 1


                    # =========================================================
                    # EcoFunction at event onset
                    # =========================================================

                    zsync_onset = safe_float(
                        eco_zsync[
                            start,
                            by,
                            ix
                        ]
                    )

                    zuwue_onset = safe_float(
                        eco_zuwue[
                            start,
                            by,
                            ix
                        ]
                    )

                    znirv_onset = safe_float(
                        eco_znirv[
                            start,
                            by,
                            ix
                        ]
                    )

                    acomp_onset = safe_float(
                        eco_acomp[
                            start,
                            by,
                            ix
                        ]
                    )


                    # =========================================================
                    # Ecological peak
                    #
                    # Search:
                    # tau = 0..+7 relative to climate onset
                    # =========================================================

                    eco_search_end = min(
                        n_time,
                        start + TAU_MAX + 1
                    )


                    acomp_post = eco_acomp[
                        start:eco_search_end,
                        by,
                        ix
                    ]


                    (
                        peak_acomp,
                        peak_local_idx
                    ) = finite_max_with_index(
                        acomp_post
                    )


                    if peak_local_idx is not None:

                        peak_idx = (
                            start
                            + peak_local_idx
                        )

                        peak_tau = (
                            peak_idx
                            - start
                        )

                        peak_days = (
                            peak_tau
                            * PHASE_DAYS
                        )

                        peak_time = get_date_string(
                            time_event[
                                peak_idx
                            ]
                        )

                        qc[event_type][
                            "valid_eco_peak"
                        ] += 1

                    else:

                        peak_idx = None
                        peak_tau = np.nan
                        peak_days = np.nan
                        peak_time = ""


                    # =========================================================
                    # Preliminary ecological recovery
                    #
                    # Definition:
                    # First post-peak phase where
                    # A_comp <= event-specific pre-event median A_comp.
                    #
                    # Search only inside current L5.1-compatible +56 d window.
                    #
                    # Final recovery metric will be revisited in L5.3.
                    # =========================================================

                    recovery_idx = None
                    recovery_tau = np.nan
                    recovery_days = np.nan
                    recovery_time = ""
                    recovery_found = False


                    if (
                        peak_idx is not None
                        and
                        np.isfinite(
                            pre_acomp_median
                        )
                    ):

                        rec_end = min(
                            n_time,
                            start + TAU_MAX + 1
                        )


                        for ridx in range(
                            peak_idx + 1,
                            rec_end
                        ):

                            av = safe_float(
                                eco_acomp[
                                    ridx,
                                    by,
                                    ix
                                ]
                            )

                            if not np.isfinite(av):
                                continue

                            if av <= pre_acomp_median:

                                recovery_idx = ridx

                                recovery_tau = (
                                    ridx
                                    - start
                                )

                                recovery_days = (
                                    recovery_tau
                                    * PHASE_DAYS
                                )

                                recovery_time = (
                                    get_date_string(
                                        time_event[
                                            ridx
                                        ]
                                    )
                                )

                                recovery_found = True

                                break


                    if recovery_found:
                        qc[event_type][
                            "recovery_found"
                        ] += 1


                    # =========================================================
                    # Change from pre-event condition
                    # =========================================================

                    if (
                        np.isfinite(
                            peak_acomp
                        )
                        and
                        np.isfinite(
                            pre_acomp_median
                        )
                    ):

                        delta_peak_acomp = (
                            peak_acomp
                            - pre_acomp_median
                        )

                    else:

                        delta_peak_acomp = np.nan


                    # =========================================================
                    # Window truncation flags
                    # =========================================================

                    truncated_pre = (
                        start + TAU_MIN < 0
                    )

                    truncated_post = (
                        start + TAU_MAX
                        >= n_time
                    )


                    # =========================================================
                    # Event record
                    # =========================================================

                    record = {

                        # -----------------------------------------------------
                        # Identity
                        # -----------------------------------------------------
                        "event_id": eid,
                        "event_type": event_type,
                        "event_label": cfg["label"],

                        # -----------------------------------------------------
                        # Pixel
                        # -----------------------------------------------------
                        "row": int(global_y),
                        "col": int(ix),
                        "y": float(
                            y_coord[
                                global_y
                            ]
                        ),
                        "x": float(
                            x_coord[
                                ix
                            ]
                        ),

                        # -----------------------------------------------------
                        # Event timing
                        # -----------------------------------------------------
                        "onset_index": int(start),
                        "onset_time": get_date_string(
                            time_event[
                                start
                            ]
                        ),

                        "termination_index": int(end),
                        "termination_time": get_date_string(
                            time_event[
                                end
                            ]
                        ),

                        # -----------------------------------------------------
                        # Event structure
                        # -----------------------------------------------------
                        "span_phases": int(
                            span_phases
                        ),
                        "duration_days": int(
                            span_phases
                            * PHASE_DAYS
                        ),

                        "active_phases_original": int(
                            active_count
                        ),

                        "gap_filled_phases": int(
                            gap_count
                        ),

                        "gap_filled_days": int(
                            gap_count
                            * PHASE_DAYS
                        ),

                        # -----------------------------------------------------
                        # Isolation
                        # -----------------------------------------------------
                        "clean_prewindow_same_event": int(
                            clean_pre_same
                        ),

                        "clean_prewindow_any_event": int(
                            clean_pre_any
                        ),

                        "clean_postwindow_any_event": int(
                            clean_post_any
                        ),

                        "window_truncated_pre": int(
                            truncated_pre
                        ),

                        "window_truncated_post": int(
                            truncated_post
                        ),

                        # -----------------------------------------------------
                        # Pre-event baseline
                        # -----------------------------------------------------
                        "pre_valid_Acomp_n": int(
                            pre_valid_n
                        ),

                        "pre_Acomp_median": (
                            pre_acomp_median
                        ),

                        # -----------------------------------------------------
                        # Onset ecofunction
                        # -----------------------------------------------------
                        "ZSync_onset": zsync_onset,
                        "ZuWUE_onset": zuwue_onset,
                        "ZNIRv_onset": znirv_onset,
                        "Acomp_onset": acomp_onset,

                        # -----------------------------------------------------
                        # Eco peak
                        # -----------------------------------------------------
                        "eco_peak_index": (
                            peak_idx
                            if peak_idx is not None
                            else np.nan
                        ),

                        "eco_peak_time": peak_time,

                        "eco_peak_tau": peak_tau,

                        "eco_peak_days_after_onset": (
                            peak_days
                        ),

                        "Acomp_peak": peak_acomp,

                        "Acomp_peak_minus_pre": (
                            delta_peak_acomp
                        ),

                        # -----------------------------------------------------
                        # Preliminary recovery
                        # -----------------------------------------------------
                        "recovery_found": int(
                            recovery_found
                        ),

                        "recovery_index": (
                            recovery_idx
                            if recovery_idx is not None
                            else np.nan
                        ),

                        "recovery_time": recovery_time,

                        "recovery_tau": recovery_tau,

                        "recovery_days_after_onset": (
                            recovery_days
                        )
                    }


                    records.append(
                        record
                    )


                    # ---------------------------------------------------------
                    # Flush event catalogue periodically
                    # ---------------------------------------------------------

                    if len(records) >= FLUSH_RECORDS:

                        flush_records(
                            event_type,
                            records
                        )

                        records = []


        # final flush for this event type / block
        flush_records(
            event_type,
            records
        )


# =============================================================================
# 10. BUILD QC SUMMARY
# =============================================================================

print_header("BUILD EVENT RECONSTRUCTION SUMMARY")


summary_rows = []


for event_type, cfg in EVENT_CONFIG.items():

    q = qc[event_type]

    n = q["reconstructed_events"]


    def pct(v):
        if n == 0:
            return np.nan

        return (
            100.0
            * v
            / n
        )


    if n > 0:

        mean_span = (
            q["duration_phase_sum"]
            / n
        )

        mean_active = (
            q["active_phase_sum"]
            / n
        )

    else:

        mean_span = np.nan
        mean_active = np.nan


    if q["raw_onsets"] > 0:

        reduction_pct = (
            100.0
            * (
                q["raw_onsets"]
                -
                q["reconstructed_events"]
            )
            /
            q["raw_onsets"]
        )

    else:

        reduction_pct = np.nan


    row = {

        "event_type": event_type,
        "event_label": cfg["label"],

        "event_cells": (
            q["event_cells"]
        ),

        "raw_onsets_before_merge": (
            q["raw_onsets"]
        ),

        "reconstructed_events": (
            q["reconstructed_events"]
        ),

        "event_count_reduction_percent": (
            reduction_pct
        ),

        "events_with_gap_fill": (
            q["events_with_gap_fill"]
        ),

        "events_with_gap_fill_percent": (
            pct(
                q["events_with_gap_fill"]
            )
        ),

        "total_gap_phases_filled": (
            q["gap_phases_filled"]
        ),

        "clean_pre_same_event_n": (
            q["clean_pre_same"]
        ),

        "clean_pre_same_event_percent": (
            pct(
                q["clean_pre_same"]
            )
        ),

        "clean_pre_any_event_n": (
            q["clean_pre_any"]
        ),

        "clean_pre_any_event_percent": (
            pct(
                q["clean_pre_any"]
            )
        ),

        "clean_post_any_event_n": (
            q["clean_post_any"]
        ),

        "clean_post_any_event_percent": (
            pct(
                q["clean_post_any"]
            )
        ),

        "valid_pre_baseline_n": (
            q["valid_pre_baseline"]
        ),

        "valid_pre_baseline_percent": (
            pct(
                q["valid_pre_baseline"]
            )
        ),

        "valid_eco_peak_n": (
            q["valid_eco_peak"]
        ),

        "valid_eco_peak_percent": (
            pct(
                q["valid_eco_peak"]
            )
        ),

        "preliminary_recovery_found_n": (
            q["recovery_found"]
        ),

        "preliminary_recovery_found_percent": (
            pct(
                q["recovery_found"]
            )
        ),

        "mean_event_span_phases": (
            mean_span
        ),

        "mean_event_span_days": (
            mean_span
            * PHASE_DAYS
        ),

        "mean_original_active_phases": (
            mean_active
        ),

        "max_gap_phases": (
            MAX_GAP_PHASES
        ),

        "phase_days": (
            PHASE_DAYS
        )
    }

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig"
)


print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 11. QC PASS / FAIL
# =============================================================================

print_header("QC SUMMARY")


qc_event_counts_pass = all(

    qc[e][
        "reconstructed_events"
    ] > 0

    for e in EVENT_CONFIG
)


qc_merge_logic_pass = all(

    qc[e][
        "reconstructed_events"
    ]
    <=
    qc[e][
        "raw_onsets"
    ]

    for e in EVENT_CONFIG
)


qc_peak_pass = all(

    (
        qc[e]["valid_eco_peak"]
        /
        max(
            qc[e]["reconstructed_events"],
            1
        )
    )
    > 0.50

    for e in EVENT_CONFIG
)


qc_baseline_pass = all(

    (
        qc[e]["valid_pre_baseline"]
        /
        max(
            qc[e]["reconstructed_events"],
            1
        )
    )
    > 0.30

    for e in EVENT_CONFIG
)


print(
    "Event reconstruction count PASS :",
    qc_event_counts_pass
)

print(
    "Gap-merging logic PASS           :",
    qc_merge_logic_pass
)

print(
    "Eco peak availability PASS       :",
    qc_peak_pass
)

print(
    "Pre-event baseline PASS           :",
    qc_baseline_pass
)


overall_pass = all(
    [
        same_shape,
        time_equal,
        x_equal,
        y_equal,
        qc_event_counts_pass,
        qc_merge_logic_pass,
        qc_peak_pass,
        qc_baseline_pass
    ]
)


# =============================================================================
# 12. WRITE REPORT
# =============================================================================

print_header("WRITE REPORT")


report_lines = []

report_lines.append(
    "Step L5.2 Independent Event Lifecycle Reconstruction Report"
)

report_lines.append(
    "=" * 80
)

report_lines.append("")

report_lines.append(
    f"L5.0 event masks: {L50_FILE}"
)

report_lines.append(
    f"L4.3 state space : {L43_FILE}"
)

report_lines.append("")

report_lines.append(
    f"Time: "
    f"{time_event[0].date()} -> "
    f"{time_event[-1].date()}"
)

report_lines.append(
    f"Grid: "
    f"time={n_time}, "
    f"y={n_y}, "
    f"x={n_x}"
)

report_lines.append("")

report_lines.append(
    f"Gap merging rule: "
    f"<= {MAX_GAP_PHASES} phase "
    f"(<= {MAX_GAP_PHASES * PHASE_DAYS} days)"
)

report_lines.append(
    f"Minimum active phases: "
    f"{MIN_ACTIVE_PHASES}"
)

report_lines.append(
    f"Eco lifecycle window: "
    f"tau={TAU_MIN}..{TAU_MAX} "
    f"({TAU_MIN * PHASE_DAYS}.."
    f"{TAU_MAX * PHASE_DAYS} days)"
)

report_lines.append("")

report_lines.append(
    "Event reconstruction:"
)


for event_type in EVENT_CONFIG:

    q = qc[event_type]

    report_lines.append(
        f"  {event_type}: "
        f"{q['raw_onsets']:,} raw onsets -> "
        f"{q['reconstructed_events']:,} "
        f"independent reconstructed pixel-events"
    )


report_lines.append("")

report_lines.append(
    "Important interpretation:"
)

report_lines.append(
    "  Event IDs remain pixel-level events. "
    "They are no longer fragmented onset observations, "
    "but spatial pixels from the same regional climate event "
    "are still not statistically independent."
)

report_lines.append(
    "  Therefore future inference must use "
    "spatial/event clustering, block bootstrap, "
    "or regional-event aggregation rather than treating "
    "all pixel-events as independent replicates."
)

report_lines.append("")

report_lines.append(
    "HEAT remains Tmax-based high-temperature event, "
    "not strict CDNH/day-night heatwave."
)

report_lines.append("")

report_lines.append(
    "Preliminary recovery definition:"
)

report_lines.append(
    "  first post-peak phase within +56 days where "
    "A_comp returns to or below the event-specific "
    "pre-event median A_comp."
)

report_lines.append(
    "  This is a provisional diagnostic metric. "
    "Final resistance/recovery/legacy definitions "
    "will be implemented in L5.3."
)

report_lines.append("")

report_lines.append(
    "QC:"
)

report_lines.append(
    f"  Event reconstruction count PASS : "
    f"{qc_event_counts_pass}"
)

report_lines.append(
    f"  Gap-merging logic PASS           : "
    f"{qc_merge_logic_pass}"
)

report_lines.append(
    f"  Eco peak availability PASS       : "
    f"{qc_peak_pass}"
)

report_lines.append(
    f"  Pre-event baseline PASS           : "
    f"{qc_baseline_pass}"
)

report_lines.append(
    f"  OVERALL PASS                      : "
    f"{overall_pass}"
)

report_lines.append("")

report_lines.append(
    "Outputs:"
)

for event_type, fp in catalog_files.items():

    report_lines.append(
        f"  {event_type}: {fp}"
    )


report_lines.append(
    f"  Summary: {summary_file}"
)

report_lines.append(
    f"  Report : {report_file}"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


print(
    "\n".join(
        report_lines
    )
)


# =============================================================================
# 13. CLOSE
# =============================================================================

ds_event.close()
ds_eco.close()


# =============================================================================
# 14. FINAL RESULT
# =============================================================================

print_header("FINAL RESULT")


if overall_pass:

    print(
        "PASS: Step L5.2 independent event lifecycle reconstruction "
        "completed successfully."
    )

else:

    print(
        "WARNING: Step L5.2 completed, "
        "but one or more QC checks failed."
    )


print(
    "\nOUTPUT DIRECTORY:"
)

print(
    OUT_DIR
)

print(
    "\nStep L5.2 Robust V1 completed."
)