# -*- coding: utf-8 -*-
"""
Step_L5_5B5_BasinAnchored_SpatialSensitivity_20km_FINAL.py

L5.5B5
Basin-Anchored Spatial Resilience Sensitivity
20-km FINAL primary spatial scale
Weihe River Basin

Scientific logic
----------------
Basin-scale persistent resilience boundaries were fixed by L5.4C2:

    CDH  : Q90.5958
    HEAT : Q91.8278
    HR   : Q93.0716

L5.5B4 showed that 10-km fixed-Q analysis was limited mainly by insufficient
post-threshold percentile-bin coverage, especially for CDH and HR.

Therefore L5.5B5 aggregates the existing 10-km block x percentile caches
into 20-km blocks and re-estimates the SAME fixed-Q model:

    z = global severity percentile - Q*

    Legacy =
        a
        + beta_pre * z
        + delta_beta * max(0, z)

    beta_post =
        beta_pre + delta_beta

The scientific threshold Q* is NOT changed.

Primary outputs
---------------
beta_pre
beta_post
delta_beta

Legacy(Q*-5)
Legacy(Q*)
Legacy(Q*+5)

PostThreshold_Amplification_5Q
CrossThreshold_Contrast_10Q

Spatial_Sensitivity_Index

No local breakpoint search is performed.

No original CSV.GZ files are opened.

Numerical solver
----------------
MKL-safe weighted normal equations + tiny ridge stabilization.
np.linalg.lstsq / DGELSD is NOT used.


Version: FINAL 20-km
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(
    r".\Weihe database V3\L5_EventEcoFunction"
)

L54_ROOT = (
    ROOT /
    "L5_4_ThresholdAnalysis"
)

L55_ROOT = (
    ROOT /
    "L5_5_SpatialHeterogeneity"
)

CACHE10_ROOT = (
    L55_ROOT /
    "L5_5B3_AggregateCache"
)

CACHE20_ROOT = (
    L55_ROOT /
    "L5_5B5_AggregateCache_20km"
)

CACHE20_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


BASIN_BOUNDARY_FILE = (
    L54_ROOT /
    "L5_4C2_Integrated_Resilience_Boundaries_FINAL.csv"
)


# =============================================================================
# 1. OUTPUT PATHS
# =============================================================================

OUT_ALL = (
    L55_ROOT /
    "L5_5B5_BasinAnchored_SpatialSensitivity_20km_FINAL.csv"
)

OUT_QC = (
    L55_ROOT /
    "L5_5B5_SpatialSensitivity_20km_QC.csv"
)

OUT_SUMMARY = (
    L55_ROOT /
    "L5_5B5_SpatialSensitivity_20km_Summary.csv"
)

OUT_CLASS = (
    L55_ROOT /
    "L5_5B5_Sensitivity_Class_Distribution_20km.csv"
)

OUT_RESPONSE = (
    L55_ROOT /
    "L5_5B5_Response_Type_Distribution_20km.csv"
)

OUT_QUALITY = (
    L55_ROOT /
    "L5_5B5_Model_Quality_20km.csv"
)

OUT_REPORT = (
    L55_ROOT /
    "L5_5B5_BasinAnchored_SpatialSensitivity_20km_Report.txt"
)


# =============================================================================
# 2. EVENT TYPES
# =============================================================================

EVENT_TYPES = [
    "CDH",
    "HEAT",
    "HR",
]


# =============================================================================
# 3. ORIGINAL GRID
# =============================================================================

NY = 414
NX = 703


# =============================================================================
# 4. SOURCE CACHE SCALE = 10 km
# =============================================================================

SOURCE_BLOCK_SIZE = 10

SOURCE_NBY = int(
    np.ceil(
        NY /
        SOURCE_BLOCK_SIZE
    )
)

SOURCE_NBX = int(
    np.ceil(
        NX /
        SOURCE_BLOCK_SIZE
    )
)

SOURCE_NBLOCK = (
    SOURCE_NBY *
    SOURCE_NBX
)


# =============================================================================
# 5. TARGET SCALE = 20 km
# =============================================================================

TARGET_BLOCK_SIZE = 20

TARGET_NBY = int(
    np.ceil(
        NY /
        TARGET_BLOCK_SIZE
    )
)

TARGET_NBX = int(
    np.ceil(
        NX /
        TARGET_BLOCK_SIZE
    )
)

TARGET_NBLOCK = (
    TARGET_NBY *
    TARGET_NBX
)


# =============================================================================
# 6. PERCENTILE SETTINGS
# =============================================================================

N_PERCENTILE_BINS = 100

PERCENTILE_CENTERS = (
    np.arange(
        N_PERCENTILE_BINS,
        dtype=np.float64
    )
    +
    0.5
)


# =============================================================================
# 7. SAMPLE REQUIREMENTS
# =============================================================================

MIN_TOTAL_EVENTS = 500

MIN_Q90_EVENTS = 50

MIN_LEGACY_EVENTS = 300

MIN_EVENTS_PER_BIN = 5

MIN_VALID_BINS = 18


# Critical:
# post-threshold slope should be supported by >=4 percentile bins
MIN_PRE_BINS = 10

MIN_POST_BINS = 4


# =============================================================================
# 8. MODEL SUPPORT
# =============================================================================

MIN_SEGMENTED_R2 = 0.05


# Fixed Q* already independently validated.
# Therefore fixed-Q model is allowed to be slightly worse than simple linear
# locally, but not catastrophically worse.
MIN_REPORTABLE_DELTA_R2 = -0.02


# =============================================================================
# 9. PREDICTION WINDOW
# =============================================================================

WINDOW_Q = 5.0


# =============================================================================
# 10. QC THRESHOLDS
# =============================================================================

# Preferred final primary-scale threshold
PREFERRED_SUPPORTED_FRACTION = 0.60

# Minimum scientifically acceptable threshold
MIN_ACCEPTABLE_SUPPORTED_FRACTION = 0.50


# At least 95% of supported blocks must have finite SSI
MIN_SSI_COMPLETENESS = 0.95


# =============================================================================
# 11. NUMERICAL SETTINGS
# =============================================================================

RIDGE_SCALE = 1e-10

EPS = 1e-8


# =============================================================================
# 12. CACHE SETTINGS
# =============================================================================

CACHE_VERSION = "L5_5B5_20km_v1"

USE_20KM_CACHE_IF_AVAILABLE = True


# =============================================================================
# 13. UTILITIES
# =============================================================================

def section(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )


def source_cache_file(event):

    return (
        CACHE10_ROOT /
        f"L5_5B3_{event}_BlockPercentile_Aggregates_10km.npz"
    )


def source_meta_file(event):

    return (
        CACHE10_ROOT /
        f"L5_5B3_{event}_Cache_Metadata.json"
    )


def target_cache_file(event):

    return (
        CACHE20_ROOT /
        f"L5_5B5_{event}_BlockPercentile_Aggregates_20km.npz"
    )


def target_meta_file(event):

    return (
        CACHE20_ROOT /
        f"L5_5B5_{event}_Cache_Metadata_20km.json"
    )


# =============================================================================
# 14. WEIGHTED R2
# =============================================================================

def weighted_r2(
    y,
    yhat,
    w
):

    mask = (
        np.isfinite(y)
        &
        np.isfinite(yhat)
        &
        np.isfinite(w)
        &
        (w > 0)
    )


    if mask.sum() < 3:

        return np.nan


    y = np.asarray(
        y[mask],
        dtype=np.float64
    )

    yhat = np.asarray(
        yhat[mask],
        dtype=np.float64
    )

    w = np.asarray(
        w[mask],
        dtype=np.float64
    )


    ybar = np.average(
        y,
        weights=w
    )


    sst = np.sum(
        w *
        (y - ybar) ** 2
    )


    sse = np.sum(
        w *
        (y - yhat) ** 2
    )


    if (
        not np.isfinite(sst)
        or
        sst <= 0
    ):

        return np.nan


    return (
        1.0 -
        sse / sst
    )


# =============================================================================
# 15. MKL-SAFE LINEAR MODEL
# =============================================================================

def weighted_linear_fit(
    x,
    y,
    w
):

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
        &
        np.isfinite(w)
        &
        (w > 0)
    )


    x = np.asarray(
        x[mask],
        dtype=np.float64
    )

    y = np.asarray(
        y[mask],
        dtype=np.float64
    )

    w = np.asarray(
        w[mask],
        dtype=np.float64
    )


    if len(x) < 3:

        return {
            "valid": False,
            "reason": "too_few_points",
        }


    x0 = float(
        np.average(
            x,
            weights=w
        )
    )


    xc = (
        x -
        x0
    )


    X = np.column_stack(
        [
            np.ones(
                len(x),
                dtype=np.float64
            ),

            xc,
        ]
    )


    XtW = (
        X.T *
        w
    )


    A = (
        XtW @ X
    )


    b = (
        XtW @ y
    )


    if (
        not np.all(
            np.isfinite(A)
        )
        or
        not np.all(
            np.isfinite(b)
        )
    ):

        return {
            "valid": False,
            "reason": "nonfinite_normal_equation",
        }


    scale = float(
        np.trace(A)
    )


    if (
        not np.isfinite(scale)
        or
        scale <= 0
    ):

        return {
            "valid": False,
            "reason": "invalid_matrix_scale",
        }


    ridge = max(
        scale *
        RIDGE_SCALE,
        1e-12
    )


    A_reg = (
        A
        +
        ridge *
        np.eye(
            2,
            dtype=np.float64
        )
    )


    try:

        beta = np.linalg.solve(
            A_reg,
            b
        )

    except Exception:

        try:

            beta = (
                np.linalg.pinv(
                    A_reg
                )
                @
                b
            )

        except Exception:

            return {
                "valid": False,
                "reason": "solver_failed",
            }


    if not np.all(
        np.isfinite(beta)
    ):

        return {
            "valid": False,
            "reason": "nonfinite_beta",
        }


    yhat = (
        X @ beta
    )


    r2 = weighted_r2(
        y,
        yhat,
        w
    )


    slope = float(
        beta[1]
    )


    intercept = float(
        beta[0]
        -
        slope *
        x0
    )


    return {

        "valid":
            True,

        "reason":
            "ok",

        "intercept":
            intercept,

        "slope":
            slope,

        "r2":
            float(r2)
            if np.isfinite(r2)
            else np.nan,
    }


# =============================================================================
# 16. FIXED BASIN-Q MODEL
# =============================================================================

def fit_fixed_q_model(
    bin_count,
    bin_sum,
    qstar
):

    count = np.asarray(
        bin_count,
        dtype=np.float64
    )


    summ = np.asarray(
        bin_sum,
        dtype=np.float64
    )


    means = np.full(
        len(count),
        np.nan,
        dtype=np.float64
    )


    positive = (
        count > 0
    )


    means[
        positive
    ] = (
        summ[
            positive
        ]
        /
        count[
            positive
        ]
    )


    # ---------------------------------------------------------------------
    # Bin validity
    # ---------------------------------------------------------------------

    good = (
        (count >= MIN_EVENTS_PER_BIN)
        &
        np.isfinite(means)
    )


    n_valid_bins = int(
        good.sum()
    )


    if n_valid_bins < MIN_VALID_BINS:

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "too_few_valid_bins",

            "n_valid_bins":
                n_valid_bins,
        }


    x = (
        PERCENTILE_CENTERS[
            good
        ]
    )


    y = (
        means[
            good
        ]
    )


    w = np.sqrt(
        count[
            good
        ]
    )


    # ---------------------------------------------------------------------
    # Threshold-side information
    # ---------------------------------------------------------------------

    n_pre = int(
        np.sum(
            x < qstar
        )
    )


    n_post = int(
        np.sum(
            x >= qstar
        )
    )


    if n_pre < MIN_PRE_BINS:

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "too_few_pre_threshold_bins",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    if n_post < MIN_POST_BINS:

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "too_few_post_threshold_bins",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    # ---------------------------------------------------------------------
    # Linear reference
    # ---------------------------------------------------------------------

    linear = weighted_linear_fit(
        x,
        y,
        w
    )


    if not linear.get(
        "valid",
        False
    ):

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "linear_reference_failed",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    # ---------------------------------------------------------------------
    # Fixed Q* segmented model
    # ---------------------------------------------------------------------

    z = (
        x -
        float(qstar)
    )


    hinge = np.maximum(
        0.0,
        z
    )


    X = np.column_stack(
        [
            np.ones(
                len(x),
                dtype=np.float64
            ),

            z,

            hinge,
        ]
    )


    XtW = (
        X.T *
        w
    )


    A = (
        XtW @ X
    )


    b = (
        XtW @ y
    )


    if (
        not np.all(
            np.isfinite(A)
        )
        or
        not np.all(
            np.isfinite(b)
        )
    ):

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "nonfinite_normal_equation",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    scale = float(
        np.trace(A)
    )


    if (
        not np.isfinite(scale)
        or
        scale <= 0
    ):

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "invalid_matrix_scale",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    ridge = max(
        scale *
        RIDGE_SCALE,
        1e-12
    )


    penalty = np.diag(
        [
            ridge * 1e-4,
            ridge,
            ridge,
        ]
    )


    A_reg = (
        A +
        penalty
    )


    try:

        beta = np.linalg.solve(
            A_reg,
            b
        )

    except Exception:

        try:

            beta = (
                np.linalg.pinv(
                    A_reg
                )
                @
                b
            )

        except Exception:

            return {

                "fit_valid":
                    False,

                "fit_supported":
                    False,

                "reason":
                    "fixed_q_solver_failed",

                "n_valid_bins":
                    n_valid_bins,

                "n_pre_bins":
                    n_pre,

                "n_post_bins":
                    n_post,
            }


    if not np.all(
        np.isfinite(beta)
    ):

        return {

            "fit_valid":
                False,

            "fit_supported":
                False,

            "reason":
                "nonfinite_beta",

            "n_valid_bins":
                n_valid_bins,

            "n_pre_bins":
                n_pre,

            "n_post_bins":
                n_post,
        }


    yhat = (
        X @ beta
    )


    r2 = weighted_r2(
        y,
        yhat,
        w
    )


    linear_r2 = (
        linear[
            "r2"
        ]
    )


    if (
        np.isfinite(r2)
        and
        np.isfinite(linear_r2)
    ):

        delta_r2 = (
            r2 -
            linear_r2
        )

    else:

        delta_r2 = np.nan


    # ---------------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------------

    intercept_q = float(
        beta[0]
    )


    beta_pre = float(
        beta[1]
    )


    delta_beta = float(
        beta[2]
    )


    beta_post = float(
        beta_pre +
        delta_beta
    )


    # ---------------------------------------------------------------------
    # Threshold-window predictions
    # ---------------------------------------------------------------------

    legacy_minus5 = (
        intercept_q
        -
        beta_pre *
        WINDOW_Q
    )


    legacy_qstar = (
        intercept_q
    )


    legacy_plus5 = (
        intercept_q
        +
        beta_post *
        WINDOW_Q
    )


    pre_change = (
        legacy_qstar -
        legacy_minus5
    )


    post_amplification = (
        legacy_plus5 -
        legacy_qstar
    )


    cross_contrast = (
        legacy_plus5 -
        legacy_minus5
    )


    amplification_ratio = (

        abs(beta_post)
        /
        max(
            abs(beta_pre),
            EPS
        )
    )


    # ---------------------------------------------------------------------
    # Fit support
    # ---------------------------------------------------------------------

    fit_supported = (

        np.isfinite(r2)

        and

        r2 >=
        MIN_SEGMENTED_R2

        and

        (
            not np.isfinite(
                delta_r2
            )

            or

            delta_r2 >=
            MIN_REPORTABLE_DELTA_R2
        )
    )


    return {

        "fit_valid":
            True,

        "fit_supported":
            bool(
                fit_supported
            ),

        "reason":
            "ok"
            if fit_supported
            else "weak_fixed_threshold_fit",

        "Qstar":
            float(
                qstar
            ),

        "n_valid_bins":
            n_valid_bins,

        "n_pre_bins":
            n_pre,

        "n_post_bins":
            n_post,

        "linear_r2":
            float(
                linear_r2
            )
            if np.isfinite(
                linear_r2
            )
            else np.nan,

        "segmented_r2":
            float(
                r2
            )
            if np.isfinite(
                r2
            )
            else np.nan,

        "delta_r2":
            float(
                delta_r2
            )
            if np.isfinite(
                delta_r2
            )
            else np.nan,

        "legacy_at_Qstar":
            intercept_q,

        "beta_pre":
            beta_pre,

        "beta_post":
            beta_post,

        "delta_beta":
            delta_beta,

        "Qminus5":
            float(
                qstar -
                WINDOW_Q
            ),

        "Qplus5":
            float(
                qstar +
                WINDOW_Q
            ),

        "Legacy_Qminus5":
            float(
                legacy_minus5
            ),

        "Legacy_Qstar":
            float(
                legacy_qstar
            ),

        "Legacy_Qplus5":
            float(
                legacy_plus5
            ),

        "PreThreshold_Change_5Q":
            float(
                pre_change
            ),

        "PostThreshold_Amplification_5Q":
            float(
                post_amplification
            ),

        "CrossThreshold_Contrast_10Q":
            float(
                cross_contrast
            ),

        "Amplification_Ratio":
            float(
                amplification_ratio
            ),
    }


# =============================================================================
# 17. ROBUST Z-SCORE
# =============================================================================

def robust_zscore(values):

    x = np.asarray(
        values,
        dtype=np.float64
    )


    out = np.full(
        len(x),
        np.nan,
        dtype=np.float64
    )


    finite = np.isfinite(
        x
    )


    if finite.sum() < 3:

        return out


    xx = x[
        finite
    ]


    med = np.median(
        xx
    )


    mad = np.median(
        np.abs(
            xx -
            med
        )
    )


    scale = (
        1.4826 *
        mad
    )


    if (
        not np.isfinite(scale)
        or
        scale <= EPS
    ):

        scale = np.std(
            xx
        )


    if (
        not np.isfinite(scale)
        or
        scale <= EPS
    ):

        out[
            finite
        ] = 0.0

        return out


    out[
        finite
    ] = (
        xx -
        med
    ) / scale


    return out


# =============================================================================
# 18. CHECK INPUTS
# =============================================================================

section(
    "CHECK INPUT FILES"
)


require_file(
    BASIN_BOUNDARY_FILE
)


print(
    f"PASS basin boundary: "
    f"{BASIN_BOUNDARY_FILE}"
)


for event in EVENT_TYPES:

    require_file(
        source_cache_file(
            event
        )
    )

    require_file(
        source_meta_file(
            event
        )
    )


    print(
        f"PASS {event:<6s} 10-km cache: "
        f"{source_cache_file(event)}"
    )


# =============================================================================
# 19. GRID STRUCTURE
# =============================================================================

section(
    "SPATIAL GRID STRUCTURE"
)


print(
    f"Original raster grid : "
    f"{NY} x {NX}"
)

print(
    f"10-km source blocks  : "
    f"{SOURCE_NBY} x {SOURCE_NBX} "
    f"= {SOURCE_NBLOCK:,}"
)

print(
    f"20-km target blocks  : "
    f"{TARGET_NBY} x {TARGET_NBX} "
    f"= {TARGET_NBLOCK:,}"
)


# =============================================================================
# 20. LOAD BASIN Q*
# =============================================================================

section(
    "LOAD FIXED BASIN RESILIENCE BOUNDARIES"
)


basin_df = pd.read_csv(
    BASIN_BOUNDARY_FILE
)


basin_df[
    "event_type"
] = (
    basin_df[
        "event_type"
    ]
    .astype(str)
    .str.strip()
    .str.upper()
)


BASIN_Q = {}


for event in EVENT_TYPES:

    x = basin_df[
        basin_df[
            "event_type"
        ]
        ==
        event
    ]


    if len(x) != 1:

        raise RuntimeError(
            f"{event}: expected one basin boundary."
        )


    q = float(
        x.iloc[0][
            "integrated_resilience_boundary_percentile"
        ]
    )


    BASIN_Q[event] = q


    print(
        f"{event:<6s} | "
        f"Q* = {q:.4f}"
    )


# =============================================================================
# 21. 10 KM -> 20 KM AGGREGATION
# =============================================================================

section(
    "AGGREGATE 10-km CACHE TO 20-km"
)


AGG20 = {}


for event in EVENT_TYPES:

    print()
    print(
        "-" * 120
    )

    print(
        f"PROCESS {event}"
    )

    print(
        "-" * 120
    )


    target_cache = target_cache_file(
        event
    )


    target_meta = target_meta_file(
        event
    )


    use_existing = (
        USE_20KM_CACHE_IF_AVAILABLE
        and
        target_cache.exists()
        and
        target_meta.exists()
    )


    if use_existing:

        try:

            with open(
                target_meta,
                "r",
                encoding="utf-8"
            ) as f:

                meta = json.load(
                    f
                )


            valid_cache = (

                meta.get(
                    "cache_version"
                )
                ==
                CACHE_VERSION

                and

                int(
                    meta.get(
                        "target_block_size",
                        -1
                    )
                )
                ==
                TARGET_BLOCK_SIZE
            )


        except Exception:

            valid_cache = False


    else:

        valid_cache = False


    # ---------------------------------------------------------------------
    # Load existing 20 km cache
    # ---------------------------------------------------------------------

    if valid_cache:

        dat = np.load(
            target_cache,
            allow_pickle=False
        )


        AGG20[event] = {

            "total_count":
                dat[
                    "total_count"
                ],

            "damage_count":
                dat[
                    "damage_count"
                ],

            "damage_sum":
                dat[
                    "damage_sum"
                ],

            "legacy_count":
                dat[
                    "legacy_count"
                ],

            "legacy_sum":
                dat[
                    "legacy_sum"
                ],
        }


        print(
            f"{event}: existing 20-km cache loaded."
        )


        continue


    # ---------------------------------------------------------------------
    # Load 10 km source
    # ---------------------------------------------------------------------

    dat10 = np.load(
        source_cache_file(
            event
        ),
        allow_pickle=False
    )


    required = [
        "total_count",
        "damage_count",
        "damage_sum",
        "legacy_count",
        "legacy_sum",
    ]


    for name in required:

        if name not in dat10.files:

            raise KeyError(
                f"{event}: 10-km cache missing '{name}'"
            )


    total10 = dat10[
        "total_count"
    ]

    damage_count10 = dat10[
        "damage_count"
    ]

    damage_sum10 = dat10[
        "damage_sum"
    ]

    legacy_count10 = dat10[
        "legacy_count"
    ]

    legacy_sum10 = dat10[
        "legacy_sum"
    ]


    expected_shape = (
        SOURCE_NBLOCK,
        N_PERCENTILE_BINS
    )


    if (
        total10.shape
        !=
        expected_shape
    ):

        raise RuntimeError(
            f"{event}: unexpected source cache shape "
            f"{total10.shape}; expected {expected_shape}"
        )


    # ---------------------------------------------------------------------
    # Initialize target arrays
    # ---------------------------------------------------------------------

    total20 = np.zeros(
        (
            TARGET_NBLOCK,
            N_PERCENTILE_BINS
        ),
        dtype=np.int64
    )


    damage_count20 = np.zeros_like(
        total20
    )


    legacy_count20 = np.zeros_like(
        total20
    )


    damage_sum20 = np.zeros(
        (
            TARGET_NBLOCK,
            N_PERCENTILE_BINS
        ),
        dtype=np.float64
    )


    legacy_sum20 = np.zeros(
        (
            TARGET_NBLOCK,
            N_PERCENTILE_BINS
        ),
        dtype=np.float64
    )


    # ---------------------------------------------------------------------
    # Aggregate every 10-km block into its parent 20-km block
    #
    # Because:
    #
    #   10-km block_y // 2 -> 20-km block_y
    #   10-km block_x // 2 -> 20-km block_x
    #
    # This safely handles the final partial column.
    # ---------------------------------------------------------------------

    for source_id in range(
        SOURCE_NBLOCK
    ):

        by10 = (
            source_id //
            SOURCE_NBX
        )


        bx10 = (
            source_id %
            SOURCE_NBX
        )


        by20 = (
            by10 //
            2
        )


        bx20 = (
            bx10 //
            2
        )


        target_id = (
            by20 *
            TARGET_NBX
            +
            bx20
        )


        total20[
            target_id
        ] += total10[
            source_id
        ]


        damage_count20[
            target_id
        ] += damage_count10[
            source_id
        ]


        damage_sum20[
            target_id
        ] += damage_sum10[
            source_id
        ]


        legacy_count20[
            target_id
        ] += legacy_count10[
            source_id
        ]


        legacy_sum20[
            target_id
        ] += legacy_sum10[
            source_id
        ]


    AGG20[event] = {

        "total_count":
            total20,

        "damage_count":
            damage_count20,

        "damage_sum":
            damage_sum20,

        "legacy_count":
            legacy_count20,

        "legacy_sum":
            legacy_sum20,
    }


    # ---------------------------------------------------------------------
    # Conservation QC
    # ---------------------------------------------------------------------

    source_total = int(
        total10.sum()
    )


    target_total = int(
        total20.sum()
    )


    source_legacy = int(
        legacy_count10.sum()
    )


    target_legacy = int(
        legacy_count20.sum()
    )


    if (
        source_total !=
        target_total
    ):

        raise RuntimeError(
            f"{event}: total-count conservation FAILED."
        )


    if (
        source_legacy !=
        target_legacy
    ):

        raise RuntimeError(
            f"{event}: legacy-count conservation FAILED."
        )


    print(
        f"{event}: aggregation PASS"
    )

    print(
        f"  total events  : "
        f"{source_total:,} -> {target_total:,}"
    )

    print(
        f"  legacy events : "
        f"{source_legacy:,} -> {target_legacy:,}"
    )


    # ---------------------------------------------------------------------
    # Save 20-km cache
    # ---------------------------------------------------------------------

    np.savez_compressed(

        target_cache,

        total_count=
            total20,

        damage_count=
            damage_count20,

        damage_sum=
            damage_sum20,

        legacy_count=
            legacy_count20,

        legacy_sum=
            legacy_sum20,
    )


    metadata = {

        "cache_version":
            CACHE_VERSION,

        "event_type":
            event,

        "source_block_size":
            SOURCE_BLOCK_SIZE,

        "target_block_size":
            TARGET_BLOCK_SIZE,

        "source_nby":
            SOURCE_NBY,

        "source_nbx":
            SOURCE_NBX,

        "target_nby":
            TARGET_NBY,

        "target_nbx":
            TARGET_NBX,

        "n_percentile_bins":
            N_PERCENTILE_BINS,

        "source_cache":
            str(
                source_cache_file(
                    event
                )
            ),
    }


    with open(
        target_meta,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"  saved cache: "
        f"{target_cache}"
    )


# =============================================================================
# 22. FIT FIXED-Q MODELS
# =============================================================================

section(
    "FIT 20-km BASIN-ANCHORED LOCAL LEGACY MODELS"
)


records = []


for event in EVENT_TYPES:

    print()
    print(
        "-" * 120
    )

    print(
        f"FIT {event}"
    )

    print(
        "-" * 120
    )


    qstar = (
        BASIN_Q[event]
    )


    agg = (
        AGG20[event]
    )


    occupied_n = 0

    adequate_n = 0

    valid_n = 0

    supported_n = 0


    for block_id in range(
        TARGET_NBLOCK
    ):

        total_bins = (
            agg[
                "total_count"
            ][
                block_id
            ]
        )


        n_total = int(
            total_bins.sum()
        )


        if n_total == 0:

            continue


        occupied_n += 1


        n_q90 = int(
            total_bins[
                90:
            ].sum()
        )


        legacy_bins = (
            agg[
                "legacy_count"
            ][
                block_id
            ]
        )


        n_legacy = int(
            legacy_bins.sum()
        )


        sampling_adequate = (

            n_total >=
            MIN_TOTAL_EVENTS

            and

            n_q90 >=
            MIN_Q90_EVENTS

            and

            n_legacy >=
            MIN_LEGACY_EVENTS
        )


        if sampling_adequate:

            adequate_n += 1


            fit = fit_fixed_q_model(

                legacy_bins,

                agg[
                    "legacy_sum"
                ][
                    block_id
                ],

                qstar
            )


        else:

            fit = {

                "fit_valid":
                    False,

                "fit_supported":
                    False,

                "reason":
                    "insufficient_samples",
            }


        if fit.get(
            "fit_valid",
            False
        ):

            valid_n += 1


        supported = bool(
            fit.get(
                "fit_supported",
                False
            )
        )


        if supported:

            supported_n += 1


        by = (
            block_id //
            TARGET_NBX
        )


        bx = (
            block_id %
            TARGET_NBX
        )


        row_start = (
            by *
            TARGET_BLOCK_SIZE
        )


        row_end = min(
            (
                by + 1
            )
            *
            TARGET_BLOCK_SIZE
            -
            1,
            NY - 1
        )


        col_start = (
            bx *
            TARGET_BLOCK_SIZE
        )


        col_end = min(
            (
                bx + 1
            )
            *
            TARGET_BLOCK_SIZE
            -
            1,
            NX - 1
        )


        records.append(
            {

                "event_type":
                    event,

                "block_size_km":
                    TARGET_BLOCK_SIZE,

                "block_id":
                    block_id,

                "block_y":
                    by,

                "block_x":
                    bx,

                "row_start":
                    row_start,

                "row_end":
                    row_end,

                "col_start":
                    col_start,

                "col_end":
                    col_end,

                "n_events":
                    n_total,

                "n_q90_events":
                    n_q90,

                "n_legacy_events":
                    n_legacy,

                "sampling_adequate":
                    bool(
                        sampling_adequate
                    ),

                "Qstar_basin":
                    qstar,

                "fit_valid":
                    bool(
                        fit.get(
                            "fit_valid",
                            False
                        )
                    ),

                "fit_supported":
                    supported,

                "fit_reason":
                    fit.get(
                        "reason",
                        "NA"
                    ),

                "n_valid_bins":
                    fit.get(
                        "n_valid_bins",
                        np.nan
                    ),

                "n_pre_bins":
                    fit.get(
                        "n_pre_bins",
                        np.nan
                    ),

                "n_post_bins":
                    fit.get(
                        "n_post_bins",
                        np.nan
                    ),

                "linear_r2":
                    fit.get(
                        "linear_r2",
                        np.nan
                    ),

                "fixedQ_segmented_r2":
                    fit.get(
                        "segmented_r2",
                        np.nan
                    ),

                "delta_r2":
                    fit.get(
                        "delta_r2",
                        np.nan
                    ),

                "legacy_at_Qstar":
                    fit.get(
                        "legacy_at_Qstar",
                        np.nan
                    ),

                "beta_pre":
                    fit.get(
                        "beta_pre",
                        np.nan
                    ),

                "beta_post":
                    fit.get(
                        "beta_post",
                        np.nan
                    ),

                "delta_beta":
                    fit.get(
                        "delta_beta",
                        np.nan
                    ),

                "Qminus5":
                    fit.get(
                        "Qminus5",
                        np.nan
                    ),

                "Qplus5":
                    fit.get(
                        "Qplus5",
                        np.nan
                    ),

                "Legacy_Qminus5":
                    fit.get(
                        "Legacy_Qminus5",
                        np.nan
                    ),

                "Legacy_Qstar":
                    fit.get(
                        "Legacy_Qstar",
                        np.nan
                    ),

                "Legacy_Qplus5":
                    fit.get(
                        "Legacy_Qplus5",
                        np.nan
                    ),

                "PreThreshold_Change_5Q":
                    fit.get(
                        "PreThreshold_Change_5Q",
                        np.nan
                    ),

                "PostThreshold_Amplification_5Q":
                    fit.get(
                        "PostThreshold_Amplification_5Q",
                        np.nan
                    ),

                "CrossThreshold_Contrast_10Q":
                    fit.get(
                        "CrossThreshold_Contrast_10Q",
                        np.nan
                    ),

                "Amplification_Ratio":
                    fit.get(
                        "Amplification_Ratio",
                        np.nan
                    ),
            }
        )


    print(
        f"{event}: "
        f"occupied={occupied_n:,} | "
        f"adequate={adequate_n:,} | "
        f"fit valid={valid_n:,} | "
        f"supported={supported_n:,}"
    )


# =============================================================================
# 23. BUILD DATAFRAME
# =============================================================================

df = pd.DataFrame(
    records
)


# =============================================================================
# 24. FIT REASONS
# =============================================================================

section(
    "FIT REASONS"
)


reason_df = (

    df
    .groupby(
        [
            "event_type",
            "fit_reason",
        ]
    )
    .size()
    .reset_index(
        name="n_blocks"
    )
)


print(
    reason_df.to_string(
        index=False
    )
)


# =============================================================================
# 25. SPATIAL STANDARDIZATION
# =============================================================================

section(
    "CALCULATE 20-km SPATIAL SENSITIVITY INDEX"
)


df[
    "Z_PostAmplification"
] = np.nan


df[
    "Z_DeltaBeta"
] = np.nan


df[
    "Z_BetaPost"
] = np.nan


df[
    "Spatial_Sensitivity_Index"
] = np.nan


for event in EVENT_TYPES:

    mask = (
        (df["event_type"] == event)
        &
        df[
            "fit_supported"
        ].astype(bool)
    )


    amp = pd.to_numeric(
        df.loc[
            mask,
            "PostThreshold_Amplification_5Q"
        ],
        errors="coerce"
    ).to_numpy()


    db = pd.to_numeric(
        df.loc[
            mask,
            "delta_beta"
        ],
        errors="coerce"
    ).to_numpy()


    bp = pd.to_numeric(
        df.loc[
            mask,
            "beta_post"
        ],
        errors="coerce"
    ).to_numpy()


    z_amp = robust_zscore(
        amp
    )


    z_db = robust_zscore(
        db
    )


    z_bp = robust_zscore(
        bp
    )


    df.loc[
        mask,
        "Z_PostAmplification"
    ] = z_amp


    df.loc[
        mask,
        "Z_DeltaBeta"
    ] = z_db


    df.loc[
        mask,
        "Z_BetaPost"
    ] = z_bp


    matrix = np.column_stack(
        [
            z_amp,
            z_db,
            z_bp,
        ]
    )


    valid_components = np.isfinite(
        matrix
    ).sum(
        axis=1
    )


    ssi = np.full(
        len(matrix),
        np.nan,
        dtype=np.float64
    )


    valid_rows = (
        valid_components >= 2
    )


    ssi[
        valid_rows
    ] = np.nanmean(
        matrix[
            valid_rows
        ],
        axis=1
    )


    df.loc[
        mask,
        "Spatial_Sensitivity_Index"
    ] = ssi


# =============================================================================
# 26. SENSITIVITY CLASS
# =============================================================================

section(
    "CLASSIFY SPATIAL SENSITIVITY"
)


df[
    "spatial_sensitivity_class"
] = "UNRESOLVED"


for event in EVENT_TYPES:

    mask = (
        (df["event_type"] == event)
        &
        df[
            "fit_supported"
        ].astype(bool)
        &
        np.isfinite(
            pd.to_numeric(
                df[
                    "Spatial_Sensitivity_Index"
                ],
                errors="coerce"
            )
        )
    )


    s = pd.to_numeric(
        df.loc[
            mask,
            "Spatial_Sensitivity_Index"
        ],
        errors="coerce"
    )


    classes = np.where(

        s >= 1.0,

        "VERY_HIGH_SENSITIVITY",

        np.where(

            s >= 0.25,

            "HIGH_SENSITIVITY",

            np.where(

                s > -0.25,

                "INTERMEDIATE",

                np.where(

                    s > -1.0,

                    "LOW_SENSITIVITY",

                    "VERY_LOW_SENSITIVITY"
                )
            )
        )
    )


    df.loc[
        mask,
        "spatial_sensitivity_class"
    ] = classes


# =============================================================================
# 27. RESPONSE TYPE
# =============================================================================

section(
    "CLASSIFY THRESHOLD RESPONSE TYPE"
)


df[
    "threshold_response_type"
] = "UNRESOLVED"


supported = (
    df[
        "fit_supported"
    ].astype(bool)
)


delta = pd.to_numeric(
    df[
        "delta_beta"
    ],
    errors="coerce"
)


# Accelerating
mask = (
    supported
    &
    np.isfinite(delta)
    &
    (delta > 0.01)
)


df.loc[
    mask,
    "threshold_response_type"
] = "ACCELERATING"


# Near linear
mask = (
    supported
    &
    np.isfinite(delta)
    &
    (
        np.abs(delta)
        <=
        0.01
    )
)


df.loc[
    mask,
    "threshold_response_type"
] = "NEAR_LINEAR"


# Decelerating
mask = (
    supported
    &
    np.isfinite(delta)
    &
    (delta < -0.01)
)


df.loc[
    mask,
    "threshold_response_type"
] = "DECELERATING"


# =============================================================================
# 28. SAVE EVENT-SPECIFIC TABLES
# =============================================================================

section(
    "SAVE EVENT-SPECIFIC OUTPUTS"
)


for event in EVENT_TYPES:

    outfile = (
        L55_ROOT /
        f"L5_5B5_{event}_BasinAnchored_SpatialSensitivity_20km_FINAL.csv"
    )


    df[
        df[
            "event_type"
        ]
        ==
        event
    ].to_csv(
        outfile,
        index=False
    )


    print(
        f"{event}: {outfile}"
    )


# =============================================================================
# 29. SAVE FULL TABLE
# =============================================================================

df.to_csv(
    OUT_ALL,
    index=False
)


# =============================================================================
# 30. SUMMARY
# =============================================================================

section(
    "20-km BASIN-ANCHORED SPATIAL SENSITIVITY SUMMARY"
)


summary_records = []


for event in EVENT_TYPES:

    x = df[
        df[
            "event_type"
        ]
        ==
        event
    ].copy()


    occupied = len(
        x
    )


    adequate = int(
        x[
            "sampling_adequate"
        ].sum()
    )


    valid = int(
        x[
            "fit_valid"
        ].sum()
    )


    supported_n = int(
        x[
            "fit_supported"
        ].sum()
    )


    sx = x[
        x[
            "fit_supported"
        ]
    ].copy()


    def finite_series(col):

        vals = pd.to_numeric(
            sx[
                col
            ],
            errors="coerce"
        )


        return vals[
            np.isfinite(
                vals
            )
        ]


    def median_col(col):

        vals = finite_series(
            col
        )


        return (
            float(
                vals.median()
            )
            if len(vals) > 0
            else np.nan
        )


    def q25_col(col):

        vals = finite_series(
            col
        )


        return (
            float(
                vals.quantile(
                    0.25
                )
            )
            if len(vals) > 0
            else np.nan
        )


    def q75_col(col):

        vals = finite_series(
            col
        )


        return (
            float(
                vals.quantile(
                    0.75
                )
            )
            if len(vals) > 0
            else np.nan
        )


    accelerating_n = int(
        (
            sx[
                "threshold_response_type"
            ]
            ==
            "ACCELERATING"
        ).sum()
    )


    positive_amp_n = int(
        (
            pd.to_numeric(
                sx[
                    "PostThreshold_Amplification_5Q"
                ],
                errors="coerce"
            )
            >
            0
        ).sum()
    )


    summary_records.append(
        {

            "event_type":
                event,

            "Qstar_basin":
                BASIN_Q[event],

            "occupied_blocks":
                occupied,

            "sampling_adequate_blocks":
                adequate,

            "fit_valid_blocks":
                valid,

            "fit_supported_blocks":
                supported_n,

            "supported_fraction_adequate":
                supported_n /
                max(
                    adequate,
                    1
                ),

            "median_beta_pre":
                median_col(
                    "beta_pre"
                ),

            "median_beta_post":
                median_col(
                    "beta_post"
                ),

            "median_delta_beta":
                median_col(
                    "delta_beta"
                ),

            "delta_beta_Q25":
                q25_col(
                    "delta_beta"
                ),

            "delta_beta_Q75":
                q75_col(
                    "delta_beta"
                ),

            "median_PostThreshold_Amplification_5Q":
                median_col(
                    "PostThreshold_Amplification_5Q"
                ),

            "PostAmplification_Q25":
                q25_col(
                    "PostThreshold_Amplification_5Q"
                ),

            "PostAmplification_Q75":
                q75_col(
                    "PostThreshold_Amplification_5Q"
                ),

            "median_CrossThreshold_Contrast_10Q":
                median_col(
                    "CrossThreshold_Contrast_10Q"
                ),

            "median_Amplification_Ratio":
                median_col(
                    "Amplification_Ratio"
                ),

            "accelerating_blocks":
                accelerating_n,

            "accelerating_fraction_supported":
                accelerating_n /
                max(
                    supported_n,
                    1
                ),

            "positive_post_amplification_blocks":
                positive_amp_n,

            "positive_post_amplification_fraction":
                positive_amp_n /
                max(
                    supported_n,
                    1
                ),
        }
    )


summary_df = pd.DataFrame(
    summary_records
)


summary_df.to_csv(
    OUT_SUMMARY,
    index=False
)


print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 31. CLASS DISTRIBUTION
# =============================================================================

section(
    "20-km SPATIAL SENSITIVITY CLASS DISTRIBUTION"
)


class_df = (

    df
    .groupby(
        [
            "event_type",
            "spatial_sensitivity_class",
        ]
    )
    .size()
    .reset_index(
        name="n_blocks"
    )
)


class_df.to_csv(
    OUT_CLASS,
    index=False
)


print(
    class_df.to_string(
        index=False
    )
)


# =============================================================================
# 32. RESPONSE DISTRIBUTION
# =============================================================================

section(
    "20-km THRESHOLD RESPONSE TYPE DISTRIBUTION"
)


response_df = (

    df
    .groupby(
        [
            "event_type",
            "threshold_response_type",
        ]
    )
    .size()
    .reset_index(
        name="n_blocks"
    )
)


response_df.to_csv(
    OUT_RESPONSE,
    index=False
)


print(
    response_df.to_string(
        index=False
    )
)


# =============================================================================
# 33. MODEL QUALITY
# =============================================================================

section(
    "20-km MODEL QUALITY"
)


quality_records = []


for event in EVENT_TYPES:

    x = df[
        (
            df[
                "event_type"
            ]
            ==
            event
        )
        &
        df[
            "fit_supported"
        ]
    ].copy()


    r2 = pd.to_numeric(
        x[
            "fixedQ_segmented_r2"
        ],
        errors="coerce"
    )


    dr2 = pd.to_numeric(
        x[
            "delta_r2"
        ],
        errors="coerce"
    )


    r2 = r2[
        np.isfinite(
            r2
        )
    ]


    dr2 = dr2[
        np.isfinite(
            dr2
        )
    ]


    quality_records.append(
        {

            "event_type":
                event,

            "supported_blocks":
                len(
                    x
                ),

            "median_R2":
                float(
                    r2.median()
                )
                if len(r2) > 0
                else np.nan,

            "R2_Q25":
                float(
                    r2.quantile(
                        0.25
                    )
                )
                if len(r2) > 0
                else np.nan,

            "R2_Q75":
                float(
                    r2.quantile(
                        0.75
                    )
                )
                if len(r2) > 0
                else np.nan,

            "median_delta_R2":
                float(
                    dr2.median()
                )
                if len(dr2) > 0
                else np.nan,

            "fraction_R2_ge_0_20":
                float(
                    np.mean(
                        r2 >= 0.20
                    )
                )
                if len(r2) > 0
                else np.nan,

            "fraction_R2_ge_0_10":
                float(
                    np.mean(
                        r2 >= 0.10
                    )
                )
                if len(r2) > 0
                else np.nan,
        }
    )


quality_df = pd.DataFrame(
    quality_records
)


quality_df.to_csv(
    OUT_QUALITY,
    index=False
)


print(
    quality_df.to_string(
        index=False
    )
)


# =============================================================================
# 34. POST-THRESHOLD BIN DIAGNOSTIC
# =============================================================================

section(
    "POST-THRESHOLD BIN DIAGNOSTIC"
)


postbin_records = []


for event in EVENT_TYPES:

    x = df[
        df[
            "event_type"
        ]
        ==
        event
    ].copy()


    adequate = x[
        x[
            "sampling_adequate"
        ]
    ].copy()


    n_adequate = len(
        adequate
    )


    too_few_post = int(
        (
            adequate[
                "fit_reason"
            ]
            ==
            "too_few_post_threshold_bins"
        ).sum()
    )


    too_few_valid = int(
        (
            adequate[
                "fit_reason"
            ]
            ==
            "too_few_valid_bins"
        ).sum()
    )


    postbin_records.append(
        {

            "event_type":
                event,

            "adequate_blocks":
                n_adequate,

            "too_few_post_threshold_bins":
                too_few_post,

            "fraction_too_few_post_bins":
                too_few_post /
                max(
                    n_adequate,
                    1
                ),

            "too_few_valid_bins":
                too_few_valid,
        }
    )


postbin_df = pd.DataFrame(
    postbin_records
)


print(
    postbin_df.to_string(
        index=False
    )
)


# =============================================================================
# 35. FINAL QC
# =============================================================================

section(
    "FINAL L5.5B5 QC"
)


qc_records = []


for event in EVENT_TYPES:

    x = df[
        df[
            "event_type"
        ]
        ==
        event
    ].copy()


    adequate = int(
        x[
            "sampling_adequate"
        ].sum()
    )


    supported_n = int(
        x[
            "fit_supported"
        ].sum()
    )


    supported_fraction = (
        supported_n /
        max(
            adequate,
            1
        )
    )


    finite_ssi = int(
        np.isfinite(
            pd.to_numeric(
                x[
                    "Spatial_Sensitivity_Index"
                ],
                errors="coerce"
            )
        ).sum()
    )


    ssi_fraction = (
        finite_ssi /
        max(
            supported_n,
            1
        )
    )


    preferred_pass = bool(
        supported_fraction >=
        PREFERRED_SUPPORTED_FRACTION
    )


    acceptable_pass = bool(
        supported_fraction >=
        MIN_ACCEPTABLE_SUPPORTED_FRACTION
    )


    metric_pass = bool(
        ssi_fraction >=
        MIN_SSI_COMPLETENESS
    )


    qc_records.append(
        {

            "event_type":
                event,

            "adequate_blocks":
                adequate,

            "supported_blocks":
                supported_n,

            "supported_fraction":
                supported_fraction,

            "finite_SSI_blocks":
                finite_ssi,

            "finite_SSI_fraction_supported":
                ssi_fraction,

            "preferred_coverage_pass":
                preferred_pass,

            "minimum_coverage_pass":
                acceptable_pass,

            "metric_pass":
                metric_pass,
        }
    )


qc_df = pd.DataFrame(
    qc_records
)


qc_df.to_csv(
    OUT_QC,
    index=False
)


print(
    qc_df.to_string(
        index=False
    )
)


preferred_all = bool(
    qc_df[
        "preferred_coverage_pass"
    ].all()
)


minimum_all = bool(
    qc_df[
        "minimum_coverage_pass"
    ].all()
)


metric_all = bool(
    qc_df[
        "metric_pass"
    ].all()
)


# -------------------------------------------------------------------------
# Final status
# -------------------------------------------------------------------------

if (
    preferred_all
    and
    metric_all
):

    FINAL_STATUS = "PASS"


elif (
    minimum_all
    and
    metric_all
):

    FINAL_STATUS = "CONDITIONAL_PASS"


else:

    FINAL_STATUS = "CHECK_REQUIRED"


print()

print(
    f"Preferred coverage >= "
    f"{PREFERRED_SUPPORTED_FRACTION:.2f} "
    f"for all events = {preferred_all}"
)

print(
    f"Minimum coverage >= "
    f"{MIN_ACCEPTABLE_SUPPORTED_FRACTION:.2f} "
    f"for all events = {minimum_all}"
)

print(
    f"Spatial metric completeness = "
    f"{metric_all}"
)

print()

print(
    f"FINAL STATUS: "
    f"{FINAL_STATUS}"
)


if FINAL_STATUS == "PASS":

    print(
        "PASS: 20-km basin-anchored spatial sensitivity "
        "is suitable as the primary spatial analysis."
    )


elif FINAL_STATUS == "CONDITIONAL_PASS":

    print(
        "CONDITIONAL PASS: 20-km results meet the minimum "
        "coverage criterion but one or more event types do not "
        "reach the preferred 60% coverage."
    )


else:

    print(
        "CHECK REQUIRED: spatial support remains insufficient "
        "for at least one event type."
    )


# =============================================================================
# 36. WRITE REPORT
# =============================================================================

section(
    "WRITE FINAL REPORT"
)


lines = []


lines.append(
    "Step L5.5B5 Basin-Anchored Spatial Resilience Sensitivity | 20-km FINAL"
)

lines.append(
    "=" * 110
)

lines.append("")

lines.append(
    "Primary spatial scale: 20 km"
)

lines.append("")

lines.append(
    "Reason for scale selection:"
)

lines.append(
    "  L5.5B4 at 10 km showed inadequate post-threshold percentile-bin "
    "coverage for CDH and HR."
)

lines.append(
    "  L5.5B5 aggregates 10-km block-percentile caches into 20-km blocks "
    "without rescanning original event databases."
)

lines.append("")

lines.append(
    "Scientific threshold definition:"
)

lines.append(
    "  basin-scale Q* values fixed from L5.4C2"
)

lines.append(
    "  no unrestricted local breakpoint search"
)

lines.append("")

lines.append(
    "Fixed thresholds:"
)


for event in EVENT_TYPES:

    lines.append(
        f"  {event:<6s}: "
        f"Q{BASIN_Q[event]:.2f}"
    )


lines.append("")

lines.append(
    "Fixed-Q model:"
)

lines.append(
    "  z = global severity percentile - Q*"
)

lines.append(
    "  Legacy = a + beta_pre*z + delta_beta*max(0,z)"
)

lines.append(
    "  beta_post = beta_pre + delta_beta"
)

lines.append("")

lines.append(
    "Primary spatial metrics:"
)

lines.append(
    "  beta_pre"
)

lines.append(
    "  beta_post"
)

lines.append(
    "  delta_beta"
)

lines.append(
    "  PostThreshold_Amplification_5Q"
)

lines.append(
    "  CrossThreshold_Contrast_10Q"
)

lines.append(
    "  Spatial_Sensitivity_Index"
)

lines.append("")

lines.append(
    "Fit reasons:"
)

lines.append(
    reason_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "20-km summary:"
)

lines.append(
    summary_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "Sensitivity class distribution:"
)

lines.append(
    class_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "Threshold response distribution:"
)

lines.append(
    response_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "Model quality:"
)

lines.append(
    quality_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "Post-threshold bin diagnostic:"
)

lines.append(
    postbin_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    "Final QC:"
)

lines.append(
    qc_df.to_string(
        index=False
    )
)

lines.append("")

lines.append(
    f"Preferred coverage pass = "
    f"{preferred_all}"
)

lines.append(
    f"Minimum coverage pass = "
    f"{minimum_all}"
)

lines.append(
    f"Spatial metric completeness = "
    f"{metric_all}"
)

lines.append("")

lines.append(
    f"FINAL STATUS: "
    f"{FINAL_STATUS}"
)


if FINAL_STATUS == "PASS":

    lines.append(
        "The 20-km basin-anchored analysis is recommended "
        "as the primary spatial sensitivity product for L5.5C."
    )


elif FINAL_STATUS == "CONDITIONAL_PASS":

    lines.append(
        "The 20-km basin-anchored analysis is usable, "
        "but spatial coverage limitations should be retained "
        "in sensitivity/uncertainty interpretation."
    )


else:

    lines.append(
        "Do not proceed to final L5.5C mapping until "
        "the remaining spatial-support problem is diagnosed."
    )


report_text = "\n".join(
    lines
)


with open(
    OUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        report_text
    )


print(
    report_text
)


# =============================================================================
# 37. END
# =============================================================================

section(
    "STEP L5.5B5 COMPLETED"
)


print(
    f"All spatial sensitivity : "
    f"{OUT_ALL}"
)

print(
    f"QC                      : "
    f"{OUT_QC}"
)

print(
    f"Summary                 : "
    f"{OUT_SUMMARY}"
)

print(
    f"Class distribution      : "
    f"{OUT_CLASS}"
)

print(
    f"Response distribution   : "
    f"{OUT_RESPONSE}"
)

print(
    f"Model quality           : "
    f"{OUT_QUALITY}"
)

print(
    f"Report                  : "
    f"{OUT_REPORT}"
)

print()

print(
    f"20-km cache directory   : "
    f"{CACHE20_ROOT}"
)

print()


if FINAL_STATUS == "PASS":

    print(
        "PASS: L5.5B5 20-km FINAL spatial sensitivity completed."
    )


elif FINAL_STATUS == "CONDITIONAL_PASS":

    print(
        "CONDITIONAL PASS: inspect coverage before final L5.5C mapping."
    )


else:

    print(
        "CHECK REQUIRED: inspect L5.5B5 before proceeding."
    )