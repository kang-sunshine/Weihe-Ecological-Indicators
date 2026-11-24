# -*- coding: utf-8 -*-
"""
Step L5.4A
Severity–Response Nonlinear Threshold Analysis
FAST V1

Weihe River Basin ecological resilience pipeline

Scientific objectives
---------------------
For CDH / HEAT / HR independently:

1. Severity -> Acomp damage
2. Severity -> Acomp peak lag
3. Severity -> Acomp legacy
4. Severity -> probability of positive functional damage

Main strategy
-------------
Full event files are NOT directly fitted.

Instead:
    full CSV.GZ scan
        ->
    clean scientifically valid samples
        ->
    severity quantile bins
        ->
    robust bin-level medians / probabilities
        ->
    weighted linear vs one-breakpoint segmented regression
        ->
    breakpoint bootstrap CI

Advantages
----------
- fast
- robust against millions of pseudo-replicated pixel-events
- avoids artificial significance from huge N
- produces publication-ready threshold tables

No NetCDF access.
"""

import os
import gc
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

BASE_DIR = r".\Weihe database V3"

L53_DIR = os.path.join(
    BASE_DIR,
    "L5_EventEcoFunction",
    "L5_3_SeverityResilience"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "L5_EventEcoFunction",
    "L5_4_ThresholdAnalysis"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


FILES = {

    "CDH": os.path.join(
        L53_DIR,
        "L5_3_CDH_Event_Severity_Resilience_Metrics.csv.gz"
    ),

    "HEAT": os.path.join(
        L53_DIR,
        "L5_3_HEAT_Event_Severity_Resilience_Metrics.csv.gz"
    ),

    "HR": os.path.join(
        L53_DIR,
        "L5_3_HR_Event_Severity_Resilience_Metrics.csv.gz"
    )
}


# =============================================================================
# 1. PERFORMANCE SETTINGS
# =============================================================================

CHUNK_SIZE = 1_000_000

# Used only to determine stable severity quantile boundaries.
SEVERITY_SAMPLE_MAX = 500_000

# Number of quantile bins.
# 80 is a good balance between smoothness and threshold resolution.
N_BINS = 80

# Minimum valid observations per bin.
MIN_BIN_N = 500

# Breakpoint cannot occur at extreme tails.
BREAK_MIN_QUANTILE = 0.10
BREAK_MAX_QUANTILE = 0.90

# Bootstrap operates on the ~80 BINS, not millions of raw events.
N_BOOT = 500

RANDOM_SEED = 20260908

RNG = np.random.default_rng(
    RANDOM_SEED
)


# =============================================================================
# 2. ANALYSIS DEFINITIONS
# =============================================================================

SEVERITY = "climate_cumulative_severity"

RESPONSES = {

    # Main resistance / functional damage response
    "Acomp_damage": {
        "column": "Acomp_damage",
        "sample_rule": "damage"
    },

    # Ecological response lag
    "Acomp_peak_lag": {
        "column": "Acomp_peak_lag_days",
        "sample_rule": "lag"
    },

    # Post-event accumulated functional legacy
    "Acomp_legacy": {
        "column": "Acomp_legacy_anomaly_days",
        "sample_rule": "legacy"
    }
}


# =============================================================================
# 3. HELPERS
# =============================================================================

def header(title):

    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def bounded_sample(
    old,
    new,
    max_n
):

    x = np.asarray(
        new,
        dtype=np.float64
    )

    x = x[
        np.isfinite(x)
    ]

    if x.size == 0:
        return old


    if x.size > max_n:

        idx = RNG.choice(
            x.size,
            max_n,
            replace=False
        )

        x = x[idx]


    if old.size == 0:

        if x.size <= max_n:
            return x.copy()

        idx = RNG.choice(
            x.size,
            max_n,
            replace=False
        )

        return x[idx]


    z = np.concatenate(
        [
            old,
            x
        ]
    )


    if z.size <= max_n:
        return z


    idx = RNG.choice(
        z.size,
        max_n,
        replace=False
    )

    return z[idx]


def valid_base_mask(df):
    """
    Core scientific filter.

    clean pre-event window is mandatory because otherwise
    severity-response relationships are contaminated by prior
    climate extremes.
    """

    return (
        (
            df[
                "clean_pre_any_event"
            ].to_numpy()
            == 1
        )
        &
        np.isfinite(
            df[
                SEVERITY
            ].to_numpy(
                dtype=float
            )
        )
    )


def response_mask(
    df,
    response_name
):

    base = valid_base_mask(
        df
    )


    if response_name == "Acomp_damage":

        # Main threshold model:
        # only genuine functional damage.
        return (
            base
            &
            (
                df[
                    "valid_resistance_sample"
                ].to_numpy()
                == 1
            )
            &
            np.isfinite(
                df[
                    "Acomp_damage"
                ].to_numpy(
                    dtype=float
                )
            )
            &
            (
                df[
                    "Acomp_damage"
                ].to_numpy(
                    dtype=float
                )
                >
                0
            )
        )


    elif response_name == "Acomp_peak_lag":

        return (
            base
            &
            np.isfinite(
                df[
                    "Acomp_peak_lag_days"
                ].to_numpy(
                    dtype=float
                )
            )
        )


    elif response_name == "Acomp_legacy":

        return (
            base
            &
            np.isfinite(
                df[
                    "Acomp_legacy_anomaly_days"
                ].to_numpy(
                    dtype=float
                )
            )
        )


    raise ValueError(
        response_name
    )


# =============================================================================
# 4. WEIGHTED REGRESSION
# =============================================================================

def weighted_linear_fit(
    x,
    y,
    w
):
    """
    Weighted y = b0 + b1*x
    """

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    w = np.asarray(
        w,
        dtype=float
    )


    X = np.column_stack(
        [
            np.ones_like(x),
            x
        ]
    )


    sw = np.sqrt(w)


    Xw = (
        X
        *
        sw[:, None]
    )

    yw = (
        y
        *
        sw
    )


    beta, _, _, _ = np.linalg.lstsq(
        Xw,
        yw,
        rcond=None
    )


    pred = (
        X
        @
        beta
    )


    rss = np.sum(
        w
        *
        (
            y - pred
        ) ** 2
    )


    return (
        beta,
        pred,
        rss
    )


def weighted_segmented_fit(
    x,
    y,
    w,
    breakpoint
):
    """
    Continuous one-breakpoint segmented regression:

        y = b0
            + b1*x
            + b2*max(0, x-c)

    Pre-break slope:
        b1

    Post-break slope:
        b1 + b2
    """

    hinge = np.maximum(
        0.0,
        x - breakpoint
    )


    X = np.column_stack(
        [
            np.ones_like(x),
            x,
            hinge
        ]
    )


    sw = np.sqrt(w)


    Xw = (
        X
        *
        sw[:, None]
    )

    yw = (
        y
        *
        sw
    )


    beta, _, _, _ = np.linalg.lstsq(
        Xw,
        yw,
        rcond=None
    )


    pred = (
        X
        @
        beta
    )


    rss = np.sum(
        w
        *
        (
            y - pred
        ) ** 2
    )


    return (
        beta,
        pred,
        rss
    )


def calculate_aic_bic(
    rss,
    n,
    k
):

    if (
        rss <= 0
        or
        n <= k
    ):

        return (
            np.nan,
            np.nan
        )


    rssn = (
        rss
        /
        n
    )


    aic = (
        n
        *
        np.log(
            rssn
        )
        +
        2 * k
    )


    bic = (
        n
        *
        np.log(
            rssn
        )
        +
        k
        *
        np.log(n)
    )


    return (
        aic,
        bic
    )


# =============================================================================
# 5. BREAKPOINT SEARCH
# =============================================================================

def find_best_breakpoint(
    x,
    y,
    w
):

    order = np.argsort(
        x
    )

    x = x[order]
    y = y[order]
    w = w[order]


    n = len(x)


    lo = int(
        np.floor(
            BREAK_MIN_QUANTILE
            *
            n
        )
    )

    hi = int(
        np.ceil(
            BREAK_MAX_QUANTILE
            *
            n
        )
    )


    lo = max(
        lo,
        4
    )

    hi = min(
        hi,
        n - 4
    )


    candidates = x[
        lo:hi
    ]


    best = None


    for c in candidates:

        beta, pred, rss = (
            weighted_segmented_fit(
                x,
                y,
                w,
                c
            )
        )


        if (
            best is None
            or
            rss < best[
                "rss"
            ]
        ):

            best = {

                "breakpoint":
                    float(c),

                "beta":
                    beta,

                "pred":
                    pred,

                "rss":
                    float(rss)
            }


    return (
        x,
        y,
        w,
        best
    )


# =============================================================================
# 6. BOOTSTRAP BREAKPOINT
# =============================================================================

def bootstrap_breakpoint(
    x,
    y,
    w,
    n_boot=N_BOOT
):

    values = []

    n = len(x)


    if n < 15:
        return np.array([])


    # Sampling bin-level observations, weighted by original bin N.
    prob = (
        w
        /
        np.sum(w)
    )


    for _ in range(
        n_boot
    ):

        idx = RNG.choice(
            n,
            size=n,
            replace=True,
            p=prob
        )


        xb = x[idx]
        yb = y[idx]
        wb = w[idx]


        # duplicate x values may exist after bootstrap
        # still numerically valid for segmented fit

        try:

            _, _, _, best = (
                find_best_breakpoint(
                    xb,
                    yb,
                    wb
                )
            )


            if best is not None:

                values.append(
                    best[
                        "breakpoint"
                    ]
                )


        except Exception:

            continue


    return np.asarray(
        values,
        dtype=float
    )


# =============================================================================
# 7. FIRST PASS:
#    BUILD SEVERITY QUANTILE BOUNDARIES
# =============================================================================

header(
    "L5.4A PASS 1 | BUILD SEVERITY DISTRIBUTIONS"
)


severity_samples = {}


READ_COLS_PASS1 = [
    SEVERITY,
    "clean_pre_any_event"
]


for event_type, filepath in FILES.items():


    print(
        f"\n{event_type}"
    )


    sample = np.empty(
        0,
        dtype=np.float64
    )


    reader = pd.read_csv(

        filepath,

        compression="gzip",

        usecols=READ_COLS_PASS1,

        chunksize=CHUNK_SIZE,

        low_memory=False
    )


    total_clean = 0


    for i, df in enumerate(
        reader,
        start=1
    ):


        severity = pd.to_numeric(
            df[
                SEVERITY
            ],
            errors="coerce"
        ).to_numpy(
            dtype=np.float64,
            copy=False
        )


        clean = pd.to_numeric(
            df[
                "clean_pre_any_event"
            ],
            errors="coerce"
        ).to_numpy(
            dtype=np.float64,
            copy=False
        )


        good = (
            np.isfinite(
                severity
            )
            &
            (
                clean == 1
            )
        )


        total_clean += int(
            good.sum()
        )


        sample = bounded_sample(
            sample,
            severity[
                good
            ],
            SEVERITY_SAMPLE_MAX
        )


        print(
            f"{event_type} "
            f"| chunk={i:03d} "
            f"| clean severity={total_clean:,}",
            end="\r"
        )


        del df
        gc.collect()


    severity_samples[
        event_type
    ] = sample


    print(
        f"\n{event_type}: "
        f"sample={len(sample):,}"
    )


# =============================================================================
# 8. BIN EDGES
# =============================================================================

bin_edges = {}


for event_type, sample in severity_samples.items():


    if len(sample) < 1000:

        raise RuntimeError(
            f"Insufficient severity sample for {event_type}"
        )


    q = np.linspace(
        0,
        1,
        N_BINS + 1
    )


    edges = np.quantile(
        sample,
        q
    )


    # Remove duplicate quantiles
    edges = np.unique(
        edges
    )


    if len(edges) < 20:

        raise RuntimeError(
            f"Too few unique severity bins for {event_type}"
        )


    # Ensure max value falls inside final bin.
    edges[0] = -np.inf
    edges[-1] = np.inf


    bin_edges[
        event_type
    ] = edges


# =============================================================================
# 9. SECOND PASS:
#    BIN FULL DATA
# =============================================================================

header(
    "L5.4A PASS 2 | BUILD FULL-DATA DOSE-RESPONSE BINS"
)


READ_COLS_PASS2 = [

    SEVERITY,

    "clean_pre_any_event",

    "valid_resistance_sample",

    "Acomp_damage",

    "Acomp_peak_lag_days",

    "Acomp_legacy_anomaly_days"
]


# Every bin stores sampled x/y values.
# We don't need every raw value for medians.
BIN_SAMPLE_MAX = 20_000


all_bin_records = []

damage_probability_records = []


for event_type, filepath in FILES.items():


    print(
        f"\nPROCESS {event_type}"
    )


    edges = bin_edges[
        event_type
    ]


    nb = (
        len(edges)
        -
        1
    )


    # -------------------------------------------------------------------------
    # Continuous responses
    # -------------------------------------------------------------------------

    store = {}


    for response_name in RESPONSES:

        store[
            response_name
        ] = [

            {
                "n": 0,

                "xsample":
                    np.empty(
                        0,
                        dtype=float
                    ),

                "ysample":
                    np.empty(
                        0,
                        dtype=float
                    )
            }

            for _
            in range(nb)
        ]


    # -------------------------------------------------------------------------
    # Probability of positive damage
    # -------------------------------------------------------------------------

    prob_total = np.zeros(
        nb,
        dtype=np.int64
    )

    prob_damage = np.zeros(
        nb,
        dtype=np.int64
    )

    prob_severity_sum = np.zeros(
        nb,
        dtype=np.float64
    )


    reader = pd.read_csv(

        filepath,

        compression="gzip",

        usecols=READ_COLS_PASS2,

        chunksize=CHUNK_SIZE,

        low_memory=False
    )


    processed = 0


    for chunk_no, df in enumerate(
        reader,
        start=1
    ):


        # Convert once
        for c in READ_COLS_PASS2:

            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )


        sev = df[
            SEVERITY
        ].to_numpy(
            dtype=float,
            copy=False
        )


        clean = (
            df[
                "clean_pre_any_event"
            ].to_numpy()
            == 1
        )


        finite_sev = np.isfinite(
            sev
        )


        # Bin index
        bindex = (
            np.searchsorted(
                edges,
                sev,
                side="right"
            )
            -
            1
        )


        # avoid final inf edge issue
        bindex = np.clip(
            bindex,
            0,
            nb - 1
        )


        # =====================================================================
        # DAMAGE PROBABILITY
        # =====================================================================

        damage_value = df[
            "Acomp_damage"
        ].to_numpy(
            dtype=float,
            copy=False
        )


        prob_good = (
            clean
            &
            finite_sev
            &
            np.isfinite(
                damage_value
            )
        )


        if prob_good.any():

            bg = bindex[
                prob_good
            ]

            sg = sev[
                prob_good
            ]

            dg = damage_value[
                prob_good
            ]


            prob_total += np.bincount(
                bg,
                minlength=nb
            )


            prob_damage += np.bincount(
                bg,
                weights=(
                    dg > 0
                ).astype(
                    np.int64
                ),
                minlength=nb
            ).astype(
                np.int64
            )


            prob_severity_sum += np.bincount(
                bg,
                weights=sg,
                minlength=nb
            )


        # =====================================================================
        # CONTINUOUS RESPONSES
        # =====================================================================

        for response_name, cfg in RESPONSES.items():


            col = cfg[
                "column"
            ]


            y = df[
                col
            ].to_numpy(
                dtype=float,
                copy=False
            )


            if response_name == "Acomp_damage":

                mask = (
                    clean
                    &
                    finite_sev
                    &
                    (
                        df[
                            "valid_resistance_sample"
                        ].to_numpy()
                        == 1
                    )
                    &
                    np.isfinite(y)
                    &
                    (
                        y > 0
                    )
                )


            else:

                mask = (
                    clean
                    &
                    finite_sev
                    &
                    np.isfinite(y)
                )


            if not mask.any():
                continue


            valid_bins = np.unique(
                bindex[
                    mask
                ]
            )


            for b in valid_bins:


                sel = (
                    mask
                    &
                    (
                        bindex == b
                    )
                )


                xs = sev[
                    sel
                ]

                ys = y[
                    sel
                ]


                st = store[
                    response_name
                ][
                    int(b)
                ]


                st["n"] += len(xs)


                # Joint subsampling keeps x-y pairing.
                if len(xs) > BIN_SAMPLE_MAX:

                    ix = RNG.choice(
                        len(xs),
                        BIN_SAMPLE_MAX,
                        replace=False
                    )

                    xs = xs[ix]
                    ys = ys[ix]


                oldx = st[
                    "xsample"
                ]

                oldy = st[
                    "ysample"
                ]


                nx = np.concatenate(
                    [
                        oldx,
                        xs
                    ]
                )

                ny = np.concatenate(
                    [
                        oldy,
                        ys
                    ]
                )


                if len(nx) > BIN_SAMPLE_MAX:

                    ix = RNG.choice(
                        len(nx),
                        BIN_SAMPLE_MAX,
                        replace=False
                    )

                    nx = nx[ix]
                    ny = ny[ix]


                st[
                    "xsample"
                ] = nx

                st[
                    "ysample"
                ] = ny


        processed += len(df)


        print(
            f"{event_type} "
            f"| chunk={chunk_no:03d} "
            f"| processed={processed:,}",
            end="\r"
        )


        del df
        gc.collect()


    print(
        f"\n{event_type}: full scan completed."
    )


    # =========================================================================
    # OUTPUT CONTINUOUS BINS
    # =========================================================================

    for response_name in RESPONSES:


        for b, st in enumerate(
            store[
                response_name
            ]
        ):


            if (
                st["n"]
                <
                MIN_BIN_N
            ):

                continue


            xs = st[
                "xsample"
            ]

            ys = st[
                "ysample"
            ]


            if (
                len(xs) == 0
                or
                len(ys) == 0
            ):

                continue


            all_bin_records.append({

                "event_type":
                    event_type,

                "response":
                    response_name,

                "bin_id":
                    b,

                "n":
                    st["n"],

                "severity_median":
                    float(
                        np.nanmedian(
                            xs
                        )
                    ),

                "severity_q25":
                    float(
                        np.nanquantile(
                            xs,
                            0.25
                        )
                    ),

                "severity_q75":
                    float(
                        np.nanquantile(
                            xs,
                            0.75
                        )
                    ),

                "response_median":
                    float(
                        np.nanmedian(
                            ys
                        )
                    ),

                "response_q25":
                    float(
                        np.nanquantile(
                            ys,
                            0.25
                        )
                    ),

                "response_q75":
                    float(
                        np.nanquantile(
                            ys,
                            0.75
                        )
                    )
            })


    # =========================================================================
    # OUTPUT DAMAGE PROBABILITY BINS
    # =========================================================================

    for b in range(
        nb
    ):


        n = prob_total[b]


        if n < MIN_BIN_N:
            continue


        probability = (
            prob_damage[b]
            /
            n
        )


        mean_severity = (
            prob_severity_sum[b]
            /
            n
        )


        damage_probability_records.append({

            "event_type":
                event_type,

            "bin_id":
                b,

            "n":
                int(n),

            "severity_mean":
                float(
                    mean_severity
                ),

            "damage_probability":
                float(
                    probability
                )
        })


# =============================================================================
# 10. SAVE BINNED DATA
# =============================================================================

header(
    "SAVE BINNED DOSE-RESPONSE DATA"
)


bins_df = pd.DataFrame(
    all_bin_records
)


prob_df = pd.DataFrame(
    damage_probability_records
)


BINS_FILE = os.path.join(
    OUT_DIR,
    "L5_4A_Severity_Response_Binned.csv"
)


PROB_FILE = os.path.join(
    OUT_DIR,
    "L5_4A_Severity_DamageProbability_Binned.csv"
)


bins_df.to_csv(
    BINS_FILE,
    index=False,
    encoding="utf-8-sig"
)


prob_df.to_csv(
    PROB_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:"
)

print(
    BINS_FILE
)

print(
    PROB_FILE
)


# =============================================================================
# 11. FIT THRESHOLD MODELS
# =============================================================================

header(
    "FIT LINEAR VS SEGMENTED MODELS"
)


threshold_records = []

prediction_records = []


for event_type in FILES:


    sev_sample = severity_samples[
        event_type
    ]


    for response_name in RESPONSES:


        d = bins_df[
            (
                bins_df[
                    "event_type"
                ]
                ==
                event_type
            )
            &
            (
                bins_df[
                    "response"
                ]
                ==
                response_name
            )
        ].copy()


        if len(d) < 20:

            print(
                f"SKIP {event_type} / {response_name}: "
                f"insufficient bins."
            )

            continue


        x = d[
            "severity_median"
        ].to_numpy(
            dtype=float
        )

        y = d[
            "response_median"
        ].to_numpy(
            dtype=float
        )

        w = d[
            "n"
        ].to_numpy(
            dtype=float
        )


        # Prevent huge bins from numerically dominating.
        # sqrt(N) retains information weighting but is more robust.
        wfit = np.sqrt(
            w
        )


        # ---------------------------------------------------------------------
        # Linear
        # ---------------------------------------------------------------------

        lin_beta, lin_pred, lin_rss = (
            weighted_linear_fit(
                x,
                y,
                wfit
            )
        )


        lin_aic, lin_bic = (
            calculate_aic_bic(
                lin_rss,
                len(x),
                2
            )
        )


        # ---------------------------------------------------------------------
        # Segmented
        # ---------------------------------------------------------------------

        (
            xs,
            ys,
            ws,
            best
        ) = find_best_breakpoint(
            x,
            y,
            wfit
        )


        c = best[
            "breakpoint"
        ]

        beta = best[
            "beta"
        ]

        seg_rss = best[
            "rss"
        ]


        seg_aic, seg_bic = (
            calculate_aic_bic(
                seg_rss,
                len(xs),
                4
            )
        )


        pre_slope = (
            beta[1]
        )


        post_slope = (
            beta[1]
            +
            beta[2]
        )


        delta_aic = (
            lin_aic
            -
            seg_aic
        )


        delta_bic = (
            lin_bic
            -
            seg_bic
        )


        # ---------------------------------------------------------------------
        # Breakpoint percentile
        # ---------------------------------------------------------------------

        break_percentile = (
            100.0
            *
            np.mean(
                sev_sample
                <=
                c
            )
        )


        # ---------------------------------------------------------------------
        # Bootstrap
        # ---------------------------------------------------------------------

        boot = bootstrap_breakpoint(
            xs,
            ys,
            ws
        )


        if len(boot) >= 30:

            ci_low = float(
                np.quantile(
                    boot,
                    0.025
                )
            )

            ci_high = float(
                np.quantile(
                    boot,
                    0.975
                )
            )

        else:

            ci_low = np.nan
            ci_high = np.nan


        # ---------------------------------------------------------------------
        # Threshold evidence
        # ---------------------------------------------------------------------

        if (
            np.isfinite(
                delta_aic
            )
            and
            delta_aic >= 10
        ):

            evidence = "strong"

        elif (
            np.isfinite(
                delta_aic
            )
            and
            delta_aic >= 4
        ):

            evidence = "moderate"

        else:

            evidence = "weak_or_none"


        threshold_records.append({

            "event_type":
                event_type,

            "response":
                response_name,

            "n_bins":
                len(xs),

            "breakpoint_severity":
                c,

            "breakpoint_percentile":
                break_percentile,

            "breakpoint_CI_low":
                ci_low,

            "breakpoint_CI_high":
                ci_high,

            "pre_break_slope":
                pre_slope,

            "post_break_slope":
                post_slope,

            "slope_change":
                post_slope
                -
                pre_slope,

            "linear_AIC":
                lin_aic,

            "segmented_AIC":
                seg_aic,

            "delta_AIC_linear_minus_segmented":
                delta_aic,

            "linear_BIC":
                lin_bic,

            "segmented_BIC":
                seg_bic,

            "delta_BIC_linear_minus_segmented":
                delta_bic,

            "threshold_evidence":
                evidence,

            "bootstrap_valid_n":
                len(boot)
        })


        # ---------------------------------------------------------------------
        # Prediction table
        # ---------------------------------------------------------------------

        hinge = np.maximum(
            0.0,
            xs - c
        )


        seg_pred = (
            beta[0]
            +
            beta[1]
            *
            xs
            +
            beta[2]
            *
            hinge
        )


        for xx, yy, ww, pp in zip(
            xs,
            ys,
            ws,
            seg_pred
        ):

            prediction_records.append({

                "event_type":
                    event_type,

                "response":
                    response_name,

                "severity":
                    xx,

                "observed_bin_median":
                    yy,

                "weight":
                    ww,

                "segmented_prediction":
                    pp,

                "breakpoint":
                    c
            })


        print(
            f"{event_type:<5} | "
            f"{response_name:<18} | "
            f"break={c:.6g} | "
            f"Pctl={break_percentile:.1f}% | "
            f"ΔAIC={delta_aic:.2f} | "
            f"{evidence}"
        )


# =============================================================================
# 12. DAMAGE PROBABILITY THRESHOLD
# =============================================================================

header(
    "FIT DAMAGE-PROBABILITY THRESHOLDS"
)


for event_type in FILES:


    d = prob_df[
        prob_df[
            "event_type"
        ]
        ==
        event_type
    ].copy()


    if len(d) < 20:
        continue


    x = d[
        "severity_mean"
    ].to_numpy(
        dtype=float
    )

    y = d[
        "damage_probability"
    ].to_numpy(
        dtype=float
    )

    w = np.sqrt(
        d[
            "n"
        ].to_numpy(
            dtype=float
        )
    )


    lin_beta, lin_pred, lin_rss = (
        weighted_linear_fit(
            x,
            y,
            w
        )
    )


    lin_aic, lin_bic = (
        calculate_aic_bic(
            lin_rss,
            len(x),
            2
        )
    )


    (
        xs,
        ys,
        ws,
        best
    ) = find_best_breakpoint(
        x,
        y,
        w
    )


    c = best[
        "breakpoint"
    ]

    beta = best[
        "beta"
    ]

    seg_rss = best[
        "rss"
    ]


    seg_aic, seg_bic = (
        calculate_aic_bic(
            seg_rss,
            len(xs),
            4
        )
    )


    delta_aic = (
        lin_aic
        -
        seg_aic
    )


    delta_bic = (
        lin_bic
        -
        seg_bic
    )


    pre_slope = beta[1]

    post_slope = (
        beta[1]
        +
        beta[2]
    )


    sev_sample = severity_samples[
        event_type
    ]


    pctl = (
        100.0
        *
        np.mean(
            sev_sample
            <=
            c
        )
    )


    boot = bootstrap_breakpoint(
        xs,
        ys,
        ws
    )


    if len(boot) >= 30:

        ci_low = float(
            np.quantile(
                boot,
                0.025
            )
        )

        ci_high = float(
            np.quantile(
                boot,
                0.975
            )
        )

    else:

        ci_low = np.nan
        ci_high = np.nan


    if delta_aic >= 10:
        evidence = "strong"

    elif delta_aic >= 4:
        evidence = "moderate"

    else:
        evidence = "weak_or_none"


    threshold_records.append({

        "event_type":
            event_type,

        "response":
            "Damage_probability",

        "n_bins":
            len(xs),

        "breakpoint_severity":
            c,

        "breakpoint_percentile":
            pctl,

        "breakpoint_CI_low":
            ci_low,

        "breakpoint_CI_high":
            ci_high,

        "pre_break_slope":
            pre_slope,

        "post_break_slope":
            post_slope,

        "slope_change":
            post_slope
            -
            pre_slope,

        "linear_AIC":
            lin_aic,

        "segmented_AIC":
            seg_aic,

        "delta_AIC_linear_minus_segmented":
            delta_aic,

        "linear_BIC":
            lin_bic,

        "segmented_BIC":
            seg_bic,

        "delta_BIC_linear_minus_segmented":
            delta_bic,

        "threshold_evidence":
            evidence,

        "bootstrap_valid_n":
            len(boot)
    })


    print(
        f"{event_type:<5} | "
        f"Damage_probability | "
        f"break={c:.6g} | "
        f"Pctl={pctl:.1f}% | "
        f"ΔAIC={delta_aic:.2f} | "
        f"{evidence}"
    )


# =============================================================================
# 13. SAVE THRESHOLD RESULTS
# =============================================================================

header(
    "SAVE L5.4A RESULTS"
)


threshold_df = pd.DataFrame(
    threshold_records
)


prediction_df = pd.DataFrame(
    prediction_records
)


THRESHOLD_FILE = os.path.join(
    OUT_DIR,
    "L5_4A_Severity_Response_Thresholds.csv"
)


PREDICTION_FILE = os.path.join(
    OUT_DIR,
    "L5_4A_Segmented_Model_Predictions.csv"
)


threshold_df.to_csv(
    THRESHOLD_FILE,
    index=False,
    encoding="utf-8-sig"
)


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False,
    encoding="utf-8-sig"
)


print(
    threshold_df.to_string(
        index=False
    )
)


print(
    "\nSaved:"
)

print(
    THRESHOLD_FILE
)

print(
    PREDICTION_FILE
)


# =============================================================================
# 14. FINAL QC / REPORT
# =============================================================================

header(
    "FINAL QC"
)


n_expected = (
    len(FILES)
    *
    4
)


n_models = len(
    threshold_df
)


MODEL_COUNT_PASS = (
    n_models
    ==
    n_expected
)


BREAKPOINT_PASS = bool(
    threshold_df[
        "breakpoint_severity"
    ].notna().all()
)


BOOTSTRAP_PASS = bool(
    (
        threshold_df[
            "bootstrap_valid_n"
        ]
        >= 100
    ).all()
)


print(
    "Expected models    :",
    n_expected
)

print(
    "Completed models   :",
    n_models
)

print(
    "Model count PASS   :",
    MODEL_COUNT_PASS
)

print(
    "Breakpoint PASS    :",
    BREAKPOINT_PASS
)

print(
    "Bootstrap PASS     :",
    BOOTSTRAP_PASS
)


OVERALL_PASS = (
    MODEL_COUNT_PASS
    and
    BREAKPOINT_PASS
    and
    BOOTSTRAP_PASS
)


REPORT_FILE = os.path.join(
    OUT_DIR,
    "L5_4A_Severity_Response_Threshold_Report.txt"
)


lines = []

lines.append(
    "Step L5.4A Severity–Response Threshold Analysis"
)

lines.append(
    "=" * 90
)

lines.append("")

lines.append(
    "Method:"
)

lines.append(
    "  Full L5.3 event tables were scanned in chunks."
)

lines.append(
    "  Only events with uncontaminated pre-event windows were used."
)

lines.append(
    "  Acomp damage threshold models additionally require Acomp_damage > 0."
)

lines.append(
    "  Full event distributions were represented by severity-quantile bins."
)

lines.append(
    "  Weighted continuous one-breakpoint segmented regression was compared with a linear model."
)

lines.append(
    "  Breakpoint uncertainty was estimated using bootstrap resampling of bin-level observations."
)

lines.append("")

lines.append(
    "Interpretation:"
)

lines.append(
    "  Positive delta_AIC means segmented regression outperforms the linear model."
)

lines.append(
    "  delta_AIC >= 10 = strong threshold evidence."
)

lines.append(
    "  delta_AIC 4–10 = moderate threshold evidence."
)

lines.append(
    "  delta_AIC < 4 = weak/no evidence."
)

lines.append("")

lines.append(
    "Because event severity units differ among CDH, HEAT and HR, raw breakpoint values must not be directly compared across event types."
)

lines.append(
    "Breakpoint percentile is the preferred cross-event comparison metric."
)

lines.append("")

lines.append(
    f"Model count PASS : {MODEL_COUNT_PASS}"
)

lines.append(
    f"Breakpoint PASS  : {BREAKPOINT_PASS}"
)

lines.append(
    f"Bootstrap PASS   : {BOOTSTRAP_PASS}"
)

lines.append(
    f"OVERALL PASS     : {OVERALL_PASS}"
)

lines.append("")

lines.append(
    f"Threshold table : {THRESHOLD_FILE}"
)

lines.append(
    f"Binned responses: {BINS_FILE}"
)

lines.append(
    f"Probability bins: {PROB_FILE}"
)

lines.append(
    f"Predictions     : {PREDICTION_FILE}"
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            lines
        )
    )


header(
    "FINAL RESULT"
)


print(
    "\n".join(
        lines
    )
)