# -*- coding: utf-8 -*-
"""
Figure 8 FINAL V1
Hydroclimatic regulation of greening-associated ecosystem resilience
Weihe River Basin, 2001–2024

Core idea
---------
This figure DOES NOT claim a causal ecological-restoration effect.

Long-term growing-season NIRv trend is used as a remote-sensing
proxy of greening / restoration intensity.

The analysis asks:

    At comparable compound drought-heat (CDH) severity,
    does stronger long-term greening correspond to lower
    ecological damage / legacy?

and:

    Is this greening-resilience association dependent on
    long-term root-zone soil-water availability?

Scientific workflow
-------------------
1. Calculate pixel-level growing-season NIRv trend, 2001–2024
2. Calculate pixel-level growing-season RootSM climatology
3. Read full CDH event database in chunks
4. Remove the basin-wide severity effect using the frozen
   L5.4A severity-response curves
5. Analyse residual damage / residual legacy across:
       greening gradient × RootSM gradient
6. Diagnose possible greening-benefit reversal

Panels
------
a  NIRv greening trend map
b  Root-zone soil-moisture climatology
c  Greening gradient -> severity-adjusted CDH damage
   under dry/intermediate/wet backgrounds
d  Greening gradient -> severity-adjusted ecological legacy
e  2-D hydroclimatic resilience surface
f  Greening resilience benefit along RootSM gradient

IMPORTANT
---------
Positive residual damage:
    greater ecological damage than expected for the same CDH severity

Negative residual damage:
    less damage than expected

Greening resilience benefit:
    mean residual damage of LOW-greening pixels
    minus
    mean residual damage of HIGH-greening pixels

Therefore:
    Benefit > 0 : high greening associated with lower damage
    Benefit < 0 : high greening associated with higher damage

Association != causality.

Author:
Weihe ecological resilience project
"""

# ======================================================================
# 0. IMPORTS
# ======================================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


# optional GIS support
try:
    import geopandas as gpd
    HAS_GPD = True
except Exception:
    HAS_GPD = False


# ======================================================================
# 1. PATHS
# ======================================================================

L4_RAW_NC = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
    r"\Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc"
)

L3_NC = (
    r".\Weihe database V3"
    r"\Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
)

CDH_CSV = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
    r"\L5_3_SeverityResilience"
    r"\L5_3_CDH_Event_Severity_Resilience_Metrics.csv.gz"
)

L54_BINNED = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
    r"\L5_4_ThresholdAnalysis"
    r"\L5_4A_Severity_Response_Binned.csv"
)

# Basin boundary folder
SHP_DIR = r".\weihe\Weihe"


# ======================================================================
# OUTPUT
# ======================================================================

OUT_DIR = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
    r"\Figures_FINAL"
    r"\Figure08_Greening_Hydroclimate_Resilience"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

OUT_PNG = os.path.join(
    OUT_DIR,
    "Figure08_Greening_Hydroclimate_Resilience_FINAL.png"
)

OUT_TIF = os.path.join(
    OUT_DIR,
    "Figure08_Greening_Hydroclimate_Resilience_FINAL.tif"
)

OUT_PDF = os.path.join(
    OUT_DIR,
    "Figure08_Greening_Hydroclimate_Resilience_FINAL.pdf"
)

OUT_PROXY_NC = os.path.join(
    OUT_DIR,
    "Figure08_Greening_Hydroclimate_Proxy_Maps.nc"
)

OUT_GRADIENT = os.path.join(
    OUT_DIR,
    "Figure08_GreeningGradient_SourceData.csv"
)

OUT_SURFACE = os.path.join(
    OUT_DIR,
    "Figure08_ResponseSurface_SourceData.csv"
)

OUT_BENEFIT = os.path.join(
    OUT_DIR,
    "Figure08_GreeningBenefit_SourceData.csv"
)


# ======================================================================
# 2. SETTINGS
# ======================================================================

START_YEAR = 2001
END_YEAR = 2024

# Growing season
GS_MONTH_START = 4
GS_MONTH_END = 10

# At least 18 annual growing-season values required
MIN_VALID_YEARS = 18

# CSV streaming
CHUNK_SIZE = 500_000

# Spatial response gradient
N_GREEN_BINS = 10
N_SM_SURFACE_BINS = 10

# low/high greening groups for panel f
LOW_GREEN_PERCENTILE = 20
HIGH_GREEN_PERCENTILE = 80

# Require enough events per aggregate cell
MIN_EVENTS_BIN = 1000


# ======================================================================
# 3. FIGURE STYLE
# ======================================================================

mpl.rcParams.update({

    "font.family": "serif",

    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif"
    ],

    "font.size": 9,

    "axes.titlesize": 10,
    "axes.labelsize": 9.5,

    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,

    "legend.fontsize": 7.8,

    "axes.linewidth": 0.8,

    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,

    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "savefig.dpi": 600
})


# ======================================================================
# 4. COLOR SETTINGS
# ======================================================================

MOISTURE_COLORS = {

    0: "#C58C5C",   # dry

    1: "#6F9E78",   # intermediate

    2: "#4472A8"    # wet
}


MOISTURE_LABELS = {

    0: "Dry",

    1: "Intermediate",

    2: "Wet"
}


# ======================================================================
# 5. UTILITIES
# ======================================================================

def sep(title):

    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def require_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )


def panel_label(ax, text):

    ax.text(
        -0.12,
        1.055,
        text,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top"
    )


def clean_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        direction="out",
        width=0.7,
        length=3.2
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.45,
        alpha=0.22,
        zorder=0
    )


def bool_series(series):

    """
    Convert mixed bool / integer / string to bool safely.
    """

    if series.dtype == bool:
        return series.to_numpy()

    if np.issubdtype(
        series.dtype,
        np.number
    ):

        return (
            pd.to_numeric(
                series,
                errors="coerce"
            ).fillna(0).to_numpy()
            >
            0
        )

    txt = (
        series
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return txt.isin(
        [
            "true",
            "1",
            "yes",
            "y"
        ]
    ).to_numpy()


def safe_mean_from_sum(
    sum_arr,
    count_arr
):

    result = np.full(
        np.shape(sum_arr),
        np.nan,
        dtype=float
    )

    valid = (
        count_arr
        >
        0
    )

    result[
        valid
    ] = (
        sum_arr[
            valid
        ]
        /
        count_arr[
            valid
        ]
    )

    return result


def safe_se(
    sum_arr,
    sumsq_arr,
    count_arr
):

    mean = safe_mean_from_sum(
        sum_arr,
        count_arr
    )

    var = np.full(
        np.shape(sum_arr),
        np.nan,
        dtype=float
    )

    valid = (
        count_arr
        >
        1
    )

    var[
        valid
    ] = (

        sumsq_arr[
            valid
        ]
        /
        count_arr[
            valid
        ]

        -

        mean[
            valid
        ] ** 2
    )

    var = np.maximum(
        var,
        0
    )

    se = np.full_like(
        mean,
        np.nan
    )

    se[
        valid
    ] = (
        np.sqrt(
            var[
                valid
            ]
        )
        /
        np.sqrt(
            count_arr[
                valid
            ]
        )
    )

    return se


def digitize_clipped(
    values,
    edges
):

    idx = np.digitize(
        values,
        edges[
            1:-1
        ],
        right=False
    )

    idx = np.clip(
        idx,
        0,
        len(edges) - 2
    )

    return idx


def get_quantile_edges(
    values,
    n_bins
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    probs = np.linspace(
        0,
        100,
        n_bins + 1
    )

    edges = np.nanpercentile(
        values,
        probs
    )

    # protect against repeated quantiles
    for i in range(
        1,
        len(edges)
    ):

        if edges[i] <= edges[i - 1]:

            edges[i] = (
                edges[i - 1]
                +
                1e-12
            )

    return edges


# ======================================================================
# 6. FIND SHAPEFILE
# ======================================================================

def find_shapefile(folder):

    if not os.path.isdir(folder):
        return None

    files = glob.glob(
        os.path.join(
            folder,
            "**",
            "*.shp"
        ),
        recursive=True
    )

    if len(files) == 0:
        return None

    return files[0]


# ======================================================================
# 7. PIXEL-WISE ANNUAL TREND
# ======================================================================

def calculate_nirv_trend(
    ds
):

    """
    Growing-season annual mean NIRv:
        April–October

    Pixel-wise OLS trend:
        NIRv ~ year

    Output:
        slope per decade
    """

    sep(
        "CALCULATE 2001–2024 GROWING-SEASON NIRv TREND"
    )

    time = pd.DatetimeIndex(
        pd.to_datetime(
            ds["time"].values
        )
    )

    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    # OLS accumulators
    n = np.zeros(
        (ny, nx),
        dtype=np.float64
    )

    sx = np.zeros_like(n)
    sy = np.zeros_like(n)

    sxx = np.zeros_like(n)
    sxy = np.zeros_like(n)

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        mask = (
            (time.year == year)
            &
            (time.month >= GS_MONTH_START)
            &
            (time.month <= GS_MONTH_END)
        )

        indices = np.where(
            mask
        )[0]

        if len(indices) == 0:

            print(
                f"{year}: NO DATA"
            )

            continue


        arr = (
            ds["NIRv"]
            .isel(
                time=indices
            )
            .mean(
                dim="time",
                skipna=True
            )
            .values
            .astype(
                np.float64
            )
        )


        valid = np.isfinite(
            arr
        )


        t = float(
            year - START_YEAR
        )


        n[
            valid
        ] += 1.0

        sx[
            valid
        ] += t

        sy[
            valid
        ] += arr[
            valid
        ]

        sxx[
            valid
        ] += t * t

        sxy[
            valid
        ] += t * arr[
            valid
        ]


        print(
            f"{year}: "
            f"steps={len(indices):2d} | "
            f"valid pixels={valid.sum():,}"
        )


    denominator = (
        n * sxx
        -
        sx * sx
    )


    slope = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32
    )


    valid = (
        (n >= MIN_VALID_YEARS)
        &
        np.isfinite(
            denominator
        )
        &
        (
            np.abs(
                denominator
            )
            >
            1e-12
        )
    )


    slope[
        valid
    ] = (

        (
            n[
                valid
            ]
            *
            sxy[
                valid
            ]

            -

            sx[
                valid
            ]
            *
            sy[
                valid
            ]
        )

        /

        denominator[
            valid
        ]

        *

        10.0

    ).astype(
        np.float32
    )


    print(
        "\nNIRv trend valid pixels:",
        np.isfinite(
            slope
        ).sum()
    )

    print(
        "NIRv slope/decade percentiles:",
        np.nanpercentile(
            slope,
            [
                1,
                5,
                25,
                50,
                75,
                95,
                99
            ]
        )
    )

    return slope


# ======================================================================
# 8. ROOTSM CLIMATOLOGY
# ======================================================================

def calculate_rootsm_climatology(
    ds
):

    sep(
        "CALCULATE GROWING-SEASON ROOTSM CLIMATOLOGY"
    )

    time = pd.DatetimeIndex(
        pd.to_datetime(
            ds["time"].values
        )
    )

    ny = ds.sizes["y"]
    nx = ds.sizes["x"]

    value_sum = np.zeros(
        (ny, nx),
        dtype=np.float64
    )

    value_count = np.zeros(
        (ny, nx),
        dtype=np.int16
    )


    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        mask = (
            (time.year == year)
            &
            (time.month >= GS_MONTH_START)
            &
            (time.month <= GS_MONTH_END)
        )

        indices = np.where(
            mask
        )[0]


        if len(indices) == 0:

            continue


        annual = (
            ds["RootSM"]
            .isel(
                time=indices
            )
            .mean(
                dim="time",
                skipna=True
            )
            .values
            .astype(
                np.float64
            )
        )


        valid = np.isfinite(
            annual
        )


        value_sum[
            valid
        ] += annual[
            valid
        ]

        value_count[
            valid
        ] += 1


        print(
            f"{year}: "
            f"valid pixels={valid.sum():,}"
        )


    climatology = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32
    )


    valid = (
        value_count
        >=
        MIN_VALID_YEARS
    )


    climatology[
        valid
    ] = (

        value_sum[
            valid
        ]
        /
        value_count[
            valid
        ]

    ).astype(
        np.float32
    )


    print(
        "\nRootSM climatology percentiles:",
        np.nanpercentile(
            climatology,
            [
                1,
                5,
                25,
                50,
                75,
                95,
                99
            ]
        )
    )


    return climatology


# ======================================================================
# 9. READ INPUT DATA
# ======================================================================

sep(
    "CHECK INPUT FILES"
)

for file_path in [

    L4_RAW_NC,
    L3_NC,
    CDH_CSV,
    L54_BINNED

]:

    require_file(
        file_path
    )

    print(
        "FOUND:",
        file_path
    )


sep(
    "OPEN NETCDF DATASETS"
)

eco = xr.open_dataset(
    L4_RAW_NC
)

clim = xr.open_dataset(
    L3_NC
)


print(
    "L4:",
    dict(
        eco.sizes
    )
)

print(
    "L3:",
    dict(
        clim.sizes
    )
)


# ======================================================================
# 10. ALIGNMENT QC
# ======================================================================

if (
    eco.sizes["y"]
    !=
    clim.sizes["y"]
    or
    eco.sizes["x"]
    !=
    clim.sizes["x"]
):

    raise RuntimeError(
        "L3 and L4 spatial dimensions are not aligned."
    )


if not np.array_equal(
    eco["x"].values,
    clim["x"].values
):

    raise RuntimeError(
        "L3 and L4 x coordinates do not match."
    )


if not np.array_equal(
    eco["y"].values,
    clim["y"].values
):

    raise RuntimeError(
        "L3 and L4 y coordinates do not match."
    )


x_coords = eco[
    "x"
].values

y_coords = eco[
    "y"
].values


NY = len(
    y_coords
)

NX = len(
    x_coords
)


print(
    "Grid alignment PASS."
)


# ======================================================================
# 11. BUILD GREENING + WATER PROXY MAPS
# ======================================================================

nirv_slope = calculate_nirv_trend(
    eco
)

rootsm_clim = calculate_rootsm_climatology(
    clim
)


# ======================================================================
# 12. VALID PIXEL MASK
# ======================================================================

proxy_valid = (
    np.isfinite(
        nirv_slope
    )
    &
    np.isfinite(
        rootsm_clim
    )
)


print(
    "\nValid joint proxy pixels:",
    proxy_valid.sum()
)


# ======================================================================
# 13. GREENING / ROOTSM QUANTILES
# ======================================================================

GREEN_EDGES = get_quantile_edges(

    nirv_slope[
        proxy_valid
    ],

    N_GREEN_BINS
)


SM_DECILE_EDGES = get_quantile_edges(

    rootsm_clim[
        proxy_valid
    ],

    N_SM_SURFACE_BINS
)


SM_TERCILE_EDGES = get_quantile_edges(

    rootsm_clim[
        proxy_valid
    ],

    3
)


green_q20 = np.nanpercentile(
    nirv_slope[
        proxy_valid
    ],
    LOW_GREEN_PERCENTILE
)


green_q80 = np.nanpercentile(
    nirv_slope[
        proxy_valid
    ],
    HIGH_GREEN_PERCENTILE
)


sep(
    "PROXY GRADIENT THRESHOLDS"
)

print(
    "Greening Q20:",
    green_q20
)

print(
    "Greening Q80:",
    green_q80
)

print(
    "RootSM terciles:",
    SM_TERCILE_EDGES
)


# ======================================================================
# 14. LOAD FROZEN L5.4A CURVES
# ======================================================================

sep(
    "LOAD FROZEN SEVERITY-RESPONSE CURVES"
)

binned = pd.read_csv(
    L54_BINNED
)


print(
    binned[
        [
            "event_type",
            "response"
        ]
    ]
    .drop_duplicates()
    .to_string(
        index=False
    )
)


def get_expected_curve(
    response_name
):

    tmp = binned[
        (
            binned[
                "event_type"
            ] == "CDH"
        )
        &
        (
            binned[
                "response"
            ] == response_name
        )
    ].copy()


    if len(tmp) == 0:

        raise RuntimeError(
            f"No CDH response curve found for {response_name}"
        )


    tmp = (
        tmp[
            [
                "severity_median",
                "response_median"
            ]
        ]
        .dropna()
        .groupby(
            "severity_median",
            as_index=False
        )[
            "response_median"
        ]
        .median()
        .sort_values(
            "severity_median"
        )
    )


    return (

        tmp[
            "severity_median"
        ].to_numpy(
            dtype=float
        ),

        tmp[
            "response_median"
        ].to_numpy(
            dtype=float
        )
    )


damage_x, damage_y = get_expected_curve(
    "Acomp_damage"
)


legacy_x, legacy_y = get_expected_curve(
    "Acomp_legacy"
)


print(
    "Damage curve points:",
    len(
        damage_x
    )
)

print(
    "Legacy curve points:",
    len(
        legacy_x
    )
)


# ======================================================================
# 15. PREPARE EVENT AGGREGATORS
# ======================================================================

# moisture tercile × greening decile

damage_sum = np.zeros(
    (
        3,
        N_GREEN_BINS
    ),
    dtype=np.float64
)

damage_sumsq = np.zeros_like(
    damage_sum
)

damage_count = np.zeros(
    (
        3,
        N_GREEN_BINS
    ),
    dtype=np.int64
)


legacy_sum = np.zeros_like(
    damage_sum
)

legacy_sumsq = np.zeros_like(
    damage_sum
)

legacy_count = np.zeros(
    (
        3,
        N_GREEN_BINS
    ),
    dtype=np.int64
)


# 2D RootSM decile × Greening decile

surface_sum = np.zeros(
    (
        N_SM_SURFACE_BINS,
        N_GREEN_BINS
    ),
    dtype=np.float64
)

surface_count = np.zeros(
    (
        N_SM_SURFACE_BINS,
        N_GREEN_BINS
    ),
    dtype=np.int64
)


# benefit:
# RootSM decile × {low-green, high-green}

benefit_sum = np.zeros(
    (
        N_SM_SURFACE_BINS,
        2
    ),
    dtype=np.float64
)

benefit_count = np.zeros(
    (
        N_SM_SURFACE_BINS,
        2
    ),
    dtype=np.int64
)


# ======================================================================
# 16. STREAM CDH EVENT DATABASE
# ======================================================================

sep(
    "STREAM CDH EVENT DATABASE"
)


USECOLS = [

    "row",
    "col",

    "climate_cumulative_severity",

    "Acomp_damage",
    "Acomp_legacy_anomaly_days",

    "valid_resistance_sample",
    "valid_recovery_analysis"
]


total_rows = 0

resistance_rows = 0
legacy_rows = 0


for chunk_id, df in enumerate(

    pd.read_csv(
        CDH_CSV,
        usecols=USECOLS,
        chunksize=CHUNK_SIZE,
        compression="gzip"
    ),

    start=1
):

    total_rows += len(
        df
    )


    # --------------------------------------------------------------
    # spatial index validation
    # --------------------------------------------------------------

    row_numeric = pd.to_numeric(
        df[
            "row"
        ],
        errors="coerce"
    )

    col_numeric = pd.to_numeric(
        df[
            "col"
        ],
        errors="coerce"
    )


    spatial_ok = (

        row_numeric.notna()
        &
        col_numeric.notna()

        &
        (
            row_numeric
            >=
            0
        )

        &
        (
            row_numeric
            <
            NY
        )

        &
        (
            col_numeric
            >=
            0
        )

        &
        (
            col_numeric
            <
            NX
        )
    )


    if spatial_ok.sum() == 0:

        continue


    work = df.loc[
        spatial_ok
    ].copy()


    rr = row_numeric.loc[
        spatial_ok
    ].astype(
        np.int32
    ).to_numpy()


    cc = col_numeric.loc[
        spatial_ok
    ].astype(
        np.int32
    ).to_numpy()


    greening = nirv_slope[
        rr,
        cc
    ]


    rootsm = rootsm_clim[
        rr,
        cc
    ]


    severity = pd.to_numeric(
        work[
            "climate_cumulative_severity"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    damage = pd.to_numeric(
        work[
            "Acomp_damage"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    legacy = pd.to_numeric(
        work[
            "Acomp_legacy_anomaly_days"
        ],
        errors="coerce"
    ).to_numpy(
        dtype=float
    )


    valid_resistance = bool_series(
        work[
            "valid_resistance_sample"
        ]
    )


    valid_recovery = bool_series(
        work[
            "valid_recovery_analysis"
        ]
    )


    proxy_ok = (
        np.isfinite(
            greening
        )
        &
        np.isfinite(
            rootsm
        )
        &
        np.isfinite(
            severity
        )
    )


    # ==============================================================
    # DAMAGE RESIDUAL
    # ==============================================================

    dmask = (

        proxy_ok
        &
        valid_resistance

        &
        np.isfinite(
            damage
        )

        &
        (
            damage
            >
            0
        )
    )


    if dmask.sum() > 0:

        dg = greening[
            dmask
        ]

        ds = rootsm[
            dmask
        ]

        sev = severity[
            dmask
        ]

        obs = damage[
            dmask
        ]


        expected = np.interp(

            sev,

            damage_x,
            damage_y,

            left=damage_y[0],

            right=damage_y[-1]
        )


        residual = (
            obs
            -
            expected
        )


        gbin = digitize_clipped(
            dg,
            GREEN_EDGES
        )


        sm_tercile = digitize_clipped(
            ds,
            SM_TERCILE_EDGES
        )


        sm_decile = digitize_clipped(
            ds,
            SM_DECILE_EDGES
        )


        # ----------------------------------------------------------
        # panel c aggregation
        # ----------------------------------------------------------

        np.add.at(

            damage_sum,

            (
                sm_tercile,
                gbin
            ),

            residual
        )


        np.add.at(

            damage_sumsq,

            (
                sm_tercile,
                gbin
            ),

            residual ** 2
        )


        np.add.at(

            damage_count,

            (
                sm_tercile,
                gbin
            ),

            1
        )


        # ----------------------------------------------------------
        # panel e response surface
        # ----------------------------------------------------------

        np.add.at(

            surface_sum,

            (
                sm_decile,
                gbin
            ),

            residual
        )


        np.add.at(

            surface_count,

            (
                sm_decile,
                gbin
            ),

            1
        )


        # ----------------------------------------------------------
        # panel f: low vs high greening
        # ----------------------------------------------------------

        low_mask = (
            dg
            <=
            green_q20
        )

        high_mask = (
            dg
            >=
            green_q80
        )


        if low_mask.sum() > 0:

            np.add.at(

                benefit_sum,

                (
                    sm_decile[
                        low_mask
                    ],
                    np.zeros(
                        low_mask.sum(),
                        dtype=np.int8
                    )
                ),

                residual[
                    low_mask
                ]
            )


            np.add.at(

                benefit_count,

                (
                    sm_decile[
                        low_mask
                    ],
                    np.zeros(
                        low_mask.sum(),
                        dtype=np.int8
                    )
                ),

                1
            )


        if high_mask.sum() > 0:

            np.add.at(

                benefit_sum,

                (
                    sm_decile[
                        high_mask
                    ],
                    np.ones(
                        high_mask.sum(),
                        dtype=np.int8
                    )
                ),

                residual[
                    high_mask
                ]
            )


            np.add.at(

                benefit_count,

                (
                    sm_decile[
                        high_mask
                    ],
                    np.ones(
                        high_mask.sum(),
                        dtype=np.int8
                    )
                ),

                1
            )


        resistance_rows += int(
            dmask.sum()
        )


    # ==============================================================
    # LEGACY RESIDUAL
    # ==============================================================

    lmask = (

        proxy_ok
        &
        valid_recovery

        &
        np.isfinite(
            legacy
        )

        &
        (
            legacy
            >=
            0
        )
    )


    if lmask.sum() > 0:

        lg = greening[
            lmask
        ]

        ls = rootsm[
            lmask
        ]

        sev = severity[
            lmask
        ]

        obs = legacy[
            lmask
        ]


        expected = np.interp(

            sev,

            legacy_x,
            legacy_y,

            left=legacy_y[0],

            right=legacy_y[-1]
        )


        residual = (
            obs
            -
            expected
        )


        gbin = digitize_clipped(
            lg,
            GREEN_EDGES
        )


        sm_tercile = digitize_clipped(
            ls,
            SM_TERCILE_EDGES
        )


        np.add.at(

            legacy_sum,

            (
                sm_tercile,
                gbin
            ),

            residual
        )


        np.add.at(

            legacy_sumsq,

            (
                sm_tercile,
                gbin
            ),

            residual ** 2
        )


        np.add.at(

            legacy_count,

            (
                sm_tercile,
                gbin
            ),

            1
        )


        legacy_rows += int(
            lmask.sum()
        )


    if (
        chunk_id == 1
        or
        chunk_id % 5 == 0
    ):

        print(

            f"chunk={chunk_id:03d} | "
            f"rows={total_rows:,} | "
            f"damage={resistance_rows:,} | "
            f"legacy={legacy_rows:,}"
        )


# ======================================================================
# 17. CALCULATE AGGREGATED METRICS
# ======================================================================

sep(
    "CALCULATE AGGREGATED RESPONSE METRICS"
)


damage_mean = safe_mean_from_sum(
    damage_sum,
    damage_count
)


damage_se = safe_se(
    damage_sum,
    damage_sumsq,
    damage_count
)


legacy_mean = safe_mean_from_sum(
    legacy_sum,
    legacy_count
)


legacy_se = safe_se(
    legacy_sum,
    legacy_sumsq,
    legacy_count
)


surface_mean = safe_mean_from_sum(
    surface_sum,
    surface_count
)


surface_mean[
    surface_count
    <
    MIN_EVENTS_BIN
] = np.nan


benefit_group_mean = safe_mean_from_sum(
    benefit_sum,
    benefit_count
)


# Benefit = low greening damage - high greening damage

benefit = (

    benefit_group_mean[
        :,
        0
    ]

    -

    benefit_group_mean[
        :,
        1
    ]
)


# ======================================================================
# 18. SOURCE DATA TABLES
# ======================================================================

green_rows = []


for moisture_id in range(
    3
):

    for gbin in range(
        N_GREEN_BINS
    ):

        green_rows.append({

            "moisture_regime":
                MOISTURE_LABELS[
                    moisture_id
                ],

            "greening_decile":
                gbin + 1,

            "damage_residual_mean":
                damage_mean[
                    moisture_id,
                    gbin
                ],

            "damage_residual_SE":
                damage_se[
                    moisture_id,
                    gbin
                ],

            "damage_n":
                damage_count[
                    moisture_id,
                    gbin
                ],

            "legacy_residual_mean":
                legacy_mean[
                    moisture_id,
                    gbin
                ],

            "legacy_residual_SE":
                legacy_se[
                    moisture_id,
                    gbin
                ],

            "legacy_n":
                legacy_count[
                    moisture_id,
                    gbin
                ]
        })


gradient_df = pd.DataFrame(
    green_rows
)


gradient_df.to_csv(
    OUT_GRADIENT,
    index=False,
    encoding="utf-8-sig"
)


surface_rows = []


for sm_bin in range(
    N_SM_SURFACE_BINS
):

    for gbin in range(
        N_GREEN_BINS
    ):

        surface_rows.append({

            "RootSM_decile":
                sm_bin + 1,

            "greening_decile":
                gbin + 1,

            "damage_residual":
                surface_mean[
                    sm_bin,
                    gbin
                ],

            "n":
                surface_count[
                    sm_bin,
                    gbin
                ]
        })


pd.DataFrame(
    surface_rows
).to_csv(
    OUT_SURFACE,
    index=False,
    encoding="utf-8-sig"
)


benefit_df = pd.DataFrame({

    "RootSM_decile":
        np.arange(
            1,
            N_SM_SURFACE_BINS + 1
        ),

    "low_greening_damage":
        benefit_group_mean[
            :,
            0
        ],

    "high_greening_damage":
        benefit_group_mean[
            :,
            1
        ],

    "greening_resilience_benefit":
        benefit,

    "low_greening_n":
        benefit_count[
            :,
            0
        ],

    "high_greening_n":
        benefit_count[
            :,
            1
        ]
})


benefit_df.to_csv(
    OUT_BENEFIT,
    index=False,
    encoding="utf-8-sig"
)


# ======================================================================
# 19. SAVE PROXY MAPS
# ======================================================================

proxy_ds = xr.Dataset(

    {

        "NIRv_trend_per_decade":
            (
                (
                    "y",
                    "x"
                ),
                nirv_slope
            ),

        "RootSM_growingseason_climatology":
            (
                (
                    "y",
                    "x"
                ),
                rootsm_clim
            )
    },

    coords={

        "x":
            x_coords,

        "y":
            y_coords
    }
)


proxy_ds[
    "NIRv_trend_per_decade"
].attrs[

    "description"

] = (
    "OLS trend in annual Apr-Oct mean NIRv, "
    "2001-2024, expressed per decade"
)


proxy_ds[
    "RootSM_growingseason_climatology"
].attrs[

    "description"

] = (
    "Mean annual Apr-Oct RootSM climatology, "
    "2001-2024"
)


proxy_ds.attrs[
    "interpretation_warning"
] = (
    "NIRv trend is a greening proxy and must not "
    "be interpreted as a causal ecological-restoration treatment."
)


proxy_ds.to_netcdf(
    OUT_PROXY_NC
)


# ======================================================================
# 20. LOAD BASIN BOUNDARY
# ======================================================================

basin = None


if HAS_GPD:

    shp_file = find_shapefile(
        SHP_DIR
    )


    if shp_file is not None:

        try:

            basin = gpd.read_file(
                shp_file
            )


            if basin.crs is not None:

                basin = basin.to_crs(
                    "EPSG:4326"
                )


            print(
                "\nBoundary:",
                shp_file
            )


        except Exception as exc:

            print(
                "\nWARNING: failed to read basin boundary:",
                repr(
                    exc
                )
            )

            basin = None


# ======================================================================
# 21. CREATE FIGURE
# ======================================================================

sep(
    "DRAW FIGURE 8"
)


fig = plt.figure(
    figsize=(
        13.4,
        9.3
    )
)


gs = fig.add_gridspec(

    2,
    3,

    left=0.055,
    right=0.985,

    bottom=0.075,
    top=0.93,

    wspace=0.28,
    hspace=0.33
)


ax_a = fig.add_subplot(
    gs[
        0,
        0
    ]
)

ax_b = fig.add_subplot(
    gs[
        0,
        1
    ]
)

ax_c = fig.add_subplot(
    gs[
        0,
        2
    ]
)

ax_d = fig.add_subplot(
    gs[
        1,
        0
    ]
)

ax_e = fig.add_subplot(
    gs[
        1,
        1
    ]
)

ax_f = fig.add_subplot(
    gs[
        1,
        2
    ]
)


# ======================================================================
# PANEL A
# NIRv trend map
# ======================================================================

trend_abs = np.nanpercentile(
    np.abs(
        nirv_slope[
            np.isfinite(
                nirv_slope
            )
        ]
    ),
    98
)


mesh_a = ax_a.pcolormesh(

    x_coords,
    y_coords,

    nirv_slope,

    shading="auto",

    cmap="BrBG",

    vmin=-trend_abs,
    vmax=trend_abs,

    rasterized=True
)


if basin is not None:

    basin.boundary.plot(

        ax=ax_a,

        color="black",

        linewidth=0.6
    )


cb_a = fig.colorbar(

    mesh_a,

    ax=ax_a,

    orientation="horizontal",

    pad=0.06,

    fraction=0.045
)


cb_a.set_label(
    "NIRv trend (decade$^{-1}$)"
)


ax_a.set_title(
    "Long-term vegetation greening"
)


ax_a.set_xlabel(
    "Longitude"
)

ax_a.set_ylabel(
    "Latitude"
)


panel_label(
    ax_a,
    "a"
)


# ======================================================================
# PANEL B
# RootSM climatology
# ======================================================================

root_vmin = np.nanpercentile(
    rootsm_clim,
    2
)

root_vmax = np.nanpercentile(
    rootsm_clim,
    98
)


mesh_b = ax_b.pcolormesh(

    x_coords,
    y_coords,

    rootsm_clim,

    shading="auto",

    cmap="YlGnBu",

    vmin=root_vmin,
    vmax=root_vmax,

    rasterized=True
)


if basin is not None:

    basin.boundary.plot(

        ax=ax_b,

        color="black",

        linewidth=0.6
    )


cb_b = fig.colorbar(

    mesh_b,

    ax=ax_b,

    orientation="horizontal",

    pad=0.06,

    fraction=0.045
)


cb_b.set_label(
    "Growing-season RootSM"
)


ax_b.set_title(
    "Long-term root-zone water availability"
)


ax_b.set_xlabel(
    "Longitude"
)

ax_b.set_ylabel(
    "Latitude"
)


panel_label(
    ax_b,
    "b"
)


# ======================================================================
# PANEL C
# Greening -> residual damage by RootSM regime
# ======================================================================

x_decile = np.arange(
    1,
    N_GREEN_BINS + 1
)


for moisture_id in range(
    3
):

    y = damage_mean[
        moisture_id
    ]

    se = damage_se[
        moisture_id
    ]


    ax_c.plot(

        x_decile,
        y,

        marker="o",

        markersize=4,

        linewidth=1.8,

        color=MOISTURE_COLORS[
            moisture_id
        ],

        label=MOISTURE_LABELS[
            moisture_id
        ]
    )


    ax_c.fill_between(

        x_decile,

        y - 1.96 * se,

        y + 1.96 * se,

        color=MOISTURE_COLORS[
            moisture_id
        ],

        alpha=0.12,

        linewidth=0
    )


ax_c.axhline(

    0,

    color="0.45",

    linestyle="--",

    linewidth=0.8
)


ax_c.set_xticks(
    x_decile
)


ax_c.set_xlabel(
    "Long-term greening intensity decile"
)


ax_c.set_ylabel(
    "Severity-adjusted ecological damage"
)


ax_c.set_title(
    "Greening–resistance association"
)


ax_c.legend(
    frameon=False,
    title="RootSM background"
)


clean_axis(
    ax_c
)


panel_label(
    ax_c,
    "c"
)


# ======================================================================
# PANEL D
# Greening -> residual legacy
# ======================================================================

for moisture_id in range(
    3
):

    y = legacy_mean[
        moisture_id
    ]

    se = legacy_se[
        moisture_id
    ]


    ax_d.plot(

        x_decile,
        y,

        marker="o",

        markersize=4,

        linewidth=1.8,

        color=MOISTURE_COLORS[
            moisture_id
        ],

        label=MOISTURE_LABELS[
            moisture_id
        ]
    )


    ax_d.fill_between(

        x_decile,

        y - 1.96 * se,

        y + 1.96 * se,

        color=MOISTURE_COLORS[
            moisture_id
        ],

        alpha=0.12,

        linewidth=0
    )


ax_d.axhline(

    0,

    color="0.45",

    linestyle="--",

    linewidth=0.8
)


ax_d.set_xticks(
    x_decile
)


ax_d.set_xlabel(
    "Long-term greening intensity decile"
)


ax_d.set_ylabel(
    "Severity-adjusted legacy anomaly (days)"
)


ax_d.set_title(
    "Greening–persistence association"
)


clean_axis(
    ax_d
)


panel_label(
    ax_d,
    "d"
)


# ======================================================================
# PANEL E
# 2D hydroclimatic resilience surface
# ======================================================================

surface_valid = surface_mean[
    np.isfinite(
        surface_mean
    )
]


if len(
    surface_valid
) > 0:

    surf_abs = np.nanpercentile(
        np.abs(
            surface_valid
        ),
        95
    )


else:

    surf_abs = 1.0


im = ax_e.imshow(

    surface_mean,

    origin="lower",

    aspect="auto",

    interpolation="nearest",

    cmap="RdBu_r",

    vmin=-surf_abs,
    vmax=surf_abs,

    extent=[
        0.5,
        10.5,
        0.5,
        10.5
    ]
)


cb_e = fig.colorbar(

    im,

    ax=ax_e,

    pad=0.025
)


cb_e.set_label(
    "Severity-adjusted CDH damage"
)


ax_e.set_xticks(
    x_decile
)


ax_e.set_yticks(
    x_decile
)


ax_e.set_xlabel(
    "Greening intensity decile"
)


ax_e.set_ylabel(
    "RootSM availability decile"
)


ax_e.set_title(
    "Hydroclimatic resilience surface"
)


panel_label(
    ax_e,
    "e"
)


# ======================================================================
# PANEL F
# Greening resilience benefit
# ======================================================================

sm_deciles = np.arange(
    1,
    N_SM_SURFACE_BINS + 1
)


ax_f.plot(

    sm_deciles,
    benefit,

    marker="o",

    markersize=5,

    linewidth=2.0,

    color="#3B6E8F"
)


ax_f.axhline(

    0,

    color="0.25",

    linewidth=0.9,

    linestyle="--"
)


ax_f.fill_between(

    sm_deciles,

    0,
    benefit,

    where=(
        benefit
        >=
        0
    ),

    alpha=0.10,

    color="#4A8C68"
)


ax_f.fill_between(

    sm_deciles,

    0,
    benefit,

    where=(
        benefit
        <
        0
    ),

    alpha=0.10,

    color="#B65858"
)


ax_f.text(

    0.04,
    0.94,

    "Benefit",

    transform=ax_f.transAxes,

    ha="left",
    va="top",

    fontsize=8,

    color="#3F7956"
)


ax_f.text(

    0.04,
    0.06,

    "Potential reversal",

    transform=ax_f.transAxes,

    ha="left",
    va="bottom",

    fontsize=8,

    color="#9C4848"
)


# ======================================================================
# 22. DETECT ZERO CROSSING
# ======================================================================

crossing_x = None


for i in range(
    len(
        benefit
    ) - 1
):

    y1 = benefit[
        i
    ]

    y2 = benefit[
        i + 1
    ]


    if not (
        np.isfinite(
            y1
        )
        and
        np.isfinite(
            y2
        )
    ):

        continue


    if (
        y1 == 0
        or
        y1 * y2 < 0
    ):

        x1 = sm_deciles[
            i
        ]

        x2 = sm_deciles[
            i + 1
        ]


        if y2 != y1:

            crossing_x = (

                x1
                +
                (
                    0 - y1
                )
                *
                (
                    x2 - x1
                )
                /
                (
                    y2 - y1
                )
            )


        else:

            crossing_x = x1


        break


if crossing_x is not None:

    ax_f.axvline(

        crossing_x,

        color="#A84D4D",

        linestyle=":",

        linewidth=1.2
    )


    ax_f.text(

        crossing_x,

        ax_f.get_ylim()[1]
        if ax_f.get_ylim()[1] > 0
        else 0,

        f"  transition ≈ D{crossing_x:.1f}",

        rotation=90,

        va="top",

        fontsize=7.5,

        color="#A84D4D"
    )


ax_f.set_xticks(
    sm_deciles
)


ax_f.set_xlabel(
    "RootSM availability decile"
)


ax_f.set_ylabel(
    "Greening resilience benefit"
)


ax_f.set_title(
    "Water dependence of greening benefit"
)


clean_axis(
    ax_f
)


panel_label(
    ax_f,
    "f"
)


# ======================================================================
# 23. MAIN FIGURE TITLE
# ======================================================================

fig.suptitle(

    "Hydroclimatic regulation of greening-associated ecosystem resilience",

    fontsize=14,

    fontweight="bold",

    y=0.985
)


# ======================================================================
# 24. SAVE
# ======================================================================

sep(
    "SAVE FIGURE"
)


fig.savefig(

    OUT_PNG,

    dpi=600,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    OUT_TIF,

    dpi=600,

    bbox_inches="tight",

    facecolor="white"
)


fig.savefig(

    OUT_PDF,

    bbox_inches="tight",

    facecolor="white"
)


plt.close(
    fig
)


# ======================================================================
# 25. FINAL QC SUMMARY
# ======================================================================

sep(
    "FIGURE 8 FINAL SUMMARY"
)


print(
    "CDH event rows scanned:",
    f"{total_rows:,}"
)


print(
    "Valid severity-adjusted damage rows:",
    f"{resistance_rows:,}"
)


print(
    "Valid severity-adjusted legacy rows:",
    f"{legacy_rows:,}"
)


print(
    "\nGreening trend:"
)

print(
    "  valid pixels =",
    np.isfinite(
        nirv_slope
    ).sum()
)

print(
    "  median slope/decade =",
    np.nanmedian(
        nirv_slope
    )
)


print(
    "\nRootSM:"
)

print(
    "  median =",
    np.nanmedian(
        rootsm_clim
    )
)


print(
    "\nGreening benefit by RootSM decile:"
)


print(
    benefit_df.to_string(
        index=False
    )
)


if crossing_x is None:

    print(
        "\nNo robust sign reversal detected "
        "across RootSM deciles."
    )

else:

    print(
        "\nPotential greening-benefit sign transition "
        f"near RootSM decile {crossing_x:.2f}."
    )


print(
    "\nIMPORTANT INTERPRETATION:"
)

print(
    "This is an observational greening-resilience association."
)

print(
    "NIRv trend is NOT yet a causal ecological-restoration treatment."
)

print(
    "Causal attribution requires restoration exposure data "
    "and DML / causal forest analysis."
)


print(
    "\nOutputs:"
)

print(
    "PNG     :",
    OUT_PNG
)

print(
    "TIFF    :",
    OUT_TIF
)

print(
    "PDF     :",
    OUT_PDF
)

print(
    "Proxy NC:",
    OUT_PROXY_NC
)

print(
    "Gradient:",
    OUT_GRADIENT
)

print(
    "Surface :",
    OUT_SURFACE
)

print(
    "Benefit :",
    OUT_BENEFIT
)


eco.close()
clim.close()


print(
    "\nDONE."
)