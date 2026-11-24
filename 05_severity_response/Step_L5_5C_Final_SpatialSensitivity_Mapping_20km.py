# -*- coding: utf-8 -*-

"""
Step_L5_5C_Final_SpatialSensitivity_Mapping_20km.py
================================================================================

FINAL 20-km Basin-Anchored Spatial Sensitivity Mapping
Weihe River Basin Ecological Resilience Analysis

Scientific status entering L5.5C
--------------------------------
CDH
    Q* = 90.5958
    Formal B5 supported blocks = 131
    Main analysis retains STRICT B5 support.
    B6 MIN_POST_BINS=2 is sensitivity evidence only and is NOT used
    to replace the formal CDH mapping support criterion.

HEAT
    Q* = 91.8278
    Formal B5 supported blocks = 325

HR
    Q* = 93.0716
    Formal B5 supported blocks = 240

Important principles
--------------------
1. NO model refitting.
2. NO new breakpoint search.
3. NO interpolation over unsupported blocks.
4. NO relaxation of CDH support criterion in primary mapping.
5. Unresolved blocks are explicitly represented.
6. Coordinates are reconstructed from the formal L3 1-km grid.
7. Final maps use the same 20-km spatial grid for CDH / HEAT / HR.

Outputs
-------
A. Event-specific final mapping CSVs
B. Combined final mapping CSV
C. Spatial sensitivity summary CSV
D. SSI maps
E. Threshold response maps
F. Statistical support maps
G. Final 3 x 3 publication figure
H. PNG 600 dpi + vector PDF
I. Final QC/report text

================================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


warnings.filterwarnings("ignore")


# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
    r"\L5_5_SpatialHeterogeneity"
)

L3_NC = (
    r".\Weihe database V3"
    r"\Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
)


EVENT_FILES = {

    "CDH":
        os.path.join(
            ROOT,
            "L5_5B5_CDH_BasinAnchored_SpatialSensitivity_20km_FINAL.csv"
        ),

    "HEAT":
        os.path.join(
            ROOT,
            "L5_5B5_HEAT_BasinAnchored_SpatialSensitivity_20km_FINAL.csv"
        ),

    "HR":
        os.path.join(
            ROOT,
            "L5_5B5_HR_BasinAnchored_SpatialSensitivity_20km_FINAL.csv"
        )
}


OUT_DIR = os.path.join(
    ROOT,
    "L5_5C_FinalSpatialMapping_20km"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# CSV

OUT_COMBINED_CSV = os.path.join(
    OUT_DIR,
    "L5_5C_AllEvents_FinalSpatialSensitivity_20km.csv"
)

OUT_SUMMARY_CSV = os.path.join(
    OUT_DIR,
    "L5_5C_FinalSpatialSensitivity_20km_Summary.csv"
)

OUT_RESPONSE_CSV = os.path.join(
    OUT_DIR,
    "L5_5C_ThresholdResponse_Distribution_20km.csv"
)

OUT_SUPPORT_CSV = os.path.join(
    OUT_DIR,
    "L5_5C_SpatialSupport_Distribution_20km.csv"
)

OUT_REPORT = os.path.join(
    OUT_DIR,
    "L5_5C_FinalSpatialSensitivity_20km_Report.txt"
)


# Figures

FIG_SSI_PNG = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_SpatialSensitivityIndex_20km.png"
)

FIG_SSI_PDF = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_SpatialSensitivityIndex_20km.pdf"
)

FIG_RESPONSE_PNG = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_ThresholdResponseType_20km.png"
)

FIG_RESPONSE_PDF = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_ThresholdResponseType_20km.pdf"
)

FIG_SUPPORT_PNG = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_StatisticalSupport_20km.png"
)

FIG_SUPPORT_PDF = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_StatisticalSupport_20km.pdf"
)

FIG_MASTER_PNG = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_FinalSpatialSensitivity_3x3_20km.png"
)

FIG_MASTER_PDF = os.path.join(
    OUT_DIR,
    "L5_5C_Fig_FinalSpatialSensitivity_3x3_20km.pdf"
)


# =============================================================================
# 1. SCIENTIFIC SETTINGS
# =============================================================================

EVENT_ORDER = [
    "CDH",
    "HEAT",
    "HR"
]


EVENT_LABELS = {

    "CDH":
        "Compound drought–heat",

    "HEAT":
        "Heat",

    "HR":
        "Heavy rainfall"
}


QSTAR = {

    "CDH":
        90.5958,

    "HEAT":
        91.8278,

    "HR":
        93.0716
}


# Formal B5-supported counts expected from previous QC

EXPECTED_SUPPORTED = {

    "CDH":
        131,

    "HEAT":
        325,

    "HR":
        240
}


# Approximate 20-km blocks = 20 x 20 formal 1-km cells

BLOCK_PIXELS = 20


# =============================================================================
# 2. PUBLICATION STYLE
# =============================================================================

plt.rcParams.update({

    "font.family":
        "Times New Roman",

    "font.size":
        10,

    "axes.titlesize":
        11,

    "axes.labelsize":
        10,

    "xtick.labelsize":
        8,

    "ytick.labelsize":
        8,

    "legend.fontsize":
        8,

    "figure.dpi":
        150,

    "savefig.dpi":
        600,

    "axes.linewidth":
        0.7,

    "xtick.major.width":
        0.7,

    "ytick.major.width":
        0.7,

    "pdf.fonttype":
        42,

    "ps.fonttype":
        42
})


# =============================================================================
# 3. COLORS
# =============================================================================

# Continuous SSI
SSI_CMAP = plt.get_cmap(
    "RdBu_r"
)


# Threshold-response categories

RESPONSE_CODE = {

    "DECELERATING":
        0,

    "NEAR_LINEAR":
        1,

    "ACCELERATING":
        2,

    "UNRESOLVED":
        3
}


RESPONSE_COLORS = [

    "#3B75AF",     # decelerating

    "#D8D8D8",     # near linear

    "#C74343",     # accelerating

    "#EFEFEF"      # unresolved
]


RESPONSE_CMAP = ListedColormap(
    RESPONSE_COLORS
)


RESPONSE_NORM = mcolors.BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5],
    RESPONSE_CMAP.N
)


# Support categories

SUPPORT_CODE = {

    "SUPPORTED":
        1,

    "UNSUPPORTED":
        0,

    "UNOCCUPIED":
        -1
}


SUPPORT_COLORS = [

    "#F2F2F2",     # unoccupied

    "#CFCFCF",     # unsupported

    "#303030"      # supported
]


SUPPORT_CMAP = ListedColormap(
    SUPPORT_COLORS
)


SUPPORT_NORM = mcolors.BoundaryNorm(
    [-1.5, -0.5, 0.5, 1.5],
    SUPPORT_CMAP.N
)


# =============================================================================
# 4. UTILITY FUNCTIONS
# =============================================================================

def sep(title):

    print(
        "\n"
        +
        "=" * 120
    )

    print(title)

    print(
        "=" * 120
    )


def sub(title):

    print(
        "\n"
        +
        "-" * 120
    )

    print(title)

    print(
        "-" * 120
    )


def finite_median(x):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return np.nan

    return float(
        np.median(x)
    )


def finite_quantile(
    x,
    q
):

    x = np.asarray(
        x,
        dtype=float
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:

        return np.nan

    return float(
        np.quantile(
            x,
            q
        )
    )


# =============================================================================
# 5. COORDINATE EDGE FUNCTION
# =============================================================================

def coordinate_edges(
    centers
):

    """
    Convert 1-D cell-center coordinates to cell-edge coordinates.

    Works for ascending or descending coordinates.
    """

    centers = np.asarray(
        centers,
        dtype=np.float64
    )

    if centers.ndim != 1:

        raise ValueError(
            "Coordinate must be one-dimensional."
        )


    if len(centers) < 2:

        raise ValueError(
            "At least two coordinate values required."
        )


    edges = np.empty(
        len(centers) + 1,
        dtype=np.float64
    )


    edges[1:-1] = (
        centers[:-1]
        +
        centers[1:]
    ) / 2.0


    edges[0] = (
        centers[0]
        -
        (
            centers[1]
            -
            centers[0]
        )
        /
        2.0
    )


    edges[-1] = (
        centers[-1]
        +
        (
            centers[-1]
            -
            centers[-2]
        )
        /
        2.0
    )


    return edges


# =============================================================================
# 6. LOAD L3 GRID
# =============================================================================

sep(
    "OPEN FORMAL L3 GRID"
)


if not os.path.exists(
    L3_NC
):

    raise FileNotFoundError(
        L3_NC
    )


ds = xr.open_dataset(
    L3_NC
)


print(
    ds
)


if (
    "x"
    not in ds.coords
    or
    "y"
    not in ds.coords
):

    raise RuntimeError(
        "L3 must contain x and y coordinates."
    )


x = np.asarray(
    ds["x"].values,
    dtype=np.float64
)


y = np.asarray(
    ds["y"].values,
    dtype=np.float64
)


NX = len(x)

NY = len(y)


print()

print(
    "L3 grid:",
    f"NY={NY}",
    f"NX={NX}"
)

print(
    "X range:",
    x.min(),
    "->",
    x.max()
)

print(
    "Y range:",
    y.min(),
    "->",
    y.max()
)


# Expected from formal L3:
# NY = 414
# NX = 703


N_BLOCK_Y = int(
    np.ceil(
        NY
        /
        BLOCK_PIXELS
    )
)


N_BLOCK_X = int(
    np.ceil(
        NX
        /
        BLOCK_PIXELS
    )
)


print()

print(
    "20-km block grid:",
    f"{N_BLOCK_Y} rows x {N_BLOCK_X} cols"
)

print(
    "Total possible blocks:",
    N_BLOCK_Y
    *
    N_BLOCK_X
)


# =============================================================================
# 7. BUILD BLOCK EDGES
# =============================================================================

sep(
    "BUILD 20-km BLOCK COORDINATES"
)


x_pixel_edges = coordinate_edges(
    x
)

y_pixel_edges = coordinate_edges(
    y
)


x_block_index = list(
    range(
        0,
        NX,
        BLOCK_PIXELS
    )
)


if x_block_index[-1] != NX:

    x_block_edges_idx = (
        x_block_index
        +
        [NX]
    )

else:

    x_block_edges_idx = (
        x_block_index
    )


y_block_index = list(
    range(
        0,
        NY,
        BLOCK_PIXELS
    )
)


if y_block_index[-1] != NY:

    y_block_edges_idx = (
        y_block_index
        +
        [NY]
    )

else:

    y_block_edges_idx = (
        y_block_index
    )


x_edges_20 = x_pixel_edges[
    x_block_edges_idx
]


y_edges_20 = y_pixel_edges[
    y_block_edges_idx
]


if len(
    x_edges_20
) != N_BLOCK_X + 1:

    raise RuntimeError(
        "20-km X-edge construction failed."
    )


if len(
    y_edges_20
) != N_BLOCK_Y + 1:

    raise RuntimeError(
        "20-km Y-edge construction failed."
    )


# Center coordinate for each block

x_center_20 = (
    x_edges_20[:-1]
    +
    x_edges_20[1:]
) / 2.0


y_center_20 = (
    y_edges_20[:-1]
    +
    y_edges_20[1:]
) / 2.0


print(
    "X block centers:",
    len(
        x_center_20
    )
)

print(
    "Y block centers:",
    len(
        y_center_20
    )
)


# =============================================================================
# 8. READ B5 EVENT RESULTS
# =============================================================================

sep(
    "LOAD FINAL B5 EVENT-SPECIFIC SPATIAL RESULTS"
)


event_data = {}


for event in EVENT_ORDER:

    sub(
        event
    )


    path = EVENT_FILES[
        event
    ]


    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            path
        )


    df = pd.read_csv(
        path
    )


    print(
        "Rows:",
        len(df)
    )


    required = [

        "event_type",

        "block_id",

        "block_y",

        "block_x",

        "row_start",

        "row_end",

        "col_start",

        "col_end",

        "sampling_adequate",

        "fit_valid",

        "fit_supported",

        "fit_reason",

        "Qstar_basin",

        "beta_pre",

        "beta_post",

        "delta_beta",

        "PostThreshold_Amplification_5Q",

        "CrossThreshold_Contrast_10Q",

        "Spatial_Sensitivity_Index",

        "spatial_sensitivity_class",

        "threshold_response_type"
    ]


    missing = [

        c
        for c in required
        if c not in df.columns
    ]


    if missing:

        raise RuntimeError(

            f"{event} missing columns:\n"

            +

            "\n".join(
                missing
            )
        )


    # -------------------------------------------------------------
    # QC Q*
    # -------------------------------------------------------------

    q_values = (
        df[
            "Qstar_basin"
        ]
        .dropna()
        .unique()
    )


    if len(
        q_values
    ) > 0:

        q_actual = float(
            q_values[0]
        )


        print(
            "Q*:",
            q_actual
        )


        if not np.isclose(
            q_actual,
            QSTAR[event],
            atol=1e-3
        ):

            raise RuntimeError(

                f"{event} Q* mismatch: "
                f"{q_actual} vs "
                f"{QSTAR[event]}"
            )


    # -------------------------------------------------------------
    # Main-analysis support decision
    # -------------------------------------------------------------

    # IMPORTANT:
    #
    # For CDH we explicitly retain formal B5 fit_supported.
    # The B6 MIN_POST_BINS=2 sensitivity result does NOT overwrite
    # the primary mapping.
    #
    # HEAT and HR also use formal B5 fit_supported.

    df[
        "Final_Map_Supported"
    ] = (
        df[
            "fit_supported"
        ]
        ==
        True
    )


    df[
        "Primary_Mapping_Rule"
    ] = (
        "STRICT_B5_FORMAL_SUPPORT"
    )


    if event == "CDH":

        df[
            "Sensitivity_Note"
        ] = (
            "B6 showed MIN_POST_BINS=2 reaches ~50% coverage; "
            "primary map retains strict B5 support."
        )

    else:

        df[
            "Sensitivity_Note"
        ] = (
            "Formal B5 support used."
        )


    n_supported = int(
        df[
            "Final_Map_Supported"
        ].sum()
    )


    print(
        "Formal supported:",
        n_supported
    )


    if n_supported != EXPECTED_SUPPORTED[
        event
    ]:

        print(
            "WARNING: expected supported count:",
            EXPECTED_SUPPORTED[
                event
            ]
        )


    event_data[
        event
    ] = df


# =============================================================================
# 9. ADD GEOGRAPHIC COORDINATES
# =============================================================================

sep(
    "ADD 20-km GEOGRAPHIC COORDINATES"
)


combined_rows = []


for event in EVENT_ORDER:

    df = event_data[
        event
    ].copy()


    lon_center = []

    lat_center = []

    lon_min = []

    lon_max = []

    lat_min = []

    lat_max = []


    for _, row in df.iterrows():

        by = int(
            row[
                "block_y"
            ]
        )

        bx = int(
            row[
                "block_x"
            ]
        )


        if not (
            0
            <=
            by
            <
            N_BLOCK_Y
        ):

            raise RuntimeError(
                f"Invalid block_y={by}"
            )


        if not (
            0
            <=
            bx
            <
            N_BLOCK_X
        ):

            raise RuntimeError(
                f"Invalid block_x={bx}"
            )


        lon_center.append(
            x_center_20[
                bx
            ]
        )


        lat_center.append(
            y_center_20[
                by
            ]
        )


        x0 = x_edges_20[
            bx
        ]

        x1 = x_edges_20[
            bx + 1
        ]


        y0 = y_edges_20[
            by
        ]

        y1 = y_edges_20[
            by + 1
        ]


        lon_min.append(
            min(
                x0,
                x1
            )
        )

        lon_max.append(
            max(
                x0,
                x1
            )
        )

        lat_min.append(
            min(
                y0,
                y1
            )
        )

        lat_max.append(
            max(
                y0,
                y1
            )
        )


    df[
        "longitude"
    ] = lon_center


    df[
        "latitude"
    ] = lat_center


    df[
        "lon_min"
    ] = lon_min


    df[
        "lon_max"
    ] = lon_max


    df[
        "lat_min"
    ] = lat_min


    df[
        "lat_max"
    ] = lat_max


    event_data[
        event
    ] = df


    combined_rows.append(
        df
    )


combined = pd.concat(
    combined_rows,
    ignore_index=True
)


combined.to_csv(
    OUT_COMBINED_CSV,
    index=False
)


print(
    "Combined mapping CSV:",
    OUT_COMBINED_CSV
)


# =============================================================================
# 10. SAVE EVENT-SPECIFIC FINAL MAPPING CSV
# =============================================================================

sep(
    "SAVE EVENT-SPECIFIC FINAL MAPPING TABLES"
)


for event in EVENT_ORDER:

    path = os.path.join(

        OUT_DIR,

        f"L5_5C_{event}_FinalSpatialSensitivity_20km.csv"
    )


    event_data[
        event
    ].to_csv(
        path,
        index=False
    )


    print(
        event,
        ":",
        path
    )


# =============================================================================
# 11. BUILD 2-D ARRAYS
# =============================================================================

sep(
    "BUILD FINAL 20-km MAP ARRAYS"
)


map_arrays = {}


for event in EVENT_ORDER:

    df = event_data[
        event
    ]


    ssi = np.full(
        (
            N_BLOCK_Y,
            N_BLOCK_X
        ),
        np.nan,
        dtype=np.float64
    )


    response = np.full(
        (
            N_BLOCK_Y,
            N_BLOCK_X
        ),
        -1,
        dtype=np.int16
    )


    support = np.full(
        (
            N_BLOCK_Y,
            N_BLOCK_X
        ),
        -1,
        dtype=np.int16
    )


    occupied = np.zeros(
        (
            N_BLOCK_Y,
            N_BLOCK_X
        ),
        dtype=bool
    )


    for _, row in df.iterrows():

        by = int(
            row[
                "block_y"
            ]
        )

        bx = int(
            row[
                "block_x"
            ]
        )


        occupied[
            by,
            bx
        ] = True


        is_supported = bool(
            row[
                "Final_Map_Supported"
            ]
        )


        if is_supported:

            support[
                by,
                bx
            ] = 1


            value = row[
                "Spatial_Sensitivity_Index"
            ]


            if np.isfinite(
                value
            ):

                ssi[
                    by,
                    bx
                ] = float(
                    value
                )


            response_name = str(
                row[
                    "threshold_response_type"
                ]
            ).upper()


            response[
                by,
                bx
            ] = RESPONSE_CODE.get(
                response_name,
                RESPONSE_CODE[
                    "UNRESOLVED"
                ]
            )


        else:

            support[
                by,
                bx
            ] = 0


            response[
                by,
                bx
            ] = RESPONSE_CODE[
                "UNRESOLVED"
            ]


    map_arrays[
        event
    ] = {

        "ssi":
            ssi,

        "response":
            response,

        "support":
            support,

        "occupied":
            occupied
    }


    print(
        event,
        "| occupied=",
        int(
            occupied.sum()
        ),
        "| supported=",
        int(
            np.sum(
                support
                ==
                1
            )
        )
    )


# =============================================================================
# 12. SSI COMMON COLOR LIMIT
# =============================================================================

sep(
    "CALCULATE COMMON SSI COLOR RANGE"
)


all_ssi = []


for event in EVENT_ORDER:

    a = map_arrays[
        event
    ][
        "ssi"
    ]


    vals = a[
        np.isfinite(a)
    ]


    all_ssi.extend(
        vals.tolist()
    )


all_ssi = np.asarray(
    all_ssi,
    dtype=float
)


if len(
    all_ssi
) == 0:

    raise RuntimeError(
        "No finite SSI values available."
    )


# Robust common scale
# Use 98th percentile absolute magnitude.

ssi_lim = float(
    np.quantile(
        np.abs(
            all_ssi
        ),
        0.98
    )
)


if not np.isfinite(
    ssi_lim
) or ssi_lim <= 0:

    ssi_lim = 1.0


print(
    "Common SSI range:",
    f"{-ssi_lim:.4f}",
    "to",
    f"{ssi_lim:.4f}"
)


SSI_NORM = mcolors.TwoSlopeNorm(

    vmin=-ssi_lim,

    vcenter=0.0,

    vmax=ssi_lim
)


# =============================================================================
# 13. SUMMARY TABLES
# =============================================================================

sep(
    "BUILD FINAL L5.5C SUMMARY"
)


summary_rows = []

response_rows = []

support_rows = []


for event in EVENT_ORDER:

    df = event_data[
        event
    ]


    adequate = (
        df[
            "sampling_adequate"
        ]
        ==
        True
    )


    supported = (
        df[
            "Final_Map_Supported"
        ]
        ==
        True
    )


    ds = df.loc[
        supported
    ].copy()


    n_occupied = len(
        df
    )


    n_adequate = int(
        adequate.sum()
    )


    n_supported = int(
        supported.sum()
    )


    coverage = (
        n_supported
        /
        n_adequate
        if n_adequate > 0
        else np.nan
    )


    summary_rows.append({

        "event_type":
            event,

        "Qstar_basin":
            QSTAR[
                event
            ],

        "occupied_blocks":
            n_occupied,

        "sampling_adequate_blocks":
            n_adequate,

        "formal_supported_blocks":
            n_supported,

        "supported_fraction_adequate":
            coverage,

        "median_SSI":
            finite_median(
                ds[
                    "Spatial_Sensitivity_Index"
                ]
            ),

        "SSI_Q25":
            finite_quantile(
                ds[
                    "Spatial_Sensitivity_Index"
                ],
                0.25
            ),

        "SSI_Q75":
            finite_quantile(
                ds[
                    "Spatial_Sensitivity_Index"
                ],
                0.75
            ),

        "median_beta_pre":
            finite_median(
                ds[
                    "beta_pre"
                ]
            ),

        "median_beta_post":
            finite_median(
                ds[
                    "beta_post"
                ]
            ),

        "median_delta_beta":
            finite_median(
                ds[
                    "delta_beta"
                ]
            ),

        "median_PostThreshold_Amplification_5Q":
            finite_median(
                ds[
                    "PostThreshold_Amplification_5Q"
                ]
            ),

        "median_CrossThreshold_Contrast_10Q":
            finite_median(
                ds[
                    "CrossThreshold_Contrast_10Q"
                ]
            ),

        "primary_mapping_rule":
            "STRICT_B5_FORMAL_SUPPORT"
    })


    # ---------------------------------------------------------
    # Response distribution
    # ---------------------------------------------------------

    response_counts = (
        df[
            "threshold_response_type"
        ]
        .fillna(
            "UNRESOLVED"
        )
        .value_counts()
    )


    for response_name, n in response_counts.items():

        response_rows.append({

            "event_type":
                event,

            "threshold_response_type":
                response_name,

            "n_blocks":
                int(
                    n
                ),

            "fraction_occupied":
                float(
                    n
                    /
                    n_occupied
                )
        })


    # ---------------------------------------------------------
    # Support distribution
    # ---------------------------------------------------------

    support_rows.append({

        "event_type":
            event,

        "support_status":
            "SUPPORTED",

        "n_blocks":
            n_supported,

        "fraction_adequate":
            coverage
    })


    support_rows.append({

        "event_type":
            event,

        "support_status":
            "UNSUPPORTED",

        "n_blocks":
            int(
                n_occupied
                -
                n_supported
            ),

        "fraction_occupied":
            float(
                (
                    n_occupied
                    -
                    n_supported
                )
                /
                n_occupied
            )
    })


summary_df = pd.DataFrame(
    summary_rows
)


response_df = pd.DataFrame(
    response_rows
)


support_df = pd.DataFrame(
    support_rows
)


summary_df.to_csv(
    OUT_SUMMARY_CSV,
    index=False
)


response_df.to_csv(
    OUT_RESPONSE_CSV,
    index=False
)


support_df.to_csv(
    OUT_SUPPORT_CSV,
    index=False
)


print()

print(
    summary_df.to_string(
        index=False
    )
)


# =============================================================================
# 14. COMMON MAP FORMATTING
# =============================================================================

def format_geo_axis(
    ax,
    title=None
):

    if title is not None:

        ax.set_title(
            title,
            pad=7
        )


    ax.set_xlabel(
        "Longitude (°E)"
    )


    ax.set_ylabel(
        "Latitude (°N)"
    )


    ax.tick_params(
        direction="out",
        length=3
    )


    for spine in ax.spines.values():

        spine.set_linewidth(
            0.7
        )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


# =============================================================================
# 15. SSI FIGURE
# =============================================================================

sep(
    "DRAW SSI MAPS"
)


fig, axes = plt.subplots(

    1,
    3,

    figsize=(
        12.0,
        4.0
    ),

    constrained_layout=True
)


mappable = None


for i, event in enumerate(
    EVENT_ORDER
):

    ax = axes[
        i
    ]


    a = map_arrays[
        event
    ][
        "ssi"
    ]


    mappable = ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        a,

        cmap=SSI_CMAP,

        norm=SSI_NORM,

        shading="flat",

        rasterized=True
    )


    supported = EXPECTED_SUPPORTED[
        event
    ]


    adequate = int(
        (
            event_data[
                event
            ][
                "sampling_adequate"
            ]
            ==
            True
        ).sum()
    )


    coverage = (
        supported
        /
        adequate
    )


    title = (

        f"{chr(97+i)}) "
        f"{EVENT_LABELS[event]}\n"
        f"Q*={QSTAR[event]:.2f}, "
        f"support={coverage:.1%}"
    )


    format_geo_axis(
        ax,
        title
    )


cbar = fig.colorbar(

    mappable,

    ax=axes,

    orientation="horizontal",

    fraction=0.05,

    pad=0.08
)


cbar.set_label(
    "Spatial Sensitivity Index (SSI)"
)


fig.savefig(
    FIG_SSI_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_SSI_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    FIG_SSI_PNG
)


# =============================================================================
# 16. RESPONSE-TYPE FIGURE
# =============================================================================

sep(
    "DRAW THRESHOLD RESPONSE MAPS"
)


fig, axes = plt.subplots(

    1,
    3,

    figsize=(
        12.0,
        4.0
    ),

    constrained_layout=True
)


for i, event in enumerate(
    EVENT_ORDER
):

    ax = axes[
        i
    ]


    a = map_arrays[
        event
    ][
        "response"
    ].copy()


    # Convert unoccupied -1 to NaN so it stays blank.

    aa = a.astype(
        float
    )


    aa[
        a < 0
    ] = np.nan


    ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        aa,

        cmap=RESPONSE_CMAP,

        norm=RESPONSE_NORM,

        shading="flat",

        rasterized=True
    )


    title = (

        f"{chr(97+i)}) "
        f"{EVENT_LABELS[event]}"
    )


    format_geo_axis(
        ax,
        title
    )


legend_handles = [

    Patch(
        facecolor=
            RESPONSE_COLORS[
                RESPONSE_CODE[
                    "DECELERATING"
                ]
            ],
        edgecolor="none",
        label="Decelerating"
    ),

    Patch(
        facecolor=
            RESPONSE_COLORS[
                RESPONSE_CODE[
                    "NEAR_LINEAR"
                ]
            ],
        edgecolor="none",
        label="Near-linear"
    ),

    Patch(
        facecolor=
            RESPONSE_COLORS[
                RESPONSE_CODE[
                    "ACCELERATING"
                ]
            ],
        edgecolor="none",
        label="Accelerating"
    ),

    Patch(
        facecolor=
            RESPONSE_COLORS[
                RESPONSE_CODE[
                    "UNRESOLVED"
                ]
            ],
        edgecolor="none",
        label="Unresolved"
    )
]


fig.legend(

    handles=legend_handles,

    loc="lower center",

    ncol=4,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        -0.02
    )
)


fig.savefig(
    FIG_RESPONSE_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_RESPONSE_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    FIG_RESPONSE_PNG
)


# =============================================================================
# 17. SUPPORT FIGURE
# =============================================================================

sep(
    "DRAW STATISTICAL SUPPORT MAPS"
)


fig, axes = plt.subplots(

    1,
    3,

    figsize=(
        12.0,
        4.0
    ),

    constrained_layout=True
)


for i, event in enumerate(
    EVENT_ORDER
):

    ax = axes[
        i
    ]


    a = map_arrays[
        event
    ][
        "support"
    ]


    ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        a,

        cmap=SUPPORT_CMAP,

        norm=SUPPORT_NORM,

        shading="flat",

        rasterized=True
    )


    n_adequate = int(
        (
            event_data[
                event
            ][
                "sampling_adequate"
            ]
            ==
            True
        ).sum()
    )


    n_supported = int(
        event_data[
            event
        ][
            "Final_Map_Supported"
        ].sum()
    )


    coverage = (
        n_supported
        /
        n_adequate
    )


    title = (

        f"{chr(97+i)}) "
        f"{EVENT_LABELS[event]}\n"
        f"{n_supported}/{n_adequate} "
        f"({coverage:.1%})"
    )


    format_geo_axis(
        ax,
        title
    )


support_legend = [

    Patch(
        facecolor="#303030",
        edgecolor="none",
        label="Statistically supported"
    ),

    Patch(
        facecolor="#CFCFCF",
        edgecolor="none",
        label="Unsupported"
    ),

    Patch(
        facecolor="#F2F2F2",
        edgecolor="none",
        label="Unoccupied"
    )
]


fig.legend(

    handles=support_legend,

    loc="lower center",

    ncol=3,

    frameon=False,

    bbox_to_anchor=(
        0.5,
        -0.02
    )
)


fig.savefig(
    FIG_SUPPORT_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_SUPPORT_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    FIG_SUPPORT_PNG
)


# =============================================================================
# 18. FINAL 3 x 3 MASTER FIGURE
# =============================================================================

sep(
    "DRAW FINAL 3 x 3 MASTER FIGURE"
)


fig, axes = plt.subplots(

    3,
    3,

    figsize=(
        12.0,
        10.2
    ),

    constrained_layout=True
)


panel_letters = iter(
    list(
        "abcdefghi"
    )
)


ssi_mappable = None


for r, event in enumerate(
    EVENT_ORDER
):

    # -----------------------------------------------------------------
    # COLUMN 1 — SSI
    # -----------------------------------------------------------------

    ax = axes[
        r,
        0
    ]


    ssi = map_arrays[
        event
    ][
        "ssi"
    ]


    ssi_mappable = ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        ssi,

        cmap=SSI_CMAP,

        norm=SSI_NORM,

        shading="flat",

        rasterized=True
    )


    letter = next(
        panel_letters
    )


    format_geo_axis(

        ax,

        f"{letter}) "
        f"{EVENT_LABELS[event]} — SSI"
    )


    # -----------------------------------------------------------------
    # COLUMN 2 — Response type
    # -----------------------------------------------------------------

    ax = axes[
        r,
        1
    ]


    resp = map_arrays[
        event
    ][
        "response"
    ].astype(
        float
    )


    resp[
        resp < 0
    ] = np.nan


    ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        resp,

        cmap=RESPONSE_CMAP,

        norm=RESPONSE_NORM,

        shading="flat",

        rasterized=True
    )


    letter = next(
        panel_letters
    )


    format_geo_axis(

        ax,

        f"{letter}) "
        f"{EVENT_LABELS[event]} — response"
    )


    # -----------------------------------------------------------------
    # COLUMN 3 — statistical support
    # -----------------------------------------------------------------

    ax = axes[
        r,
        2
    ]


    support = map_arrays[
        event
    ][
        "support"
    ]


    ax.pcolormesh(

        x_edges_20,

        y_edges_20,

        support,

        cmap=SUPPORT_CMAP,

        norm=SUPPORT_NORM,

        shading="flat",

        rasterized=True
    )


    n_adequate = int(
        (
            event_data[
                event
            ][
                "sampling_adequate"
            ]
            ==
            True
        ).sum()
    )


    n_supported = int(
        event_data[
            event
        ][
            "Final_Map_Supported"
        ].sum()
    )


    coverage = (
        n_supported
        /
        n_adequate
    )


    letter = next(
        panel_letters
    )


    format_geo_axis(

        ax,

        f"{letter}) "
        f"{EVENT_LABELS[event]} — support "
        f"({coverage:.1%})"
    )


# -------------------------------------------------------------------------
# SSI colorbar
# -------------------------------------------------------------------------

cbar = fig.colorbar(

    ssi_mappable,

    ax=axes[:, 0],

    orientation="horizontal",

    fraction=0.035,

    pad=0.06
)


cbar.set_label(
    "Spatial Sensitivity Index (SSI)"
)


# -------------------------------------------------------------------------
# Response legend
# -------------------------------------------------------------------------

response_legend = [

    Patch(
        facecolor="#3B75AF",
        edgecolor="none",
        label="Decelerating"
    ),

    Patch(
        facecolor="#D8D8D8",
        edgecolor="none",
        label="Near-linear"
    ),

    Patch(
        facecolor="#C74343",
        edgecolor="none",
        label="Accelerating"
    ),

    Patch(
        facecolor="#EFEFEF",
        edgecolor="none",
        label="Unresolved"
    )
]


axes[
    2,
    1
].legend(

    handles=response_legend,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        -0.22
    ),

    ncol=2,

    frameon=False
)


# -------------------------------------------------------------------------
# Support legend
# -------------------------------------------------------------------------

axes[
    2,
    2
].legend(

    handles=support_legend,

    loc="upper center",

    bbox_to_anchor=(
        0.5,
        -0.22
    ),

    ncol=1,

    frameon=False
)


fig.savefig(
    FIG_MASTER_PNG,
    dpi=600,
    bbox_inches="tight"
)


fig.savefig(
    FIG_MASTER_PDF,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    FIG_MASTER_PNG
)


# =============================================================================
# 19. FINAL QC
# =============================================================================

sep(
    "FINAL L5.5C QC"
)


qc_pass = True


for event in EVENT_ORDER:

    df = event_data[
        event
    ]


    supported = int(
        df[
            "Final_Map_Supported"
        ].sum()
    )


    finite_ssi = int(
        np.isfinite(
            df.loc[
                df[
                    "Final_Map_Supported"
                ],
                "Spatial_Sensitivity_Index"
            ]
        ).sum()
    )


    print()

    print(
        event
    )


    print(
        "  supported blocks:",
        supported
    )


    print(
        "  finite SSI     :",
        finite_ssi
    )


    print(
        "  Q*             :",
        QSTAR[
            event
        ]
    )


    if supported != EXPECTED_SUPPORTED[
        event
    ]:

        print(
            "  WARNING: supported count differs from expected B5."
        )

        qc_pass = False


    if finite_ssi != supported:

        print(
            "  WARNING: supported block without finite SSI."
        )

        qc_pass = False


print()


if qc_pass:

    FINAL_STATUS = (
        "PASS_FINAL_MAPPING"
    )

else:

    FINAL_STATUS = (
        "CHECK_REQUIRED"
    )


print(
    "FINAL STATUS:",
    FINAL_STATUS
)


# =============================================================================
# 20. WRITE REPORT
# =============================================================================

sep(
    "WRITE FINAL L5.5C REPORT"
)


report = []


report.append(
    "Step L5.5C | Final 20-km Basin-Anchored Spatial Sensitivity Mapping"
)


report.append(
    "=" * 110
)


report.append("")


report.append(
    "PRIMARY SCIENTIFIC MAPPING RULE"
)


report.append(
    "All three event types use the formal L5.5B5 fit_supported criterion."
)


report.append(
    "CDH B6 MIN_POST_BINS=2 is retained only as sensitivity evidence "
    "and is not used to overwrite the primary CDH spatial map."
)


report.append("")


report.append(
    "Fixed basin thresholds:"
)


for event in EVENT_ORDER:

    report.append(
        f"  {event:<6s}: Q{QSTAR[event]:.4f}"
    )


report.append("")


report.append(
    "Formal supported blocks:"
)


for event in EVENT_ORDER:

    n = int(
        event_data[
            event
        ][
            "Final_Map_Supported"
        ].sum()
    )


    report.append(
        f"  {event:<6s}: {n}"
    )


report.append("")


report.append(
    "Spatial summary:"
)


report.append(
    summary_df.to_string(
        index=False
    )
)


report.append("")


report.append(
    "Interpretation framework:"
)


report.append(
    "  SSI map:"
)


report.append(
    "    Continuous basin-anchored spatial sensitivity."
)


report.append(
    "    Unsupported blocks are not interpolated."
)


report.append(
    "  Threshold-response map:"
)


report.append(
    "    ACCELERATING / DECELERATING / NEAR_LINEAR / UNRESOLVED."
)


report.append(
    "  Statistical-support map:"
)


report.append(
    "    Explicitly distinguishes supported, unsupported, and unoccupied blocks."
)


report.append("")


report.append(
    "CDH sensitivity decision:"
)


report.append(
    "  L5.5B6 showed that MIN_POST_BINS=2 raises CDH support to "
    "approximately 50%, with stable basin-level median metrics."
)


report.append(
    "  However, two-bin post-threshold slopes can be locally unstable."
)


report.append(
    "  Therefore the primary L5.5C CDH map retains strict B5-supported blocks."
)


report.append("")


report.append(
    "Outputs:"
)


report.append(
    f"  Combined CSV : {OUT_COMBINED_CSV}"
)


report.append(
    f"  Summary CSV  : {OUT_SUMMARY_CSV}"
)


report.append(
    f"  Response CSV : {OUT_RESPONSE_CSV}"
)


report.append(
    f"  Support CSV  : {OUT_SUPPORT_CSV}"
)


report.append(
    f"  SSI figure   : {FIG_SSI_PNG}"
)


report.append(
    f"  Response fig : {FIG_RESPONSE_PNG}"
)


report.append(
    f"  Support fig  : {FIG_SUPPORT_PNG}"
)


report.append(
    f"  Master figure: {FIG_MASTER_PNG}"
)


report.append("")


report.append(
    f"FINAL STATUS: {FINAL_STATUS}"
)


with open(
    OUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report
        )
    )


print(
    OUT_REPORT
)


# =============================================================================
# 21. CLOSE DATASET
# =============================================================================

ds.close()


# =============================================================================
# 22. COMPLETED
# =============================================================================

sep(
    "STEP L5.5C COMPLETED"
)


print(
    "Primary spatial scale : 20 km"
)


print(
    "CDH mapping rule      : STRICT B5 supported-only"
)


print(
    "HEAT mapping rule     : STRICT B5 supported-only"
)


print(
    "HR mapping rule       : STRICT B5 supported-only"
)


print(
    "Interpolation         : NONE"
)


print(
    "Final master figure  :",
    FIG_MASTER_PNG
)


print()

print(
    "FINAL STATUS:",
    FINAL_STATUS
)