# -*- coding: utf-8 -*-
"""
Step L4.3 - Build EcoFunction State Space and Composite Negative Anomaly (Robust V1)

Purpose
-------
Construct a 3-D ecological function state-space product for the Weihe River Basin
from the validated L4.2 phenology-normalized Z-score cube.

Input
-----
.\Weihe database V3\L4_EcoFunction\
Weihe_L4_EcoFunction_Zscore_1000m_2000_2024_V3_1.nc

Input variables:
    ZSync
    ZuWUE
    ZNIRv
    phase8

Outputs
-------
.\Weihe database V3\L4_EcoFunction\
Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc

Output variables:
    ZSync
    ZuWUE
    ZNIRv
    ZSync_cap
    ZuWUE_cap
    ZNIRv_cap
    Aneg_Sync
    Aneg_uWUE
    Aneg_NIRv
    A_comp
    EcoState
    phase8

Definitions
-----------
1. Winsorized/capped Z-scores for magnitude-sensitive calculations:

       Z_cap = min(max(Z, -Z_CAP_ABS), Z_CAP_ABS)

   The original Z variables are retained unchanged for auditability.

2. Dimension-specific negative anomalies:

       Aneg_i = max(0, -Z_cap_i)

3. Composite negative anomaly magnitude:

       A_comp = sqrt((Aneg_Sync^2 + Aneg_uWUE^2 + Aneg_NIRv^2) / 3)

4. Ecological function states:

       0 = NoData / invalid
       1 = Stable functional state
       2 = Early water-carbon decoupling
       3 = Water-use impairment
       4 = Carbon/photosynthetic impairment
       5 = Severe compound degradation

   State logic, using capped Z-scores:

       Stable:
           all three dimensions >= -0.5

       Severe compound degradation:
           at least two dimensions < -0.5
           OR A_comp >= 1.0

       Early water-carbon decoupling:
           ZSync < -0.5
           and ZuWUE >= -0.5
           and ZNIRv >= -0.5

       Water-use impairment:
           ZuWUE < -0.5
           and ZNIRv >= -0.5
           and not severe

       Carbon/photosynthetic impairment:
           ZNIRv < -0.5
           and A_comp < 1.0
           and not severe

Important notes
---------------
- This script does not modify the L4.2 Z-score product.
- Raw Z variables are copied into the L4.3 product unchanged.
- Capped Z variables are only used to prevent rare extreme Z values from
  dominating magnitude-sensitive composite metrics.
- State thresholds are intentionally conservative and interpretable.
- Processing is performed in y-blocks and written directly to NetCDF4.
"""

import os
import gc
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset, date2num

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. PATHS
# =============================================================================

L4_Z_NC = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
    r"\Weihe_L4_EcoFunction_Zscore_1000m_2000_2024_V3_1.nc"
)

OUT_DIR = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_STATE_NC = os.path.join(
    OUT_DIR,
    "Weihe_L4_EcoFunction_StateSpace_1000m_2000_2024_V3_1.nc"
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

# Capping is used only for magnitude-sensitive downstream metrics.
# Original Z variables are retained unchanged.
Z_CAP_ABS = 10.0

# State threshold in Z-score units.
Z_DAMAGE_THRESHOLD = -0.5

# Severe compound anomaly threshold.
ACOMP_SEVERE_THRESHOLD = 1.0

# Low-memory spatial block.
BLOCK_Y = 24

# NetCDF compression/chunking.
COMPLEVEL = 4
CHUNK_TIME = 16
CHUNK_Y = 24
CHUNK_X = 352

INPUT_Z_VARS = [
    "ZSync",
    "ZuWUE",
    "ZNIRv",
]

CAP_VARS = [
    "ZSync_cap",
    "ZuWUE_cap",
    "ZNIRv_cap",
]

ANEG_VARS = [
    "Aneg_Sync",
    "Aneg_uWUE",
    "Aneg_NIRv",
]

FLOAT_OUTPUT_VARS = (
    INPUT_Z_VARS
    +
    CAP_VARS
    +
    ANEG_VARS
    +
    [
        "A_comp"
    ]
)

STATE_LABELS = {
    0: "NoData_or_invalid",
    1: "Stable_functional_state",
    2: "Early_water_carbon_decoupling",
    3: "Water_use_impairment",
    4: "Carbon_photosynthetic_impairment",
    5: "Severe_compound_degradation",
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def sep(title):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def masked_to_nan(arr, dtype=np.float32):
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)

    arr = np.asarray(
        arr,
        dtype=dtype
    )

    arr = np.where(
        np.isfinite(arr)
        &
        (arr > -9000.0),
        arr,
        np.nan
    ).astype(
        dtype,
        copy=False
    )

    return arr


def safe_nan_stats(arr):
    finite = np.isfinite(arr)
    n = int(finite.sum())

    if n == 0:
        return np.nan, np.nan, np.nan, 100.0

    vals = arr[finite].astype(
        np.float64,
        copy=False
    )

    return (
        float(vals.min()),
        float(vals.max()),
        float(vals.mean()),
        100.0 * (arr.size - n) / arr.size,
    )


def make_time_coord(nc, time_index):
    nc.createDimension(
        "time",
        len(time_index)
    )

    tvar = nc.createVariable(
        "time",
        "i4",
        ("time",)
    )

    units = (
        "days since 1970-01-01 00:00:00"
    )

    calendar = "standard"

    py_dates = [
        ts.to_pydatetime()
        for ts in pd.DatetimeIndex(
            time_index
        )
    ]

    tvar[:] = date2num(
        py_dates,
        units=units,
        calendar=calendar
    ).astype(
        np.int32
    )

    tvar.units = units
    tvar.calendar = calendar
    tvar.standard_name = "time"
    tvar.axis = "T"


def make_xy_coords(nc, x, y):
    nc.createDimension(
        "y",
        len(y)
    )

    nc.createDimension(
        "x",
        len(x)
    )

    yvar = nc.createVariable(
        "y",
        "f8",
        ("y",)
    )

    xvar = nc.createVariable(
        "x",
        "f8",
        ("x",)
    )

    yvar[:] = np.asarray(
        y,
        dtype=np.float64
    )

    xvar[:] = np.asarray(
        x,
        dtype=np.float64
    )

    yvar.standard_name = "latitude"
    yvar.long_name = "latitude"
    yvar.units = "degrees_north"
    yvar.axis = "Y"

    xvar.standard_name = "longitude"
    xvar.long_name = "longitude"
    xvar.units = "degrees_east"
    xvar.axis = "X"


def copy_crs_if_present(xr_ds, nc):
    if "crs" not in xr_ds.variables:
        return

    src = xr_ds["crs"]
    crs = nc.createVariable(
        "crs",
        "i4"
    )

    try:
        src_value = np.asarray(
            src.values
        ).reshape(-1)

        if (
            src_value.size > 0
            and
            np.isfinite(src_value[0])
        ):
            crs.assignValue(
                np.int32(src_value[0])
            )
        else:
            crs.assignValue(
                np.int32(0)
            )
    except Exception:
        crs.assignValue(
            np.int32(0)
        )

    for key, value in src.attrs.items():
        try:
            setattr(
                crs,
                key,
                value
            )
        except Exception:
            setattr(
                crs,
                key,
                str(value)
            )


def block_scan_float_var(nc_path, var_name, block_y=BLOCK_Y):
    with Dataset(
        nc_path,
        "r"
    ) as nc:

        var = nc.variables[
            var_name
        ]

        nt, ny, nx = var.shape

        total = nt * ny * nx
        valid_n = 0
        sum_v = 0.0
        min_v = np.inf
        max_v = -np.inf

        count_abs_gt5 = 0
        count_abs_gt10 = 0

        for y0 in range(
            0,
            ny,
            block_y
        ):

            y1 = min(
                y0 + block_y,
                ny
            )

            arr = masked_to_nan(
                var[
                    :,
                    y0:y1,
                    :
                ],
                dtype=np.float32
            )

            finite = np.isfinite(
                arr
            )

            n = int(
                finite.sum()
            )

            if n > 0:

                vals = arr[
                    finite
                ].astype(
                    np.float64,
                    copy=False
                )

                valid_n += n
                sum_v += float(
                    vals.sum()
                )
                min_v = min(
                    min_v,
                    float(vals.min())
                )
                max_v = max(
                    max_v,
                    float(vals.max())
                )

                abs_vals = np.abs(
                    vals
                )

                count_abs_gt5 += int(
                    (
                        abs_vals > 5.0
                    ).sum()
                )

                count_abs_gt10 += int(
                    (
                        abs_vals > 10.0
                    ).sum()
                )

        if valid_n == 0:

            return {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "nodata_pct": 100.0,
                "valid_n": 0,
                "abs_gt5": 0,
                "abs_gt10": 0,
            }

        return {
            "min": min_v,
            "max": max_v,
            "mean": sum_v / valid_n,
            "nodata_pct": (
                100.0
                *
                (
                    total
                    -
                    valid_n
                )
                /
                total
            ),
            "valid_n": valid_n,
            "abs_gt5": count_abs_gt5,
            "abs_gt10": count_abs_gt10,
        }


def block_scan_state_counts(nc_path, block_y=BLOCK_Y):
    with Dataset(
        nc_path,
        "r"
    ) as nc:

        var = nc.variables[
            "EcoState"
        ]

        nt, ny, nx = var.shape

        counts = {
            k: 0
            for k in STATE_LABELS
        }

        total = nt * ny * nx

        for y0 in range(
            0,
            ny,
            block_y
        ):

            y1 = min(
                y0 + block_y,
                ny
            )

            arr = np.asarray(
                var[
                    :,
                    y0:y1,
                    :
                ],
                dtype=np.int16
            )

            for k in STATE_LABELS:

                counts[
                    k
                ] += int(
                    (
                        arr == k
                    ).sum()
                )

        rows = []

        for k in sorted(
            STATE_LABELS
        ):

            rows.append(
                {
                    "state_code": k,
                    "state_label": STATE_LABELS[k],
                    "count": counts[k],
                    "percent_of_all": (
                        100.0
                        *
                        counts[k]
                        /
                        total
                    ),
                }
            )

        return pd.DataFrame(
            rows
        )


# =============================================================================
# 3. OPEN AND VALIDATE INPUT
# =============================================================================

sep("OPEN VALIDATED L4.2 Z-SCORE DATASET")

require_file(
    L4_Z_NC,
    "L4.2 Z-score NetCDF"
)

print("Input:")
print(L4_Z_NC)

ds = xr.open_dataset(
    L4_Z_NC,
    decode_times=True,
    cache=False
)

print("\nDimensions:")
print(dict(ds.sizes))

for dim in (
    "time",
    "y",
    "x"
):

    if dim not in ds.dims:
        raise RuntimeError(
            f"Required dimension missing: {dim}"
        )

for var in INPUT_Z_VARS + ["phase8"]:

    if var not in ds.variables:
        raise KeyError(
            f"Required variable missing: {var}"
        )

print("\nRequired variables:")
for var in INPUT_Z_VARS + ["phase8"]:
    print(
        f"{var:10s}: FOUND"
    )

time_index = pd.DatetimeIndex(
    pd.to_datetime(
        ds["time"].values
    )
)

x = np.asarray(
    ds["x"].values,
    dtype=np.float64
)

y = np.asarray(
    ds["y"].values,
    dtype=np.float64
)

phase8 = np.asarray(
    ds["phase8"].values,
    dtype=np.int8
)

nt = len(time_index)
ny = len(y)
nx = len(x)

print(
    "\nTime:",
    time_index[0].date(),
    "->",
    time_index[-1].date()
)

print(
    f"Grid: time={nt}, y={ny}, x={nx}"
)

print(
    f"Z capping threshold for magnitude metrics: ±{Z_CAP_ABS:g}"
)

print(
    f"Damage threshold: Z < {Z_DAMAGE_THRESHOLD:g}"
)

print(
    f"Severe compound threshold: A_comp >= {ACOMP_SEVERE_THRESHOLD:g}"
)

if (
    phase8.min() < 1
    or
    phase8.max() > 46
):
    raise RuntimeError(
        "phase8 must be in the range 1..46."
    )

# =============================================================================
# 4. CREATE OUTPUT NETCDF
# =============================================================================

sep("CREATE OUTPUT STATE-SPACE NETCDF")

if os.path.exists(
    OUT_STATE_NC
):
    print(
        "Removing existing output:\n"
        + OUT_STATE_NC
    )

    os.remove(
        OUT_STATE_NC
    )

nc_out = Dataset(
    OUT_STATE_NC,
    "w",
    format="NETCDF4"
)

make_time_coord(
    nc_out,
    time_index
)

make_xy_coords(
    nc_out,
    x,
    y
)

phase_var = nc_out.createVariable(
    "phase8",
    "i1",
    ("time",),
    zlib=True,
    complevel=COMPLEVEL
)

phase_var[:] = phase8
phase_var.long_name = (
    "calendar-anchored 8-day phenological phase"
)
phase_var.valid_min = np.int8(1)
phase_var.valid_max = np.int8(46)

copy_crs_if_present(
    ds,
    nc_out
)

created = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

nc_out.Conventions = "CF-1.8"
nc_out.title = (
    "Weihe River Basin L4.3 EcoFunction State Space Dataset"
)
nc_out.processing_level = (
    "L4.3 EcoFunction State Space"
)
nc_out.dataset_version = "V3.1"
nc_out.created_utc = created
nc_out.source = os.path.basename(
    L4_Z_NC
)
nc_out.time_coverage_start = str(
    time_index[0].date()
)
nc_out.time_coverage_end = str(
    time_index[-1].date()
)
nc_out.temporal_resolution = (
    "8-day master phases"
)
nc_out.spatial_resolution = (
    "1000 m master grid"
)
nc_out.state_space_dimensions = (
    "ZSync, ZuWUE, ZNIRv"
)
nc_out.original_z_retained = (
    "ZSync, ZuWUE, ZNIRv are copied from L4.2 without clipping."
)
nc_out.z_capping_for_magnitude_metrics = (
    f"Z_cap = clip(Z, -{Z_CAP_ABS:g}, {Z_CAP_ABS:g})"
)
nc_out.negative_anomaly_definition = (
    "Aneg_i = max(0, -Z_cap_i)"
)
nc_out.composite_anomaly_definition = (
    "A_comp = sqrt((Aneg_Sync^2 + Aneg_uWUE^2 + Aneg_NIRv^2) / 3)"
)
nc_out.damage_threshold = (
    f"Z < {Z_DAMAGE_THRESHOLD:g}"
)
nc_out.severe_compound_threshold = (
    f"A_comp >= {ACOMP_SEVERE_THRESHOLD:g} or at least two dimensions below damage threshold"
)
nc_out.state_code_0 = STATE_LABELS[0]
nc_out.state_code_1 = STATE_LABELS[1]
nc_out.state_code_2 = STATE_LABELS[2]
nc_out.state_code_3 = STATE_LABELS[3]
nc_out.state_code_4 = STATE_LABELS[4]
nc_out.state_code_5 = STATE_LABELS[5]

# =============================================================================
# 5. DEFINE OUTPUT VARIABLES
# =============================================================================

sep("DEFINE OUTPUT VARIABLES")

out_float_vars = {}

for var in FLOAT_OUTPUT_VARS:

    outv = nc_out.createVariable(
        var,
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=COMPLEVEL,
        shuffle=True,
        fill_value=np.float32(-9999.0),
        chunksizes=(
            min(CHUNK_TIME, nt),
            min(CHUNK_Y, ny),
            min(CHUNK_X, nx),
        )
    )

    outv.units = "1"

    if var in INPUT_Z_VARS:
        outv.long_name = (
            f"Original {var} from L4.2"
        )
        outv.processing = (
            "Copied unchanged from L4.2 phenology-normalized Z-score dataset"
        )

    elif var in CAP_VARS:
        source_var = var.replace(
            "_cap",
            ""
        )
        outv.long_name = (
            f"Capped {source_var} for magnitude-sensitive calculations"
        )
        outv.source_variable = source_var
        outv.capping = (
            f"clip({source_var}, -{Z_CAP_ABS:g}, {Z_CAP_ABS:g})"
        )

    elif var in ANEG_VARS:
        outv.long_name = (
            f"Negative anomaly magnitude of {var.replace('Aneg_', '')}"
        )
        outv.definition = (
            "max(0, -Z_cap)"
        )
        outv.comment = (
            "Positive values indicate negative functional anomaly intensity."
        )

    elif var == "A_comp":
        outv.long_name = (
            "Composite negative ecological function anomaly magnitude"
        )
        outv.definition = (
            "sqrt((Aneg_Sync^2 + Aneg_uWUE^2 + Aneg_NIRv^2) / 3)"
        )
        outv.comment = (
            "Calculated from capped Z-scores to prevent rare extreme values from "
            "dominating magnitude-sensitive analyses."
        )

    out_float_vars[
        var
    ] = outv

state_var = nc_out.createVariable(
    "EcoState",
    "i1",
    ("time", "y", "x"),
    zlib=True,
    complevel=COMPLEVEL,
    shuffle=True,
    fill_value=np.int8(0),
    chunksizes=(
        min(CHUNK_TIME, nt),
        min(CHUNK_Y, ny),
        min(CHUNK_X, nx),
    )
)

state_var.long_name = (
    "Ecological function state class"
)
state_var.units = "1"
state_var.flag_values = np.asarray(
    sorted(STATE_LABELS.keys()),
    dtype=np.int8
)
state_var.flag_meanings = " ".join(
    STATE_LABELS[k]
    for k in sorted(
        STATE_LABELS
    )
)
state_var.definition = (
    "0=NoData/invalid; 1=stable; 2=early Sync decoupling; "
    "3=water-use impairment; 4=carbon/photosynthetic impairment; "
    "5=severe compound degradation"
)
state_var.damage_threshold = (
    Z_DAMAGE_THRESHOLD
)
state_var.severe_compound_threshold = (
    ACOMP_SEVERE_THRESHOLD
)

print(
    "Float variables:"
)
for var in FLOAT_OUTPUT_VARS:
    print(
        f"  {var}"
    )

print(
    "\nState variable:"
)
print(
    "  EcoState"
)

# =============================================================================
# 6. PROCESS BLOCKS
# =============================================================================

sep("BUILD STATE SPACE IN LOW-MEMORY BLOCKS")

n_blocks = int(
    np.ceil(
        ny / BLOCK_Y
    )
)

global_state_counts = {
    k: 0
    for k in STATE_LABELS
}

for ib, y0 in enumerate(
    range(
        0,
        ny,
        BLOCK_Y
    ),
    start=1
):

    y1 = min(
        y0 + BLOCK_Y,
        ny
    )

    byn = y1 - y0

    print(
        f"Block {ib:02d}/{n_blocks:02d} | "
        f"y={y0}:{y1}"
    )

    zsync = masked_to_nan(
        ds["ZSync"]
        .isel(
            y=slice(
                y0,
                y1
            )
        )
        .values,
        dtype=np.float32
    )

    zuwue = masked_to_nan(
        ds["ZuWUE"]
        .isel(
            y=slice(
                y0,
                y1
            )
        )
        .values,
        dtype=np.float32
    )

    znirv = masked_to_nan(
        ds["ZNIRv"]
        .isel(
            y=slice(
                y0,
                y1
            )
        )
        .values,
        dtype=np.float32
    )

    expected_shape = (
        nt,
        byn,
        nx
    )

    if (
        zsync.shape != expected_shape
        or
        zuwue.shape != expected_shape
        or
        znirv.shape != expected_shape
    ):
        raise RuntimeError(
            "Unexpected block shape in Z variables."
        )

    valid = (
        np.isfinite(
            zsync
        )
        &
        np.isfinite(
            zuwue
        )
        &
        np.isfinite(
            znirv
        )
    )

    zsync_cap = np.clip(
        zsync,
        -Z_CAP_ABS,
        Z_CAP_ABS
    ).astype(
        np.float32
    )

    zuwue_cap = np.clip(
        zuwue,
        -Z_CAP_ABS,
        Z_CAP_ABS
    ).astype(
        np.float32
    )

    znirv_cap = np.clip(
        znirv,
        -Z_CAP_ABS,
        Z_CAP_ABS
    ).astype(
        np.float32
    )

    # Keep invalid cells as NaN in capped fields.
    zsync_cap[
        ~np.isfinite(
            zsync
        )
    ] = np.nan

    zuwue_cap[
        ~np.isfinite(
            zuwue
        )
    ] = np.nan

    znirv_cap[
        ~np.isfinite(
            znirv
        )
    ] = np.nan

    aneg_sync = np.maximum(
        0.0,
        -zsync_cap
    ).astype(
        np.float32
    )

    aneg_uwue = np.maximum(
        0.0,
        -zuwue_cap
    ).astype(
        np.float32
    )

    aneg_nirv = np.maximum(
        0.0,
        -znirv_cap
    ).astype(
        np.float32
    )

    aneg_sync[
        ~valid
    ] = np.nan

    aneg_uwue[
        ~valid
    ] = np.nan

    aneg_nirv[
        ~valid
    ] = np.nan

    acomp = np.sqrt(
        (
            aneg_sync * aneg_sync
            +
            aneg_uwue * aneg_uwue
            +
            aneg_nirv * aneg_nirv
        )
        /
        np.float32(
            3.0
        )
    ).astype(
        np.float32
    )

    acomp[
        ~valid
    ] = np.nan

    below_sync = (
        zsync_cap
        <
        Z_DAMAGE_THRESHOLD
    )

    below_uwue = (
        zuwue_cap
        <
        Z_DAMAGE_THRESHOLD
    )

    below_nirv = (
        znirv_cap
        <
        Z_DAMAGE_THRESHOLD
    )

    n_below = (
        below_sync.astype(
            np.int8
        )
        +
        below_uwue.astype(
            np.int8
        )
        +
        below_nirv.astype(
            np.int8
        )
    )

    stable = (
        valid
        &
        (
            n_below == 0
        )
    )

    severe = (
        valid
        &
        (
            (
                n_below >= 2
            )
            |
            (
                acomp
                >=
                ACOMP_SEVERE_THRESHOLD
            )
        )
    )

    early_decoupling = (
        valid
        &
        ~severe
        &
        below_sync
        &
        ~below_uwue
        &
        ~below_nirv
    )

    water_impairment = (
        valid
        &
        ~severe
        &
        below_uwue
        &
        ~below_nirv
    )

    carbon_impairment = (
        valid
        &
        ~severe
        &
        below_nirv
        &
        (
            acomp
            <
            ACOMP_SEVERE_THRESHOLD
        )
    )

    state = np.zeros(
        expected_shape,
        dtype=np.int8
    )

    state[
        stable
    ] = 1

    state[
        early_decoupling
    ] = 2

    state[
        water_impairment
    ] = 3

    state[
        carbon_impairment
    ] = 4

    state[
        severe
    ] = 5

    for k in STATE_LABELS:
        global_state_counts[
            k
        ] += int(
            (
                state == k
            ).sum()
        )

    # Write original Z variables unchanged.
    out_float_vars["ZSync"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        zsync
    )

    out_float_vars["ZuWUE"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        zuwue
    )

    out_float_vars["ZNIRv"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        znirv
    )

    out_float_vars["ZSync_cap"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        zsync_cap
    )

    out_float_vars["ZuWUE_cap"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        zuwue_cap
    )

    out_float_vars["ZNIRv_cap"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        znirv_cap
    )

    out_float_vars["Aneg_Sync"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        aneg_sync
    )

    out_float_vars["Aneg_uWUE"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        aneg_uwue
    )

    out_float_vars["Aneg_NIRv"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        aneg_nirv
    )

    out_float_vars["A_comp"][
        :,
        y0:y1,
        :
    ] = np.ma.masked_invalid(
        acomp
    )

    state_var[
        :,
        y0:y1,
        :
    ] = state

    (
        amin,
        amax,
        amean,
        anod
    ) = safe_nan_stats(
        acomp
    )

    valid_pct = (
        100.0
        *
        int(valid.sum())
        /
        valid.size
    )

    print(
        f"  valid all-3D = {valid_pct:.3f}% | "
        f"A_comp min={amin:.4f}, "
        f"max={amax:.4f}, "
        f"mean={amean:.4f}, "
        f"NoData={anod:.3f}%"
    )

    del zsync
    del zuwue
    del znirv
    del valid
    del zsync_cap
    del zuwue_cap
    del znirv_cap
    del aneg_sync
    del aneg_uwue
    del aneg_nirv
    del acomp
    del below_sync
    del below_uwue
    del below_nirv
    del n_below
    del stable
    del severe
    del early_decoupling
    del water_impairment
    del carbon_impairment
    del state

    gc.collect()

    nc_out.sync()

# =============================================================================
# 7. CLOSE WRITERS
# =============================================================================

sep("FINALIZE NETCDF WRITING")

nc_out.sync()
nc_out.close()
ds.close()

print("WRITE COMPLETE")
print(OUT_STATE_NC)

# =============================================================================
# 8. POST-WRITE STRUCTURAL QC
# =============================================================================

sep("POST-WRITE STRUCTURAL QC")

q = xr.open_dataset(
    OUT_STATE_NC,
    decode_times=True,
    cache=False
)

print("Dimensions:")
print(dict(q.sizes))

dims_ok = (
    q.sizes.get("time") == nt
    and
    q.sizes.get("y") == ny
    and
    q.sizes.get("x") == nx
)

time_ok = np.array_equal(
    pd.DatetimeIndex(
        q.time.values
    ).values,
    time_index.values
)

x_ok = np.array_equal(
    np.asarray(
        q.x.values
    ),
    x
)

y_ok = np.array_equal(
    np.asarray(
        q.y.values
    ),
    y
)

phase_ok = np.array_equal(
    np.asarray(
        q["phase8"].values,
        dtype=np.int8
    ),
    phase8
)

vars_ok = all(
    var in q.data_vars
    for var in (
        FLOAT_OUTPUT_VARS
        +
        [
            "EcoState"
        ]
    )
)

print(
    f"\nDimensions PASS : {dims_ok}"
)
print(
    f"Time PASS       : {time_ok}"
)
print(
    f"X labels PASS   : {x_ok}"
)
print(
    f"Y labels PASS   : {y_ok}"
)
print(
    f"phase8 PASS     : {phase_ok}"
)
print(
    f"Variables PASS  : {vars_ok}"
)

q.close()

# =============================================================================
# 9. LOW-MEMORY VARIABLE QC
# =============================================================================

sep("LOW-MEMORY VARIABLE QC")

float_qc = {}

for var in FLOAT_OUTPUT_VARS:

    stats = block_scan_float_var(
        OUT_STATE_NC,
        var
    )

    float_qc[
        var
    ] = stats

    print(
        f"{var:10s}: "
        f"min={stats['min']:.6f} | "
        f"max={stats['max']:.6f} | "
        f"mean={stats['mean']:.6f} | "
        f"NoData={stats['nodata_pct']:.3f}% | "
        f"|x|>5={stats['abs_gt5']:,} | "
        f"|x|>10={stats['abs_gt10']:,}"
    )

finite_ok = all(
    np.isfinite(
        stats["min"]
    )
    and
    np.isfinite(
        stats["max"]
    )
    and
    np.isfinite(
        stats["mean"]
    )
    for stats in float_qc.values()
)

cap_range_ok = all(
    (
        float_qc[var]["min"]
        >=
        -Z_CAP_ABS
        -
        1.0e-5
    )
    and
    (
        float_qc[var]["max"]
        <=
        Z_CAP_ABS
        +
        1.0e-5
    )
    for var in CAP_VARS
)

aneg_range_ok = all(
    float_qc[var]["min"] >= -1.0e-6
    for var in ANEG_VARS + ["A_comp"]
)

# =============================================================================
# 10. STATE COUNTS
# =============================================================================

sep("ECOSTATE COUNTS")

state_df = block_scan_state_counts(
    OUT_STATE_NC
)

print(
    state_df.to_string(
        index=False
    )
)

state_count_sum_ok = (
    int(
        state_df[
            "count"
        ].sum()
    )
    ==
    nt * ny * nx
)

# Check whether every valid finite A_comp pixel has a nonzero state.
# This should be true by construction.
with Dataset(
    OUT_STATE_NC,
    "r"
) as nc:

    acomp_var = nc.variables[
        "A_comp"
    ]

    state_nc = nc.variables[
        "EcoState"
    ]

    invalid_state_on_valid = 0
    valid_acomp_total = 0

    for y0 in range(
        0,
        ny,
        BLOCK_Y
    ):

        y1 = min(
            y0 + BLOCK_Y,
            ny
        )

        acomp = masked_to_nan(
            acomp_var[
                :,
                y0:y1,
                :
            ],
            dtype=np.float32
        )

        state = np.asarray(
            state_nc[
                :,
                y0:y1,
                :
            ],
            dtype=np.int8
        )

        valid_acomp = np.isfinite(
            acomp
        )

        valid_acomp_total += int(
            valid_acomp.sum()
        )

        invalid_state_on_valid += int(
            (
                valid_acomp
                &
                (
                    state == 0
                )
            ).sum()
        )

        del acomp
        del state

valid_state_ok = (
    invalid_state_on_valid == 0
)

print(
    f"\nState count sum PASS       : {state_count_sum_ok}"
)
print(
    f"Valid A_comp state PASS    : {valid_state_ok}"
)
print(
    f"Valid A_comp cells         : {valid_acomp_total:,}"
)
print(
    f"Valid cells with state=0   : {invalid_state_on_valid:,}"
)

# =============================================================================
# 11. REPRESENTATIVE TIME-SLICE QC
# =============================================================================

sep("REPRESENTATIVE TIME-SLICE QC")

q = xr.open_dataset(
    OUT_STATE_NC,
    decode_times=True,
    cache=False
)

indices = [
    0,
    nt // 2,
    nt - 1
]

for idx in indices:

    date = pd.Timestamp(
        q.time.values[
            idx
        ]
    ).date()

    phase = int(
        q["phase8"]
        .isel(
            time=idx
        )
        .values
    )

    print(
        f"\nTime[{idx}] = {date} | phase8={phase}"
    )

    for var in [
        "ZSync",
        "ZuWUE",
        "ZNIRv",
        "A_comp"
    ]:

        arr = np.asarray(
            q[var]
            .isel(
                time=idx
            )
            .values,
            dtype=np.float32
        )

        (
            vmin,
            vmax,
            mean,
            nodata
        ) = safe_nan_stats(
            arr
        )

        print(
            f"  {var:10s} "
            f"min={vmin:11.5f} "
            f"max={vmax:11.5f} "
            f"mean={mean:11.5f} "
            f"NoData={nodata:7.3f}%"
        )

    state_arr = np.asarray(
        q["EcoState"]
        .isel(
            time=idx
        )
        .values,
        dtype=np.int8
    )

    print(
        "  EcoState counts:"
    )

    for k in sorted(
        STATE_LABELS
    ):

        count_k = int(
            (
                state_arr == k
            ).sum()
        )

        pct_k = (
            100.0
            *
            count_k
            /
            state_arr.size
        )

        print(
            f"    S{k} "
            f"{STATE_LABELS[k]:35s} "
            f"{count_k:9d} "
            f"({pct_k:7.3f}%)"
        )

q.close()

# =============================================================================
# 12. FINAL STATUS
# =============================================================================

sep("FINAL RESULT")

final_ok = all(
    [
        dims_ok,
        time_ok,
        x_ok,
        y_ok,
        phase_ok,
        vars_ok,
        finite_ok,
        cap_range_ok,
        aneg_range_ok,
        state_count_sum_ok,
        valid_state_ok,
    ]
)

if final_ok:

    print(
        "PASS: Step L4.3 EcoFunction State Space dataset "
        "constructed successfully."
    )

else:

    print(
        "WARNING: Step L4.3 completed, but one or more QC "
        "conditions failed. Review failed QC before continuing."
    )

print(
    f"Dimensions PASS        : {dims_ok}"
)
print(
    f"Time PASS              : {time_ok}"
)
print(
    f"X labels PASS          : {x_ok}"
)
print(
    f"Y labels PASS          : {y_ok}"
)
print(
    f"phase8 PASS            : {phase_ok}"
)
print(
    f"Variables PASS         : {vars_ok}"
)
print(
    f"Finite float QC PASS   : {finite_ok}"
)
print(
    f"Capped range PASS      : {cap_range_ok}"
)
print(
    f"Anomaly range PASS     : {aneg_range_ok}"
)
print(
    f"State count sum PASS   : {state_count_sum_ok}"
)
print(
    f"Valid state PASS       : {valid_state_ok}"
)

print("\nFINAL STATE-SPACE FILE:")
print(OUT_STATE_NC)

print("\nStep L4.3 completed.")
