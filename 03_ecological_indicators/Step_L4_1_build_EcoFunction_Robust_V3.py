# -*- coding: utf-8 -*-

"""
Step L4.1 - Build Weihe EcoFunction Cube

Purpose
-------
Build the raw ecological functional cube for the Weihe River Basin.

Inputs
------
1. Old ecological cube:
   Weihe_Cube_1km_8Day_formal.nc

   Reused variables:
       GPP
       ET
       NIRv

2. Validated L3 climate cube:
   Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc

   Reused variable:
       VPD

Outputs
-------
Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc

Variables:
    GPP
    ET
    NIRv
    VPD
    uWUE
    Sync

Robust V3 safeguards
--------------------
1. Rebinds OLD ECO GPP/ET/NIRv to the exact validated L3 master coordinate labels
   after numerical time-space equivalence is confirmed. No interpolation,
   reprojection, or resampling is performed.
2. Calculates 7-phase GPP-ET Synchrony in spatial blocks using cumulative
   sufficient statistics and a disk-backed memmap; no 4-D rolling window is built.
3. Performs strict NoData conservation checks after Dataset assembly and after
   NetCDF writing; any unexpected valid-cell loss aborts or fails the build.

Important
---------
This step DOES NOT calculate:
    ZSync
    ZuWUE
    ZNIRv

Those will be constructed separately in Step L4.2 using
a phenological baseline.
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS
# =============================================================================

OLD_ECO_NC = (
    r".\Weihe database V2"
    r"\Weihe_Processed"
    r"\Weihe_Cube_1km_8Day_formal.nc"
)

L3_NC = (
    r".\Weihe database V3"
    r"\Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
)

OUT_DIR = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_NC = os.path.join(
    OUT_DIR,
    "Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc"
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

# -------------------------------------------------------------------------
# uWUE validity thresholds
# -------------------------------------------------------------------------

# ET values <= threshold are not allowed in denominator.
# This avoids uWUE explosion when ET approaches zero.
ET_MIN = 0.05

# GPP <= threshold treated as inactive / unreliable carbon uptake.
GPP_MIN = 0.01

# VPD must be positive.
VPD_MIN = 0.001


# -------------------------------------------------------------------------
# Synchrony rolling window
# -------------------------------------------------------------------------

# Previous methodological design:
# 7 x 8-day steps ≈ 56 days
SYNC_WINDOW = 7

# Require at least this many valid paired observations in a window
SYNC_MIN_PERIODS = 5

# Spatial block size used by the memory-safe Synchrony calculation.
# 24 rows keeps the working set well below the previous 4-D rolling-window peak.
SYNC_BLOCK_Y = 24


# -------------------------------------------------------------------------
# Plausibility limits
# -------------------------------------------------------------------------

# These are safety filters, not normalization thresholds.

UWUE_MIN = 0.0

# Deliberately loose upper threshold.
# Only extreme denominator-driven explosions are removed.
UWUE_MAX = 50.0

NIRV_MIN = -0.10
NIRV_MAX = 1.00


# -------------------------------------------------------------------------
# NetCDF compression
# -------------------------------------------------------------------------

COMPLEVEL = 4

CHUNK_TIME = 16
CHUNK_Y = 207
CHUNK_X = 352

# Maximum permitted change in NoData percentage points when variables are
# assembled onto the L3 master grid or re-opened from the written NetCDF.
# Exact coordinate rebinding should make this effectively zero.
NODATA_CONSERVATION_TOL = 1e-4


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def sep(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def exact_coord_equal(a, b, atol=1e-10):
    if a.shape != b.shape:
        return False

    return np.allclose(
        np.asarray(a),
        np.asarray(b),
        rtol=0,
        atol=atol,
        equal_nan=True
    )


def stat_string(da):
    try:
        vmin = float(
            da.min(skipna=True).values
        )

        vmax = float(
            da.max(skipna=True).values
        )

        mean = float(
            da.mean(skipna=True).values
        )

        total = int(
            np.prod(
                da.shape
            )
        )

        valid = int(
            da.count().values
        )

        nodata = (
            (total - valid)
            / total
            * 100.0
        )

        return (
            f"min={vmin:.6f} | "
            f"max={vmax:.6f} | "
            f"mean={mean:.6f} | "
            f"NoData={nodata:.3f}%"
        )

    except Exception as exc:

        return (
            f"STAT FAILED: {repr(exc)}"
        )


def nodata_percent(da):
    """Return percentage of NaN/invalid cells in a DataArray."""
    total = int(np.prod(da.shape))
    valid = int(da.count().values)
    return (total - valid) / total * 100.0


def bind_to_master_grid(da, master_ds, name):
    """
    Rebind an already-validated DataArray to the exact L3 master coordinate
    labels WITHOUT interpolation, reprojection, resampling, or reindexing.

    Spatial/temporal equivalence must have been checked before this function
    is called.  This step only replaces nearly-identical floating coordinate
    labels with the canonical L3 labels so xarray cannot silently align them
    as different labels later.
    """
    required = ("time", "y", "x")

    if set(da.dims) != set(required):
        raise RuntimeError(
            f"{name} dimensions are {da.dims}; expected time/y/x."
        )

    da = da.transpose(*required)

    return da.assign_coords(
        time=master_ds["time"].values,
        y=master_ds["y"].values,
        x=master_ds["x"].values
    )


# =============================================================================
# 3. OPEN DATASETS
# =============================================================================

sep("OPEN INPUT DATASETS")

require_file(
    OLD_ECO_NC,
    "Old ecological NetCDF"
)

require_file(
    L3_NC,
    "Validated L3 NetCDF"
)

print("OLD ECO:")
print(OLD_ECO_NC)

print("\nL3:")
print(L3_NC)

ds_eco = xr.open_dataset(
    OLD_ECO_NC,
    decode_times=True
)

ds_l3 = xr.open_dataset(
    L3_NC,
    decode_times=True
)

print("\nOLD ECO dimensions:")
print(dict(ds_eco.sizes))

print("\nL3 dimensions:")
print(dict(ds_l3.sizes))


# =============================================================================
# 4. REQUIRED VARIABLE CHECK
# =============================================================================

sep("REQUIRED VARIABLE CHECK")

eco_required = [
    "GPP",
    "ET",
    "NIRv"
]

l3_required = [
    "VPD"
]

for var in eco_required:

    if var not in ds_eco.data_vars:

        raise KeyError(
            f"Required ecological variable missing: {var}"
        )

    print(
        f"{var:10s}: FOUND in OLD ECO"
    )

for var in l3_required:

    if var not in ds_l3.data_vars:

        raise KeyError(
            f"Required L3 variable missing: {var}"
        )

    print(
        f"{var:10s}: FOUND in L3"
    )


# =============================================================================
# 5. EXACT TIME-SPACE ALIGNMENT CHECK
# =============================================================================

sep("EXACT TIME-SPACE ALIGNMENT CHECK")

for dim in [
    "time",
    "y",
    "x"
]:

    n1 = ds_eco.sizes.get(
        dim
    )

    n2 = ds_l3.sizes.get(
        dim
    )

    status = (
        n1 == n2
    )

    print(
        f"{dim:6s}: "
        f"ECO={n1} | "
        f"L3={n2} | "
        f"{'PASS' if status else 'FAIL'}"
    )

    if not status:

        raise RuntimeError(
            f"Dimension mismatch: {dim}"
        )


time_eco = pd.DatetimeIndex(
    pd.to_datetime(
        ds_eco.time.values
    )
)

time_l3 = pd.DatetimeIndex(
    pd.to_datetime(
        ds_l3.time.values
    )
)

time_equal = np.array_equal(
    time_eco.values,
    time_l3.values
)

x_equal = exact_coord_equal(
    ds_eco.x.values,
    ds_l3.x.values
)

y_equal = exact_coord_equal(
    ds_eco.y.values,
    ds_l3.y.values
)

print(
    "\nTime:",
    "PASS"
    if time_equal
    else "FAIL"
)

print(
    "X:",
    "PASS"
    if x_equal
    else "FAIL"
)

print(
    "Y:",
    "PASS"
    if y_equal
    else "FAIL"
)

if not (
    time_equal
    and x_equal
    and y_equal
):

    raise RuntimeError(
        "OLD ECO and L3 are not numerically aligned."
    )

# Distinguish numerical equivalence from exact coordinate-label identity.
x_label_exact = np.array_equal(ds_eco.x.values, ds_l3.x.values)
y_label_exact = np.array_equal(ds_eco.y.values, ds_l3.y.values)

x_max_abs_diff = float(np.nanmax(np.abs(ds_eco.x.values - ds_l3.x.values)))
y_max_abs_diff = float(np.nanmax(np.abs(ds_eco.y.values - ds_l3.y.values)))

print("\nExact coordinate LABEL identity before rebinding:")
print(f"X labels exact : {x_label_exact} | max abs diff={x_max_abs_diff:.16g}")
print(f"Y labels exact : {y_label_exact} | max abs diff={y_max_abs_diff:.16g}")

print(
    "\nPASS: numerical time-space alignment confirmed."
)
print(
    "OLD ECO ecological variables will now be rebound to the canonical L3 master labels "
    "without interpolation or resampling."
)


# =============================================================================
# 6. LOAD CORE VARIABLES AND BIND TO L3 MASTER GRID
# =============================================================================

sep("LOAD CORE ECOFUNCTION VARIABLES")

# OLD ECO and L3 have already passed the numerical time-space equivalence
# check.  Explicitly replace OLD ECO coordinate labels with the canonical L3
# labels.  This is NOT interpolation, resampling, or reprojection.
GPP = bind_to_master_grid(
    ds_eco["GPP"].astype("float32"),
    ds_l3,
    "GPP"
)

ET = bind_to_master_grid(
    ds_eco["ET"].astype("float32"),
    ds_l3,
    "ET"
)

NIRv = bind_to_master_grid(
    ds_eco["NIRv"].astype("float32"),
    ds_l3,
    "NIRv"
)

VPD = (
    ds_l3["VPD"]
    .astype("float32")
    .transpose("time", "y", "x")
)

# Hard verification: all working variables must now use EXACTLY the L3 labels.
for _name, _da in {
    "GPP": GPP,
    "ET": ET,
    "NIRv": NIRv,
    "VPD": VPD,
}.items():
    if not np.array_equal(_da.time.values, ds_l3.time.values):
        raise RuntimeError(f"{_name}: time labels are not exactly L3 master labels.")
    if not np.array_equal(_da.y.values, ds_l3.y.values):
        raise RuntimeError(f"{_name}: y labels are not exactly L3 master labels.")
    if not np.array_equal(_da.x.values, ds_l3.x.values):
        raise RuntimeError(f"{_name}: x labels are not exactly L3 master labels.")

print("PASS: GPP/ET/NIRv rebound to exact L3 master coordinate labels.")

print("GPP :", stat_string(GPP))
print("ET  :", stat_string(ET))
print("NIRv:", stat_string(NIRv))
print("VPD :", stat_string(VPD))


# =============================================================================
# 7. BASIC PHYSICAL QC MASKS
# =============================================================================

sep("BUILD VALIDITY MASKS")

# -------------------------------------------------------------------------
# GPP
# -------------------------------------------------------------------------

gpp_valid = (
    np.isfinite(GPP)
    &
    (GPP >= GPP_MIN)
)

# -------------------------------------------------------------------------
# ET
# -------------------------------------------------------------------------

et_valid = (
    np.isfinite(ET)
    &
    (ET >= ET_MIN)
)

# -------------------------------------------------------------------------
# VPD
# -------------------------------------------------------------------------

vpd_valid = (
    np.isfinite(VPD)
    &
    (VPD >= VPD_MIN)
)

# -------------------------------------------------------------------------
# NIRv
# -------------------------------------------------------------------------

nirv_valid = (
    np.isfinite(NIRv)
    &
    (NIRv >= NIRV_MIN)
    &
    (NIRv <= NIRV_MAX)
)

print(
    "Validity masks created."
)


# =============================================================================
# 8. CLEAN CORE VARIABLES
# =============================================================================

sep("CLEAN CORE VARIABLES")

GPP_clean = GPP.where(
    np.isfinite(GPP)
)

ET_clean = ET.where(
    np.isfinite(ET)
)

VPD_clean = VPD.where(
    np.isfinite(VPD)
)

NIRv_clean = NIRv.where(
    nirv_valid
)

GPP_clean.name = "GPP"
ET_clean.name = "ET"
VPD_clean.name = "VPD"
NIRv_clean.name = "NIRv"


# =============================================================================
# 9. CALCULATE uWUE
# =============================================================================

sep("CALCULATE uWUE")

# -------------------------------------------------------------------------
# Definition:
#
# uWUE = GPP * sqrt(VPD) / ET
#
# Only calculate where:
#     GPP >= GPP_MIN
#     ET  >= ET_MIN
#     VPD >= VPD_MIN
# -------------------------------------------------------------------------

uwue_valid = (
    gpp_valid
    &
    et_valid
    &
    vpd_valid
)

uWUE = (
    GPP
    *
    np.sqrt(
        VPD
    )
    /
    ET
)

uWUE = uWUE.where(
    uwue_valid
)

# Additional safety removal
uWUE = uWUE.where(
    (uWUE >= UWUE_MIN)
    &
    (uWUE <= UWUE_MAX)
)

uWUE = uWUE.astype(
    "float32"
)

uWUE.name = "uWUE"

print(
    stat_string(uWUE)
)


# =============================================================================
# 10. CALCULATE SYNCHRONY - MEMORY-SAFE ROBUST V3
# =============================================================================

sep("CALCULATE GPP-ET SYNCHRONY")

print(
    f"Rolling window = {SYNC_WINDOW} phases "
    f"(~{SYNC_WINDOW * 8} days)"
)

print(
    f"Minimum valid paired observations = "
    f"{SYNC_MIN_PERIODS}"
)

print(
    f"Spatial processing block = "
    f"{SYNC_BLOCK_Y} y-rows"
)

# -------------------------------------------------------------------------
# Methodological definition is unchanged:
#
#     Sync = Pearson r(GPP_window, ET_window)
#
# using a centered 7-phase (~56-day) window.
#
# The previous version used:
#
#     rolling(...).construct("window")
#
# which explicitly created a 4-D array:
#
#     (time, y, x, window)
#
# and caused an ~8.67 GiB allocation failure.
#
# Robust V3 instead:
#   1. reads a small spatial block (all time, limited y rows);
#   2. calculates rolling sufficient statistics using cumulative sums;
#   3. writes the resulting Sync block directly to a disk-backed memmap.
#
# Thus, no full 4-D rolling-window array is ever constructed.
# -------------------------------------------------------------------------


def centered_window_bounds(n_time, window):
    """
    Return start/end indices for an odd centered rolling window.

    For window=7, each central time step uses:
        t-3 ... t+3

    Near the beginning/end of the series, the window is truncated exactly
    as xarray rolling(center=True, min_periods=1) would do.
    """
    if window < 1:
        raise ValueError("SYNC_WINDOW must be >= 1.")

    if window % 2 == 0:
        raise ValueError(
            "This implementation expects an odd SYNC_WINDOW "
            "for an unambiguous centered window."
        )

    half = window // 2

    t = np.arange(
        n_time,
        dtype=np.int64
    )

    start_idx = np.maximum(
        0,
        t - half
    )

    end_idx = np.minimum(
        n_time,
        t + half + 1
    )

    return start_idx, end_idx


def cumulative_with_zero(arr, dtype=np.float32):
    """
    Cumulative sum along time with a leading zero plane.

    Input:
        arr.shape = (time, y_block, x)

    Output:
        out.shape = (time + 1, y_block, x)
    """
    out = np.empty(
        (arr.shape[0] + 1,) + arr.shape[1:],
        dtype=dtype
    )

    out[0] = 0

    np.cumsum(
        arr,
        axis=0,
        dtype=dtype,
        out=out[1:]
    )

    return out


def rolling_sum_from_cumulative(csum, start_idx, end_idx):
    """
    Window sum from cumulative sums.

    For every time t:
        sum[t] = csum[end_idx[t]] - csum[start_idx[t]]
    """
    return (
        csum[end_idx]
        -
        csum[start_idx]
    )


def calculate_sync_block(
    gpp_block,
    et_block,
    start_idx,
    end_idx,
    min_periods
):
    """
    Calculate centered rolling Pearson r for one spatial block.

    Parameters
    ----------
    gpp_block, et_block : np.ndarray
        Shape = (time, y_block, x), float32.

    start_idx, end_idx : np.ndarray
        Rolling-window boundaries.

    min_periods : int
        Minimum number of finite paired observations.

    Returns
    -------
    sync_block : np.ndarray
        Shape = (time, y_block, x), float32.
    """

    # ---------------------------------------------------------------------
    # Pairwise validity exactly follows the original script:
    # only finite GPP + finite ET are required for Synchrony.
    #
    # GPP_MIN / ET_MIN thresholds are used for uWUE, not for Sync.
    # ---------------------------------------------------------------------

    pair = (
        np.isfinite(gpp_block)
        &
        np.isfinite(et_block)
    )

    # Use zero-filled paired arrays for cumulative sufficient statistics.
    x = np.where(
        pair,
        gpp_block,
        0.0
    ).astype(
        np.float32,
        copy=False
    )

    y = np.where(
        pair,
        et_block,
        0.0
    ).astype(
        np.float32,
        copy=False
    )

    # Valid-pair count.
    pair_i16 = pair.astype(
        np.int16,
        copy=False
    )

    c_n = cumulative_with_zero(
        pair_i16,
        dtype=np.int32
    )

    c_x = cumulative_with_zero(
        x,
        dtype=np.float32
    )

    c_y = cumulative_with_zero(
        y,
        dtype=np.float32
    )

    # Compute quadratic/product arrays one at a time.
    x2 = (
        x * x
    ).astype(
        np.float32,
        copy=False
    )

    c_x2 = cumulative_with_zero(
        x2,
        dtype=np.float32
    )

    del x2

    y2 = (
        y * y
    ).astype(
        np.float32,
        copy=False
    )

    c_y2 = cumulative_with_zero(
        y2,
        dtype=np.float32
    )

    del y2

    xy = (
        x * y
    ).astype(
        np.float32,
        copy=False
    )

    c_xy = cumulative_with_zero(
        xy,
        dtype=np.float32
    )

    del xy

    # ---------------------------------------------------------------------
    # Rolling sufficient statistics
    # ---------------------------------------------------------------------

    n = rolling_sum_from_cumulative(
        c_n,
        start_idx,
        end_idx
    ).astype(
        np.float32,
        copy=False
    )

    sx = rolling_sum_from_cumulative(
        c_x,
        start_idx,
        end_idx
    )

    sy = rolling_sum_from_cumulative(
        c_y,
        start_idx,
        end_idx
    )

    sxx = rolling_sum_from_cumulative(
        c_x2,
        start_idx,
        end_idx
    )

    syy = rolling_sum_from_cumulative(
        c_y2,
        start_idx,
        end_idx
    )

    sxy = rolling_sum_from_cumulative(
        c_xy,
        start_idx,
        end_idx
    )

    # Cumulative arrays are no longer needed.
    del c_n
    del c_x
    del c_y
    del c_x2
    del c_y2
    del c_xy

    # ---------------------------------------------------------------------
    # Pearson r from sufficient statistics:
    #
    #            n*Sxy - Sx*Sy
    # r = ---------------------------------
    #     sqrt[(n*Sxx-Sx²)(n*Syy-Sy²)]
    # ---------------------------------------------------------------------

    numerator = (
        n * sxy
        -
        sx * sy
    )

    var_x_term = (
        n * sxx
        -
        sx * sx
    )

    var_y_term = (
        n * syy
        -
        sy * sy
    )

    # Floating-point roundoff may create tiny negative values.
    np.maximum(
        var_x_term,
        0.0,
        out=var_x_term
    )

    np.maximum(
        var_y_term,
        0.0,
        out=var_y_term
    )

    denominator = np.sqrt(
        var_x_term
        *
        var_y_term
    ).astype(
        np.float32,
        copy=False
    )

    # Initialize all cells as NaN.
    sync_block = np.full(
        gpp_block.shape,
        np.nan,
        dtype=np.float32
    )

    valid_corr = (
        (n >= min_periods)
        &
        np.isfinite(denominator)
        &
        (denominator > 1.0e-12)
    )

    np.divide(
        numerator,
        denominator,
        out=sync_block,
        where=valid_corr
    )

    # Numerical safety.
    np.clip(
        sync_block,
        -1.0,
        1.0,
        out=sync_block
    )

    return sync_block


n_time = ds_eco.sizes["time"]
n_y = ds_eco.sizes["y"]
n_x = ds_eco.sizes["x"]

window_start_idx, window_end_idx = centered_window_bounds(
    n_time,
    SYNC_WINDOW
)

# -------------------------------------------------------------------------
# Disk-backed float32 array.
#
# This prevents a full ~1.2 GiB Sync array from having to live entirely in
# RAM while blocks are being calculated.
# -------------------------------------------------------------------------

SYNC_TMP = os.path.join(
    OUT_DIR,
    "_TEMP_Sync_RobustV3_float32.dat"
)

if os.path.exists(SYNC_TMP):
    print(
        "Removing existing temporary Sync memmap:"
    )
    print(
        SYNC_TMP
    )
    os.remove(
        SYNC_TMP
    )

Sync_memmap = np.memmap(
    SYNC_TMP,
    mode="w+",
    dtype="float32",
    shape=(
        n_time,
        n_y,
        n_x
    )
)

Sync_memmap[:] = np.nan

n_blocks = int(
    np.ceil(
        n_y / SYNC_BLOCK_Y
    )
)

for block_id, y0 in enumerate(
    range(
        0,
        n_y,
        SYNC_BLOCK_Y
    ),
    start=1
):

    y1 = min(
        y0 + SYNC_BLOCK_Y,
        n_y
    )

    print(
        f"Sync block {block_id:02d}/{n_blocks:02d} | "
        f"y={y0}:{y1}"
    )

    # Read only this spatial block from the NetCDF files.
    gpp_block = np.asarray(
        ds_eco["GPP"]
        .isel(
            y=slice(
                y0,
                y1
            )
        )
        .values,
        dtype=np.float32
    )

    et_block = np.asarray(
        ds_eco["ET"]
        .isel(
            y=slice(
                y0,
                y1
            )
        )
        .values,
        dtype=np.float32
    )

    sync_block = calculate_sync_block(
        gpp_block,
        et_block,
        window_start_idx,
        window_end_idx,
        SYNC_MIN_PERIODS
    )

    Sync_memmap[
        :,
        y0:y1,
        :
    ] = sync_block

    Sync_memmap.flush()

    # Block-level QC
    finite_sync = np.isfinite(
        sync_block
    )

    if np.any(
        finite_sync
    ):

        block_min = float(
            np.nanmin(
                sync_block
            )
        )

        block_max = float(
            np.nanmax(
                sync_block
            )
        )

        block_mean = float(
            np.nanmean(
                sync_block
            )
        )

        block_nodata = float(
            (
                1.0
                -
                finite_sync.mean()
            )
            *
            100.0
        )

        print(
            f"  min={block_min:.6f} | "
            f"max={block_max:.6f} | "
            f"mean={block_mean:.6f} | "
            f"NoData={block_nodata:.3f}%"
        )

    else:

        print(
            "  WARNING: entire block is NoData."
        )

    del gpp_block
    del et_block
    del sync_block
    del finite_sync

    gc.collect()


Sync_memmap.flush()

Sync = xr.DataArray(
    Sync_memmap,
    dims=(
        "time",
        "y",
        "x"
    ),
    coords={
        "time": ds_l3.time,
        "y": ds_l3.y,
        "x": ds_l3.x
    },
    name="Sync"
)

print(
    "\nSynchrony calculation complete."
)

print(
    stat_string(
        Sync
    )
)


# =============================================================================
# 11. FREE TEMPORARY SYNCHRONY WORK ARRAYS
# =============================================================================

sep("CLEAR SYNCHRONY WORK ARRAYS")

del window_start_idx
del window_end_idx

gc.collect()

print(
    "Block-level Synchrony work arrays cleared."
)

print(
    "Sync itself remains disk-backed until NetCDF writing is complete."
)

# Record the authoritative pre-assembly NoData percentages.
reference_nodata = {
    "GPP": nodata_percent(GPP_clean),
    "ET": nodata_percent(ET_clean),
    "NIRv": nodata_percent(NIRv_clean),
    "VPD": nodata_percent(VPD_clean),
    "uWUE": nodata_percent(uWUE),
    "Sync": nodata_percent(Sync),
}


# =============================================================================
# 12. BUILD OUTPUT DATASET
# =============================================================================

sep("BUILD L4 DATASET")

ds_out = xr.Dataset(
    data_vars={

        "GPP": GPP_clean,

        "ET": ET_clean,

        "NIRv": NIRv_clean,

        "VPD": VPD_clean,

        "uWUE": uWUE,

        "Sync": Sync,
    },

    coords={

        "time": ds_l3.time,

        "y": ds_l3.y,

        "x": ds_l3.x,
    }
)

# -------------------------------------------------------------------------
# CRITICAL QC: xarray assembly must not change the valid-cell population.
# -------------------------------------------------------------------------
sep("DATA CONSERVATION CHECK AFTER DATASET ASSEMBLY")

assembly_nodata_ok = True

for var in ["GPP", "ET", "NIRv", "VPD", "uWUE", "Sync"]:
    before = reference_nodata[var]
    after = nodata_percent(ds_out[var])
    delta = after - before
    ok = abs(delta) <= NODATA_CONSERVATION_TOL
    assembly_nodata_ok = assembly_nodata_ok and ok

    print(
        f"{var:8s}: before={before:9.4f}% | "
        f"after={after:9.4f}% | delta={delta:+.6f} pp | "
        f"{'PASS' if ok else 'FAIL'}"
    )

if not assembly_nodata_ok:
    raise RuntimeError(
        "Dataset assembly changed NoData coverage. "
        "Aborting before NetCDF writing to prevent propagation of a coordinate-alignment error."
    )

print("PASS: no valid ecological cells were lost during Dataset assembly.")


# =============================================================================
# 13. COPY CRS
# =============================================================================

if "crs" in ds_l3.variables:

    ds_out["crs"] = ds_l3["crs"].copy()


# =============================================================================
# 14. VARIABLE ATTRIBUTES
# =============================================================================

ds_out["GPP"].attrs = {
    "long_name":
        "Gross Primary Productivity",
    "source":
        "Weihe_Cube_1km_8Day_formal.nc",
    "processing":
        "Inherited from validated time-space aligned ecological cube; coordinate labels rebound to canonical L3 master grid without resampling"
}

ds_out["ET"].attrs = {
    "long_name":
        "Evapotranspiration",
    "source":
        "Weihe_Cube_1km_8Day_formal.nc",
    "processing":
        "Inherited from validated time-space aligned ecological cube; coordinate labels rebound to canonical L3 master grid without resampling"
}

ds_out["NIRv"].attrs = {
    "long_name":
        "Near-Infrared Reflectance of Vegetation",
    "source":
        "Weihe_Cube_1km_8Day_formal.nc",
    "valid_range":
        f"{NIRV_MIN} to {NIRV_MAX}"
}

ds_out["VPD"].attrs = {
    "long_name":
        "Vapor Pressure Deficit",
    "units":
        "kPa",
    "source":
        "Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
}

ds_out["uWUE"].attrs = {
    "long_name":
        "Underlying Water Use Efficiency",
    "formula":
        "GPP * sqrt(VPD) / ET",
    "GPP_min_threshold":
        GPP_MIN,
    "ET_min_threshold":
        ET_MIN,
    "VPD_min_threshold":
        VPD_MIN,
    "valid_range":
        f"{UWUE_MIN} to {UWUE_MAX}",
    "comment":
        "Calculated only where GPP, ET and VPD satisfy validity thresholds."
}

ds_out["Sync"].attrs = {
    "long_name":
        "Rolling GPP-ET Synchrony",
    "definition":
        "Pearson correlation between GPP and ET within a centered rolling window",
    "rolling_window_phases":
        SYNC_WINDOW,
    "approximate_window_days":
        SYNC_WINDOW * 8,
    "minimum_valid_pairs":
        SYNC_MIN_PERIODS,
    "valid_range":
        "-1 to 1"
}


# =============================================================================
# 15. GLOBAL ATTRIBUTES
# =============================================================================

ds_out.attrs = {

    "Conventions":
        "CF-1.8",

    "title":
        "Weihe River Basin L4 Raw Ecological Functional Dataset",

    "processing_level":
        "L4.1 RAW EcoFunction",

    "dataset_version":
        "V3.1 Robust V3",

    "master_grid_policy":
        "All ecological variables use exact L3 time/y/x coordinate labels after numerical equivalence validation; no interpolation or resampling applied",

    "time_coverage_start":
        str(
            pd.Timestamp(
                ds_out.time.values[0]
            ).date()
        ),

    "time_coverage_end":
        str(
            pd.Timestamp(
                ds_out.time.values[-1]
            ).date()
        ),

    "temporal_resolution":
        "8-day master phases",

    "spatial_resolution":
        "1000 m master grid",

    "geospatial_crs":
        "EPSG:4326",

    "functional_state_dimensions":
        "Sync, uWUE, NIRv",

    "future_standardized_dimensions":
        "ZSync, ZuWUE, ZNIRv",

    "sync_definition":
        (
            f"Centered Pearson correlation of GPP and ET "
            f"over {SYNC_WINDOW} 8-day phases"
        ),

    "uwue_definition":
        "GPP * sqrt(VPD) / ET",

    "nirv_role":
        "Vegetation photosynthetic/carbon uptake proxy",

    "important_note":
        (
            "This dataset contains raw ecological functional variables. "
            "Phenological Z-score standardization is NOT performed here."
        )
}


# =============================================================================
# 16. PRE-SAVE QC
# =============================================================================

sep("PRE-SAVE QC")

print(
    "Dimensions:",
    dict(ds_out.sizes)
)

print(
    "\nTime:",
    pd.Timestamp(
        ds_out.time.values[0]
    ).date(),
    "->",
    pd.Timestamp(
        ds_out.time.values[-1]
    ).date()
)

print(
    "\nVariables:"
)

for var in [
    "GPP",
    "ET",
    "NIRv",
    "VPD",
    "uWUE",
    "Sync"
]:

    print(
        f"\n{var}"
    )

    print(
        stat_string(
            ds_out[var]
        )
    )


# =============================================================================
# 17. ENCODING
# =============================================================================

sep("PREPARE NETCDF ENCODING")

encoding = {}

for var in [
    "GPP",
    "ET",
    "NIRv",
    "VPD",
    "uWUE",
    "Sync"
]:

    encoding[var] = {

        "dtype":
            "float32",

        "zlib":
            True,

        "complevel":
            COMPLEVEL,

        "_FillValue":
            np.float32(-9999.0),

        "chunksizes": (
            CHUNK_TIME,
            CHUNK_Y,
            CHUNK_X
        )
    }


# CRS scalar variable
if "crs" in ds_out.variables:

    encoding["crs"] = {
        "dtype":
            "int32"
    }


# =============================================================================
# 18. SAVE
# =============================================================================

sep("WRITE L4 NETCDF")

print(
    "Output:"
)

print(
    OUT_NC
)

ds_out.to_netcdf(
    OUT_NC,
    engine="netcdf4",
    format="NETCDF4",
    encoding=encoding
)

print(
    "\nWRITE COMPLETE"
)


# =============================================================================
# 19. CLOSE INPUTS
# =============================================================================

ds_eco.close()
ds_l3.close()
ds_out.close()

# Release the disk-backed Sync array before deleting the temporary file.
try:
    Sync_memmap.flush()
except Exception:
    pass

try:
    del Sync
except Exception:
    pass

try:
    del Sync_memmap
except Exception:
    pass

gc.collect()

if os.path.exists(SYNC_TMP):
    try:
        os.remove(SYNC_TMP)
        print("\nTemporary Sync memmap removed:")
        print(SYNC_TMP)
    except PermissionError:
        print("\nWARNING: temporary Sync memmap is still locked by Windows.")
        print("It can be deleted manually after Python exits:")
        print(SYNC_TMP)


# =============================================================================
# 20. POST-BUILD QC
# =============================================================================

sep("POST-BUILD NETCDF QC")

qc = xr.open_dataset(
    OUT_NC,
    decode_times=True
)

print(
    "Dimensions:",
    dict(qc.sizes)
)

print(
    "Time:",
    len(qc.time),
    "|",
    pd.Timestamp(
        qc.time.values[0]
    ).date(),
    "->",
    pd.Timestamp(
        qc.time.values[-1]
    ).date()
)

print(
    "\nData variables:"
)

for var in qc.data_vars:

    if var == "crs":
        continue

    da = qc[var]

    print(
        f"\n{var:8s}: "
        f"shape={da.shape}, "
        f"dtype={da.dtype}"
    )

    print(
        "  " +
        stat_string(
            da
        )
    )


# =============================================================================
# 21. POST-WRITE DATA CONSERVATION QC
# =============================================================================

sep("POST-WRITE DATA CONSERVATION QC")

postwrite_nodata_ok = True

for var in ["GPP", "ET", "NIRv", "VPD", "uWUE", "Sync"]:
    before = reference_nodata[var]
    after = nodata_percent(qc[var])
    delta = after - before
    ok = abs(delta) <= NODATA_CONSERVATION_TOL
    postwrite_nodata_ok = postwrite_nodata_ok and ok

    print(
        f"{var:8s}: reference={before:9.4f}% | "
        f"written={after:9.4f}% | delta={delta:+.6f} pp | "
        f"{'PASS' if ok else 'FAIL'}"
    )

print(
    "PASS: written NetCDF preserves valid-cell coverage."
    if postwrite_nodata_ok
    else "FAIL: written NetCDF changed valid-cell coverage."
)


# =============================================================================
# 22. REPRESENTATIVE TIME-SLICE QC
# =============================================================================

sep("REPRESENTATIVE TIME-SLICE QC")

indices = [
    0,
    len(qc.time) // 2,
    len(qc.time) - 1
]

for idx in indices:

    date = pd.Timestamp(
        qc.time.values[idx]
    ).date()

    print(
        f"\nTime[{idx}] = {date}"
    )

    for var in [
        "GPP",
        "ET",
        "NIRv",
        "VPD",
        "uWUE",
        "Sync"
    ]:

        da = qc[var].isel(
            time=idx
        )

        try:

            vmin = float(
                da.min(
                    skipna=True
                ).values
            )

            vmax = float(
                da.max(
                    skipna=True
                ).values
            )

            mean = float(
                da.mean(
                    skipna=True
                ).values
            )

            print(
                f"  {var:8s} "
                f"min={vmin:12.6f} "
                f"max={vmax:12.6f} "
                f"mean={mean:12.6f}"
            )

        except Exception as exc:

            print(
                f"  {var:8s} FAILED: "
                f"{repr(exc)}"
            )


# =============================================================================
# 23. FINAL STATUS
# =============================================================================

sep("FINAL RESULT")

expected_dims = {
    "time": 1142,
    "y": 414,
    "x": 703
}

dims_ok = all(
    qc.sizes.get(k) == v
    for k, v in expected_dims.items()
)

required_out = [
    "GPP",
    "ET",
    "NIRv",
    "VPD",
    "uWUE",
    "Sync"
]

vars_ok = all(
    var in qc.data_vars
    for var in required_out
)

time_ok = (
    len(qc.time) == 1142
)

sync_range_ok = True

try:

    sync_min = float(
        qc["Sync"]
        .min(skipna=True)
        .values
    )

    sync_max = float(
        qc["Sync"]
        .max(skipna=True)
        .values
    )

    sync_range_ok = (
        sync_min >= -1.0001
        and
        sync_max <= 1.0001
    )

except Exception:

    sync_range_ok = False


uwue_range_ok = True

try:

    uwue_min = float(
        qc["uWUE"]
        .min(skipna=True)
        .values
    )

    uwue_max = float(
        qc["uWUE"]
        .max(skipna=True)
        .values
    )

    uwue_range_ok = (
        uwue_min >= UWUE_MIN
        and
        uwue_max <= UWUE_MAX
    )

except Exception:

    uwue_range_ok = False


if (
    dims_ok
    and
    vars_ok
    and
    time_ok
    and
    sync_range_ok
    and
    uwue_range_ok
    and
    assembly_nodata_ok
    and
    postwrite_nodata_ok
):

    print(
        "PASS: L4.1 EcoFunction RAW dataset "
        "constructed successfully with master-grid and data-conservation QC."
    )

else:

    print(
        "WARNING: L4.1 completed but one or more "
        "QC conditions failed."
    )


print(
    f"Dimensions PASS : {dims_ok}"
)

print(
    f"Variables PASS  : {vars_ok}"
)

print(
    f"Time PASS       : {time_ok}"
)

print(
    f"Sync range PASS : {sync_range_ok}"
)

print(
    f"uWUE range PASS : {uwue_range_ok}"
)

print(
    f"Assembly conservation PASS : {assembly_nodata_ok}"
)

print(
    f"Post-write conservation PASS: {postwrite_nodata_ok}"
)

print(
    "\nFINAL FILE:"
)

print(
    OUT_NC
)

qc.close()

print(
    "\nStep L4.1 completed."
)