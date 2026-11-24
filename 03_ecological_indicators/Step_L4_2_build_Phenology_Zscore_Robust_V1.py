# -*- coding: utf-8 -*-
"""
Step L4.2 - Build Phenology-normalized EcoFunction Z-score Cube (Robust V1)

Purpose
-------
Construct phenology-normalized ecological functional anomalies for the Weihe
River Basin from the validated L4.1 raw EcoFunction cube.

Input
-----
L4.1 raw EcoFunction NetCDF:
    Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc

Input variables:
    Sync
    uWUE
    NIRv

Baseline
--------
2000-2020 inclusive.

Phenological phase definition
-----------------------------
The master timeline follows calendar-anchored 8-day phases:

    phase = floor((day_of_year - 1) / 8) + 1

This gives phases 1..46.

For each variable X, each phase p, and each pixel:

    Z(X,t) = [X(t) - mean_base(X,p)] / std_base(X,p)

where mean_base and std_base are calculated from observations in the same
8-day phase during 2000-2020.

Important safeguards
--------------------
1. No month-level aggregation is used.
2. No spatial interpolation/resampling is performed.
3. Baseline standard deviation uses sample SD (ddof=1).
4. A baseline requires at least MIN_BASE_VALID valid annual observations.
5. Near-zero baseline SD is masked rather than amplified.
6. Z-scores are NOT clipped. Extreme values are reported by QC only.
7. Processing is performed in y-blocks and written directly to NetCDF4 to
   avoid multi-GB in-memory arrays.
8. A separate phenology-baseline NetCDF is written for auditability.

Outputs
-------
1. Weihe_L4_EcoFunction_Zscore_1000m_2000_2024_V3_1.nc
   Variables:
       ZSync
       ZuWUE
       ZNIRv
       phase8

2. Weihe_L4_EcoFunction_PhenologyBaseline_2000_2020_V3_1.nc
   Variables:
       Sync_mean, Sync_std, Sync_n
       uWUE_mean, uWUE_std, uWUE_n
       NIRv_mean, NIRv_std, NIRv_n
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

L4_RAW_NC = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
    r"\Weihe_L4_EcoFunction_RAW_1000m_2000_2024_V3_1.nc"
)

OUT_DIR = (
    r".\Weihe database V3"
    r"\L4_EcoFunction"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_Z_NC = os.path.join(
    OUT_DIR,
    "Weihe_L4_EcoFunction_Zscore_1000m_2000_2024_V3_1.nc"
)

OUT_BASELINE_NC = os.path.join(
    OUT_DIR,
    "Weihe_L4_EcoFunction_PhenologyBaseline_2000_2020_V3_1.nc"
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

BASELINE_START = 2000
BASELINE_END = 2020

# 46 calendar-anchored 8-day phases per year.
N_PHASES = 46

# Each phase normally has 20-21 baseline observations (one per year).
# Requiring at least 10 valid years tolerates local data gaps while preventing
# poorly supported Z-scores.
MIN_BASE_VALID = 10

# Avoid division by effectively zero temporal variability.
MIN_BASE_STD = 1.0e-6

# Low-memory spatial processing block.
BLOCK_Y = 24

# NetCDF compression/chunking.
COMPLEVEL = 4
CHUNK_TIME = 16
CHUNK_Y = 24
CHUNK_X = 352

# QC thresholds only; Z-scores are NOT clipped.
QC_Z_ABS_1 = 10.0
QC_Z_ABS_2 = 20.0

RAW_TO_Z = {
    "Sync": "ZSync",
    "uWUE": "ZuWUE",
    "NIRv": "ZNIRv",
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def sep(title):
    print("\n" + "=" * 104)
    print(title)
    print("=" * 104)


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")


def phase8_from_dates(dates):
    """Calendar-anchored 8-day phase index, 1..46."""
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=np.int16)
    return ((doy - 1) // 8 + 1).astype(np.int8)


def safe_nan_stats(arr):
    finite = np.isfinite(arr)
    n = int(finite.sum())

    if n == 0:
        return np.nan, np.nan, np.nan, 100.0

    vals = arr[finite].astype(np.float64, copy=False)

    return (
        float(vals.min()),
        float(vals.max()),
        float(vals.mean()),
        100.0 * (arr.size - n) / arr.size,
    )


def create_z_coordinates(nc, time_index, x, y):
    """Create CF-style time/y/x coordinates in the Z-score output."""
    nc.createDimension("time", len(time_index))
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))

    tvar = nc.createVariable("time", "i4", ("time",))
    t_units = "days since 1970-01-01 00:00:00"
    t_calendar = "standard"

    py_dates = [
        ts.to_pydatetime()
        for ts in pd.DatetimeIndex(time_index)
    ]

    tvar[:] = date2num(
        py_dates,
        units=t_units,
        calendar=t_calendar
    ).astype(np.int32)

    tvar.units = t_units
    tvar.calendar = t_calendar
    tvar.standard_name = "time"
    tvar.axis = "T"

    yvar = nc.createVariable("y", "f8", ("y",))
    xvar = nc.createVariable("x", "f8", ("x",))

    yvar[:] = np.asarray(y, dtype=np.float64)
    xvar[:] = np.asarray(x, dtype=np.float64)

    yvar.standard_name = "latitude"
    yvar.long_name = "latitude"
    yvar.units = "degrees_north"
    yvar.axis = "Y"

    xvar.standard_name = "longitude"
    xvar.long_name = "longitude"
    xvar.units = "degrees_east"
    xvar.axis = "X"


def create_baseline_coordinates(nc, x, y):
    """Create phase/y/x coordinates in the baseline audit output."""
    nc.createDimension("phase", N_PHASES)
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))

    pvar = nc.createVariable("phase", "i1", ("phase",))
    pvar[:] = np.arange(1, N_PHASES + 1, dtype=np.int8)
    pvar.long_name = "calendar-anchored 8-day phenological phase"
    pvar.valid_min = np.int8(1)
    pvar.valid_max = np.int8(46)

    ps = nc.createVariable("phase_start_doy", "i2", ("phase",))
    ps[:] = (1 + 8 * np.arange(N_PHASES)).astype(np.int16)
    ps.long_name = "starting day-of-year of each 8-day phase"
    ps.units = "day_of_year"

    yvar = nc.createVariable("y", "f8", ("y",))
    xvar = nc.createVariable("x", "f8", ("x",))

    yvar[:] = np.asarray(y, dtype=np.float64)
    xvar[:] = np.asarray(x, dtype=np.float64)

    yvar.standard_name = "latitude"
    yvar.long_name = "latitude"
    yvar.units = "degrees_north"
    yvar.axis = "Y"

    xvar.standard_name = "longitude"
    xvar.long_name = "longitude"
    xvar.units = "degrees_east"
    xvar.axis = "X"


def copy_crs_if_present(xr_ds, nc):
    """Copy scalar CRS metadata when present."""
    if "crs" not in xr_ds.variables:
        return

    src = xr_ds["crs"]
    crs = nc.createVariable("crs", "i4")

    try:
        src_value = np.asarray(src.values).reshape(-1)
        if src_value.size > 0 and np.isfinite(src_value[0]):
            crs.assignValue(np.int32(src_value[0]))
        else:
            crs.assignValue(np.int32(0))
    except Exception:
        crs.assignValue(np.int32(0))

    for key, value in src.attrs.items():
        try:
            setattr(crs, key, value)
        except Exception:
            setattr(crs, key, str(value))


def block_scan_netcdf(nc_path, var_name, block_y=BLOCK_Y):
    """Low-memory statistics for a 3-D NetCDF variable."""
    with Dataset(nc_path, "r") as nc:
        var = nc.variables[var_name]
        nt, ny, nx = var.shape

        total = nt * ny * nx
        valid_n = 0
        sum_v = 0.0
        min_v = np.inf
        max_v = -np.inf
        gt10 = 0
        gt20 = 0

        for y0 in range(0, ny, block_y):
            y1 = min(y0 + block_y, ny)

            arr = var[:, y0:y1, :]

            if np.ma.isMaskedArray(arr):
                arr = arr.filled(np.nan)

            arr = np.asarray(arr, dtype=np.float32)
            finite = np.isfinite(arr)
            n = int(finite.sum())

            if n > 0:
                vals = arr[finite].astype(np.float64, copy=False)

                valid_n += n
                sum_v += float(vals.sum())
                min_v = min(min_v, float(vals.min()))
                max_v = max(max_v, float(vals.max()))

                absvals = np.abs(vals)
                gt10 += int((absvals > QC_Z_ABS_1).sum())
                gt20 += int((absvals > QC_Z_ABS_2).sum())

        if valid_n == 0:
            return {
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "nodata_pct": 100.0,
                "valid_n": 0,
                "abs_gt10": 0,
                "abs_gt20": 0,
            }

        return {
            "min": min_v,
            "max": max_v,
            "mean": sum_v / valid_n,
            "nodata_pct": 100.0 * (total - valid_n) / total,
            "valid_n": valid_n,
            "abs_gt10": gt10,
            "abs_gt20": gt20,
        }


# =============================================================================
# 3. OPEN AND VALIDATE L4.1
# =============================================================================

sep("OPEN VALIDATED L4.1 ECOFUNCTION DATASET")

require_file(
    L4_RAW_NC,
    "L4.1 raw EcoFunction NetCDF"
)

print("Input:")
print(L4_RAW_NC)

ds = xr.open_dataset(
    L4_RAW_NC,
    decode_times=True,
    cache=False
)

print("\nDimensions:")
print(dict(ds.sizes))

for dim in ("time", "y", "x"):
    if dim not in ds.dims:
        raise RuntimeError(
            f"Required dimension missing: {dim}"
        )

for var in RAW_TO_Z:
    if var not in ds.data_vars:
        raise KeyError(
            f"Required L4.1 variable missing: {var}"
        )

print("\nRequired variables:")
for var in RAW_TO_Z:
    print(f"{var:8s}: FOUND")

time_index = pd.DatetimeIndex(
    pd.to_datetime(ds["time"].values)
)

x = np.asarray(
    ds["x"].values,
    dtype=np.float64
)

y = np.asarray(
    ds["y"].values,
    dtype=np.float64
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

# =============================================================================
# 4. BUILD AND AUDIT 8-DAY PHASE INDEX
# =============================================================================

sep("BUILD 8-DAY PHENOLOGICAL PHASE INDEX")

phase8 = phase8_from_dates(time_index)
years = time_index.year.to_numpy(dtype=np.int16)

print(
    f"Phase range: "
    f"{int(phase8.min())} -> {int(phase8.max())}"
)

print(
    f"Unique phases: "
    f"{len(np.unique(phase8))}"
)

if (
    phase8.min() != 1
    or
    phase8.max() != 46
    or
    len(np.unique(phase8)) != 46
):
    raise RuntimeError(
        "Expected complete phase range 1..46 was not found."
    )

baseline_mask = (
    (years >= BASELINE_START)
    &
    (years <= BASELINE_END)
)

baseline_indices = np.where(
    baseline_mask
)[0]

if baseline_indices.size == 0:
    raise RuntimeError(
        "No timestamps found in the requested baseline period."
    )

print(
    f"Baseline: {BASELINE_START}-{BASELINE_END} inclusive | "
    f"{baseline_indices.size} timestamps"
)

phase_counts_all = np.bincount(
    phase8.astype(np.int16),
    minlength=N_PHASES + 1
)[1:]

phase_counts_base = np.bincount(
    phase8[baseline_mask].astype(np.int16),
    minlength=N_PHASES + 1
)[1:]

print("\nPhase timestamp counts:")
print("Phase | all years | baseline timestamps")

for p in range(1, N_PHASES + 1):
    print(
        f"{p:5d} | "
        f"{int(phase_counts_all[p - 1]):9d} | "
        f"{int(phase_counts_base[p - 1]):19d}"
    )

# Dataset begins on 2000-03-05, so phases 1-8 contain 20 baseline years
# (2001-2020). Later phases contain 21 baseline years (2000-2020).
if phase_counts_base.min() < 20:
    raise RuntimeError(
        "At least one 8-day phase has fewer than 20 baseline timestamps. "
        "Inspect the L4.1 master time axis before proceeding."
    )

print(
    "\nPASS: calendar-anchored 46-phase timeline is consistent."
)

# =============================================================================
# 5. CREATE OUTPUT FILES
# =============================================================================

sep("CREATE OUTPUT NETCDF FILES")

for path in (
    OUT_Z_NC,
    OUT_BASELINE_NC
):
    if os.path.exists(path):
        print(
            "Removing existing output:\n"
            + path
        )
        os.remove(path)

z_nc = Dataset(
    OUT_Z_NC,
    "w",
    format="NETCDF4"
)

base_nc = Dataset(
    OUT_BASELINE_NC,
    "w",
    format="NETCDF4"
)

create_z_coordinates(
    z_nc,
    time_index,
    x,
    y
)

create_baseline_coordinates(
    base_nc,
    x,
    y
)

phase_var = z_nc.createVariable(
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
    z_nc
)

copy_crs_if_present(
    ds,
    base_nc
)

created = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

# Z-score global attributes
z_nc.Conventions = "CF-1.8"
z_nc.title = (
    "Weihe River Basin L4.2 Phenology-normalized "
    "EcoFunction Z-score Dataset"
)
z_nc.processing_level = (
    "L4.2 Phenology-normalized EcoFunction"
)
z_nc.dataset_version = "V3.1"
z_nc.created_utc = created
z_nc.source = os.path.basename(L4_RAW_NC)
z_nc.time_coverage_start = str(
    time_index[0].date()
)
z_nc.time_coverage_end = str(
    time_index[-1].date()
)
z_nc.temporal_resolution = (
    "8-day master phases"
)
z_nc.spatial_resolution = (
    "1000 m master grid"
)
z_nc.baseline_period = (
    f"{BASELINE_START}-{BASELINE_END}"
)
z_nc.phase_definition = (
    "phase = floor((day_of_year - 1) / 8) + 1; "
    "phases 1..46"
)
z_nc.standardization = (
    "Pixel-wise, phase-specific Z-score using baseline "
    "mean and sample standard deviation (ddof=1)"
)
z_nc.minimum_valid_baseline_observations = (
    MIN_BASE_VALID
)
z_nc.minimum_baseline_std = (
    MIN_BASE_STD
)
z_nc.zscore_clipping = "None"
z_nc.note = (
    "Each timestamp is standardized only against observations "
    "from the same calendar-anchored 8-day phase during the "
    "2000-2020 baseline."
)

# Baseline audit global attributes
base_nc.Conventions = "CF-1.8"
base_nc.title = (
    "Weihe River Basin L4.2 8-day Phenology "
    "Baseline Statistics"
)
base_nc.processing_level = (
    "L4.2 baseline audit product"
)
base_nc.dataset_version = "V3.1"
base_nc.created_utc = created
base_nc.source = os.path.basename(L4_RAW_NC)
base_nc.baseline_period = (
    f"{BASELINE_START}-{BASELINE_END}"
)
base_nc.phase_definition = (
    z_nc.phase_definition
)
base_nc.standard_deviation = (
    "Sample standard deviation (ddof=1)"
)
base_nc.minimum_valid_baseline_observations = (
    MIN_BASE_VALID
)
base_nc.minimum_baseline_std = (
    MIN_BASE_STD
)

print("Z-score output:")
print(OUT_Z_NC)

print("\nBaseline audit output:")
print(OUT_BASELINE_NC)

# =============================================================================
# 6. DEFINE OUTPUT VARIABLES
# =============================================================================

sep("DEFINE OUTPUT VARIABLES")

z_vars = {}
base_vars = {}

for raw_name, z_name in RAW_TO_Z.items():

    zvar = z_nc.createVariable(
        z_name,
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

    zvar.long_name = (
        f"Phenology-normalized Z-score of {raw_name}"
    )
    zvar.source_variable = raw_name
    zvar.baseline_period = (
        f"{BASELINE_START}-{BASELINE_END}"
    )
    zvar.baseline_grouping = (
        "same calendar-anchored 8-day phase"
    )
    zvar.formula = (
        "Z = (X - phase_baseline_mean) / phase_baseline_std"
    )
    zvar.baseline_std_ddof = 1
    zvar.minimum_valid_baseline_observations = (
        MIN_BASE_VALID
    )
    zvar.minimum_baseline_std = (
        MIN_BASE_STD
    )
    zvar.units = "1"
    zvar.clipping = "None"

    z_vars[raw_name] = zvar

    mean_var = base_nc.createVariable(
        f"{raw_name}_mean",
        "f4",
        ("phase", "y", "x"),
        zlib=True,
        complevel=COMPLEVEL,
        shuffle=True,
        fill_value=np.float32(-9999.0),
        chunksizes=(
            1,
            min(CHUNK_Y, ny),
            min(CHUNK_X, nx),
        )
    )

    std_var = base_nc.createVariable(
        f"{raw_name}_std",
        "f4",
        ("phase", "y", "x"),
        zlib=True,
        complevel=COMPLEVEL,
        shuffle=True,
        fill_value=np.float32(-9999.0),
        chunksizes=(
            1,
            min(CHUNK_Y, ny),
            min(CHUNK_X, nx),
        )
    )

    n_var = base_nc.createVariable(
        f"{raw_name}_n",
        "i2",
        ("phase", "y", "x"),
        zlib=True,
        complevel=COMPLEVEL,
        shuffle=True,
        fill_value=np.int16(-32767),
        chunksizes=(
            1,
            min(CHUNK_Y, ny),
            min(CHUNK_X, nx),
        )
    )

    mean_var.long_name = (
        f"Baseline phase mean of {raw_name}"
    )

    std_var.long_name = (
        f"Baseline phase sample standard deviation of {raw_name}"
    )

    n_var.long_name = (
        f"Valid baseline observation count of {raw_name}"
    )

    mean_var.baseline_period = (
        f"{BASELINE_START}-{BASELINE_END}"
    )

    std_var.baseline_period = (
        f"{BASELINE_START}-{BASELINE_END}"
    )

    n_var.baseline_period = (
        f"{BASELINE_START}-{BASELINE_END}"
    )

    std_var.ddof = 1

    base_vars[raw_name] = (
        mean_var,
        std_var,
        n_var
    )

    print(
        f"{raw_name:8s} -> {z_name}"
    )

# =============================================================================
# 7. PRE-COMPUTE PHASE INDEX LISTS
# =============================================================================

all_idx_by_phase = {
    p: np.where(
        phase8 == p
    )[0]
    for p in range(1, N_PHASES + 1)
}

base_idx_by_phase = {
    p: np.where(
        baseline_mask
        &
        (phase8 == p)
    )[0]
    for p in range(1, N_PHASES + 1)
}

# =============================================================================
# 8. PROCESS EACH VARIABLE IN LOW-MEMORY BLOCKS
# =============================================================================

sep("CALCULATE PHASE-SPECIFIC BASELINES AND Z-SCORES")

n_blocks = int(
    np.ceil(
        ny / BLOCK_Y
    )
)

for raw_name, z_name in RAW_TO_Z.items():

    sep(
        f"PROCESS {raw_name} -> {z_name}"
    )

    src = ds[raw_name]
    zvar = z_vars[raw_name]

    mean_var, std_var, n_var = (
        base_vars[raw_name]
    )

    var_valid_total = 0
    z_valid_total = 0

    for ib, y0 in enumerate(
        range(0, ny, BLOCK_Y),
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

        # Read only one y-block into RAM.
        block = np.asarray(
            src.isel(
                y=slice(y0, y1)
            ).values,
            dtype=np.float32
        )

        if block.shape != (
            nt,
            byn,
            nx
        ):
            raise RuntimeError(
                f"Unexpected block shape for {raw_name}: "
                f"{block.shape}"
            )

        block[
            ~np.isfinite(block)
        ] = np.nan

        var_valid_total += int(
            np.isfinite(block).sum()
        )

        z_block = np.full(
            block.shape,
            np.nan,
            dtype=np.float32
        )

        for p in range(
            1,
            N_PHASES + 1
        ):

            idx_base = (
                base_idx_by_phase[p]
            )

            idx_all = (
                all_idx_by_phase[p]
            )

            # <=21 baseline years for one phase.
            b = block[
                idx_base,
                :,
                :
            ]

            n = np.sum(
                np.isfinite(b),
                axis=0,
                dtype=np.int16
            )

            with np.errstate(
                invalid="ignore",
                divide="ignore"
            ):
                mean = np.nanmean(
                    b,
                    axis=0,
                    dtype=np.float64
                )

                std = np.nanstd(
                    b,
                    axis=0,
                    ddof=1,
                    dtype=np.float64
                )

            mean = mean.astype(
                np.float32
            )

            std = std.astype(
                np.float32
            )

            enough = (
                n >= MIN_BASE_VALID
            )

            stable_std = (
                np.isfinite(std)
                &
                (std > MIN_BASE_STD)
            )

            baseline_ok = (
                enough
                &
                stable_std
                &
                np.isfinite(mean)
            )

            mean_out = np.where(
                enough
                &
                np.isfinite(mean),
                mean,
                np.nan
            ).astype(
                np.float32
            )

            std_out = np.where(
                baseline_ok,
                std,
                np.nan
            ).astype(
                np.float32
            )

            mean_var[
                p - 1,
                y0:y1,
                :
            ] = np.ma.masked_invalid(
                mean_out
            )

            std_var[
                p - 1,
                y0:y1,
                :
            ] = np.ma.masked_invalid(
                std_out
            )

            n_var[
                p - 1,
                y0:y1,
                :
            ] = n.astype(
                np.int16
            )

            x_phase = block[
                idx_all,
                :,
                :
            ]

            with np.errstate(
                invalid="ignore",
                divide="ignore",
                over="ignore"
            ):
                z_phase = (
                    (
                        x_phase
                        -
                        mean[
                            None,
                            :,
                            :
                        ]
                    )
                    /
                    std[
                        None,
                        :,
                        :
                    ]
                ).astype(
                    np.float32
                )

            # Mask pixels whose phase baseline is unsupported.
            z_phase = np.where(
                baseline_ok[
                    None,
                    :,
                    :
                ],
                z_phase,
                np.nan
            ).astype(
                np.float32
            )

            z_phase[
                ~np.isfinite(z_phase)
            ] = np.nan

            z_block[
                idx_all,
                :,
                :
            ] = z_phase

            del b
            del n
            del mean
            del std
            del enough
            del stable_std
            del baseline_ok
            del mean_out
            del std_out
            del x_phase
            del z_phase

        z_valid_total += int(
            np.isfinite(z_block).sum()
        )

        (
            bmin,
            bmax,
            bmean,
            bnodata
        ) = safe_nan_stats(
            z_block
        )

        print(
            f"  {z_name}: "
            f"min={bmin:.4f} | "
            f"max={bmax:.4f} | "
            f"mean={bmean:.4f} | "
            f"NoData={bnodata:.3f}%"
        )

        zvar[
            :,
            y0:y1,
            :
        ] = np.ma.masked_invalid(
            z_block
        )

        del block
        del z_block

        gc.collect()

    raw_nodata = (
        100.0
        *
        (
            nt * ny * nx
            -
            var_valid_total
        )
        /
        (
            nt * ny * nx
        )
    )

    z_nodata = (
        100.0
        *
        (
            nt * ny * nx
            -
            z_valid_total
        )
        /
        (
            nt * ny * nx
        )
    )

    print(
        f"\n{raw_name} summary:\n"
        f"  raw NoData : {raw_nodata:.4f}%\n"
        f"  {z_name} NoData: {z_nodata:.4f}%\n"
        f"  Additional masking is permitted only where "
        f"baseline n<{MIN_BASE_VALID} or "
        f"SD<={MIN_BASE_STD:g}."
    )

    z_nc.sync()
    base_nc.sync()

# =============================================================================
# 9. CLOSE WRITERS
# =============================================================================

sep("FINALIZE NETCDF WRITING")

z_nc.sync()
base_nc.sync()

z_nc.close()
base_nc.close()
ds.close()

print("WRITE COMPLETE")

print("\nZ-score file:")
print(OUT_Z_NC)

print("\nBaseline audit file:")
print(OUT_BASELINE_NC)

# =============================================================================
# 10. POST-WRITE STRUCTURAL QC
# =============================================================================

sep("POST-WRITE STRUCTURAL QC")

qz = xr.open_dataset(
    OUT_Z_NC,
    decode_times=True,
    cache=False
)

qb = xr.open_dataset(
    OUT_BASELINE_NC,
    decode_times=False,
    cache=False
)

print("Z-score dimensions:")
print(dict(qz.sizes))

print("\nBaseline dimensions:")
print(dict(qb.sizes))

dims_ok = (
    qz.sizes.get("time") == nt
    and
    qz.sizes.get("y") == ny
    and
    qz.sizes.get("x") == nx
    and
    qb.sizes.get("phase") == N_PHASES
    and
    qb.sizes.get("y") == ny
    and
    qb.sizes.get("x") == nx
)

time_ok = np.array_equal(
    pd.DatetimeIndex(
        qz["time"].values
    ).values,
    time_index.values
)

phase_ok = np.array_equal(
    np.asarray(
        qz["phase8"].values,
        dtype=np.int8
    ),
    phase8
)

vars_ok = all(
    z_name in qz.data_vars
    for z_name in RAW_TO_Z.values()
)

baseline_vars_expected = []

for raw_name in RAW_TO_Z:
    baseline_vars_expected.extend(
        [
            f"{raw_name}_mean",
            f"{raw_name}_std",
            f"{raw_name}_n",
        ]
    )

baseline_vars_ok = all(
    var in qb.data_vars
    for var in baseline_vars_expected
)

print(
    f"\nDimensions PASS       : {dims_ok}"
)
print(
    f"Time axis PASS        : {time_ok}"
)
print(
    f"phase8 PASS           : {phase_ok}"
)
print(
    f"Z variables PASS      : {vars_ok}"
)
print(
    f"Baseline vars PASS    : {baseline_vars_ok}"
)

qz.close()
qb.close()

# =============================================================================
# 11. LOW-MEMORY Z-SCORE QC
# =============================================================================

sep("LOW-MEMORY Z-SCORE QC")

z_qc = {}

for z_name in RAW_TO_Z.values():

    stats = block_scan_netcdf(
        OUT_Z_NC,
        z_name
    )

    z_qc[z_name] = stats

    print(
        f"{z_name:8s}: "
        f"min={stats['min']:.6f} | "
        f"max={stats['max']:.6f} | "
        f"mean={stats['mean']:.6f} | "
        f"NoData={stats['nodata_pct']:.3f}% | "
        f"|Z|>{QC_Z_ABS_1:g}: "
        f"{stats['abs_gt10']:,} | "
        f"|Z|>{QC_Z_ABS_2:g}: "
        f"{stats['abs_gt20']:,}"
    )

finite_qc_ok = all(
    np.isfinite(stats["min"])
    and
    np.isfinite(stats["max"])
    and
    np.isfinite(stats["mean"])
    for stats in z_qc.values()
)

# =============================================================================
# 12. BASELINE SELF-CONSISTENCY QC
# =============================================================================

sep("BASELINE SELF-CONSISTENCY QC")

baseline_check = {}

with Dataset(
    OUT_Z_NC,
    "r"
) as nc:

    for z_name in RAW_TO_Z.values():

        var = nc.variables[
            z_name
        ]

        n_total = 0
        sum_total = 0.0
        sumsq_total = 0.0

        for y0 in range(
            0,
            ny,
            BLOCK_Y
        ):

            y1 = min(
                y0 + BLOCK_Y,
                ny
            )

            arr = var[
                baseline_indices,
                y0:y1,
                :
            ]

            if np.ma.isMaskedArray(
                arr
            ):
                arr = arr.filled(
                    np.nan
                )

            arr = np.asarray(
                arr,
                dtype=np.float32
            )

            finite = np.isfinite(
                arr
            )

            if not finite.any():
                continue

            vals = arr[
                finite
            ].astype(
                np.float64,
                copy=False
            )

            n_total += vals.size
            sum_total += float(
                vals.sum()
            )
            sumsq_total += float(
                np.square(
                    vals
                ).sum()
            )

        if n_total > 1:

            mean0 = (
                sum_total
                /
                n_total
            )

            var0 = (
                (
                    sumsq_total
                    -
                    n_total
                    *
                    mean0
                    *
                    mean0
                )
                /
                (
                    n_total - 1
                )
            )

            sd0 = float(
                np.sqrt(
                    max(
                        var0,
                        0.0
                    )
                )
            )

        else:

            mean0 = np.nan
            sd0 = np.nan

        baseline_check[
            z_name
        ] = (
            mean0,
            sd0
        )

        print(
            f"{z_name:8s}: "
            f"baseline pooled mean="
            f"{mean0:.6f} | "
            f"baseline pooled SD="
            f"{sd0:.6f}"
        )

# Each pixel-phase is standardized independently.
# Pooled baseline mean should be extremely close to 0.
baseline_mean_ok = all(
    np.isfinite(mean0)
    and
    abs(mean0) < 0.05
    for mean0, sd0
    in baseline_check.values()
)

# Pooled SD can deviate slightly from exactly 1 because valid sample sizes
# differ by pixel and phase.
baseline_sd_ok = all(
    np.isfinite(sd0)
    and
    0.90 <= sd0 <= 1.10
    for mean0, sd0
    in baseline_check.values()
)

print(
    f"\nBaseline pooled mean PASS : "
    f"{baseline_mean_ok}"
)

print(
    f"Baseline pooled SD PASS   : "
    f"{baseline_sd_ok}"
)

# =============================================================================
# 13. REPRESENTATIVE TIME-SLICE QC
# =============================================================================

sep("REPRESENTATIVE TIME-SLICE QC")

qz = xr.open_dataset(
    OUT_Z_NC,
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
        qz.time.values[idx]
    ).date()

    phase = int(
        qz["phase8"]
        .isel(
            time=idx
        )
        .values
    )

    print(
        f"\nTime[{idx}] = "
        f"{date} | "
        f"phase8={phase}"
    )

    for z_name in RAW_TO_Z.values():

        arr = np.asarray(
            qz[z_name]
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
            f"  {z_name:8s} "
            f"min={vmin:11.5f} "
            f"max={vmax:11.5f} "
            f"mean={mean:11.5f} "
            f"NoData={nodata:7.3f}%"
        )

qz.close()

# =============================================================================
# 14. FINAL STATUS
# =============================================================================

sep("FINAL RESULT")

final_ok = all(
    [
        dims_ok,
        time_ok,
        phase_ok,
        vars_ok,
        baseline_vars_ok,
        finite_qc_ok,
        baseline_mean_ok,
        baseline_sd_ok,
    ]
)

if final_ok:

    print(
        "PASS: Step L4.2 phenology-normalized "
        "EcoFunction Z-score dataset constructed successfully."
    )

else:

    print(
        "WARNING: Step L4.2 completed, but one or more QC "
        "conditions failed. Do NOT continue to the next analytical "
        "level until the failed QC item is reviewed."
    )

print(
    f"Dimensions PASS           : {dims_ok}"
)

print(
    f"Time axis PASS            : {time_ok}"
)

print(
    f"phase8 PASS               : {phase_ok}"
)

print(
    f"Z variables PASS          : {vars_ok}"
)

print(
    f"Baseline variables PASS   : {baseline_vars_ok}"
)

print(
    f"Finite Z QC PASS          : {finite_qc_ok}"
)

print(
    f"Baseline mean≈0 PASS      : {baseline_mean_ok}"
)

print(
    f"Baseline SD≈1 PASS        : {baseline_sd_ok}"
)

print("\nFINAL Z-SCORE FILE:")
print(OUT_Z_NC)

print("\nBASELINE AUDIT FILE:")
print(OUT_BASELINE_NC)

print("\nStep L4.2 completed.")
