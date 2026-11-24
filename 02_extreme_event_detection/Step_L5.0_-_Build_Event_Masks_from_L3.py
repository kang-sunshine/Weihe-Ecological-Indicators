# -*- coding: utf-8 -*-
"""
Step L5.0 - Build Pixel-wise Extreme Climate Event Masks from L3 (Robust V1)

Purpose
-------
Construct event-mask variables from the continuous L3 climate-stress cube.

This step is required because the current L3 product contains continuous
climate variables only:
    Tmax, Temp, Precip, VPD, RootSM, DeepSM, RAD

It does NOT contain ready-to-use 0/1 event masks.

Supported event masks in this version
-------------------------------------
1. CDH_event
   Compound dry-hot event:
       Tmax >= phase-specific Tmax q90
       AND RootSM <= phase-specific RootSM q10

2. HEAT_event
   High-temperature event:
       Tmax >= phase-specific Tmax q90

   Important:
       This is NOT a strict CDNH/day-night heatwave because the current L3
       dataset does not contain Tmin. If Tmin is added later, HEAT_event can
       be upgraded to a true CDNH_event:
           Tmax >= Tmax q90 AND Tmin >= Tmin q90

3. HR_event
   Heavy rainfall event:
       Precip >= phase-specific Precip q95
       AND Precip >= 1 mm per 8-day phase

Baseline
--------
2000-2020 inclusive, grouped by calendar-anchored 8-day phase.

Outputs
-------
Directory:
    .\Weihe database V3\L5_EventEcoFunction

1. Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc
   Variables:
       CDH_event
       HEAT_event
       HR_event
       phase8

2. Weihe_L5_0_EventThresholds_2000_2020_V3_1.nc
   Variables:
       Tmax_q90
       RootSM_q10
       Precip_q95

3. L5_0_EventMask_Counts.csv
4. L5_0_EventMask_Report.txt

Notes
-----
- No L3/L4 input file is modified.
- No interpolation or resampling is performed.
- Processing is performed by y-blocks.
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

L3_NC = (
    r".\Weihe database V3"
    r"\Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
)

OUT_DIR = (
    r".\Weihe database V3"
    r"\L5_EventEcoFunction"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_MASK_NC = os.path.join(
    OUT_DIR,
    "Weihe_L5_0_EventMasks_1000m_2000_2024_V3_1.nc"
)

OUT_THRESHOLD_NC = os.path.join(
    OUT_DIR,
    "Weihe_L5_0_EventThresholds_2000_2020_V3_1.nc"
)

OUT_COUNTS_CSV = os.path.join(
    OUT_DIR,
    "L5_0_EventMask_Counts.csv"
)

OUT_REPORT_TXT = os.path.join(
    OUT_DIR,
    "L5_0_EventMask_Report.txt"
)

# =============================================================================
# 1. SETTINGS
# =============================================================================

BASELINE_START = 2000
BASELINE_END = 2020

N_PHASES = 46

BLOCK_Y = 24

COMPLEVEL = 4
CHUNK_TIME = 16
CHUNK_Y = 24
CHUNK_X = 352

TMAX_Q = 0.90
ROOTSM_Q = 0.10
PRECIP_Q = 0.95

PRECIP_MIN_MM = 1.0

REQUIRED_L3_VARS = [
    "Tmax",
    "Precip",
    "RootSM",
]

EVENT_VARS = [
    "CDH_event",
    "HEAT_event",
    "HR_event",
]

# =============================================================================
# 2. HELPERS
# =============================================================================

def sep(title):
    print("\n" + "=" * 108)
    print(title)
    print("=" * 108)


def require_file(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found:\n{path}")


def phase8_from_dates(dates):
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy(dtype=np.int16)
    return ((doy - 1) // 8 + 1).astype(np.int8)


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


def create_time_xy(nc, time_index, x, y):
    nc.createDimension("time", len(time_index))
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))

    tvar = nc.createVariable("time", "i4", ("time",))
    units = "days since 1970-01-01 00:00:00"
    calendar = "standard"

    py_dates = [ts.to_pydatetime() for ts in pd.DatetimeIndex(time_index)]

    tvar[:] = date2num(py_dates, units=units, calendar=calendar).astype(np.int32)
    tvar.units = units
    tvar.calendar = calendar
    tvar.standard_name = "time"
    tvar.long_name = "master 8-day phase start date"
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


def create_phase_yx(nc, x, y):
    nc.createDimension("phase", N_PHASES)
    nc.createDimension("y", len(y))
    nc.createDimension("x", len(x))

    pvar = nc.createVariable("phase", "i1", ("phase",))
    pvar[:] = np.arange(1, N_PHASES + 1, dtype=np.int8)
    pvar.long_name = "calendar-anchored 8-day phase"
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
    yvar.units = "degrees_north"
    yvar.axis = "Y"

    xvar.standard_name = "longitude"
    xvar.units = "degrees_east"
    xvar.axis = "X"


def copy_crs_if_present(xr_ds, nc):
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


def block_stat_mask(mask):
    total = mask.size
    count = int(mask.sum())
    pct = 100.0 * count / total
    return count, pct


# =============================================================================
# 3. OPEN L3
# =============================================================================

sep("OPEN L3 CLIMATE-STRESS DATASET")

require_file(L3_NC, "L3 climate-stress NetCDF")

print("Input:")
print(L3_NC)

ds = xr.open_dataset(
    L3_NC,
    decode_times=True,
    cache=False
)

print("\nDimensions:")
print(dict(ds.sizes))

for var in REQUIRED_L3_VARS:
    if var not in ds.data_vars:
        raise KeyError(
            f"Required L3 variable missing: {var}"
        )

print("\nRequired L3 variables:")
for var in REQUIRED_L3_VARS:
    print(f"{var:10s}: FOUND")

time_index = pd.DatetimeIndex(pd.to_datetime(ds.time.values))
x = np.asarray(ds.x.values, dtype=np.float64)
y = np.asarray(ds.y.values, dtype=np.float64)

nt = len(time_index)
ny = len(y)
nx = len(x)

years = time_index.year.to_numpy(dtype=np.int16)
phase8 = phase8_from_dates(time_index)

baseline_mask = (
    (years >= BASELINE_START)
    &
    (years <= BASELINE_END)
)

print(
    "\nTime:",
    time_index[0].date(),
    "->",
    time_index[-1].date()
)

print(f"Grid: time={nt}, y={ny}, x={nx}")

print(
    f"Baseline: {BASELINE_START}-{BASELINE_END} | "
    f"{int(baseline_mask.sum())} timestamps"
)

print(
    f"Phase range: {int(phase8.min())} -> {int(phase8.max())} | "
    f"unique={len(np.unique(phase8))}"
)

if (
    phase8.min() != 1
    or
    phase8.max() != 46
    or
    len(np.unique(phase8)) != 46
):
    raise RuntimeError("Expected complete 46-phase time axis was not found.")

# Pre-compute phase index lists.
all_idx_by_phase = {
    p: np.where(phase8 == p)[0]
    for p in range(1, N_PHASES + 1)
}

base_idx_by_phase = {
    p: np.where(baseline_mask & (phase8 == p))[0]
    for p in range(1, N_PHASES + 1)
}

# =============================================================================
# 4. CREATE OUTPUT FILES
# =============================================================================

sep("CREATE OUTPUT NETCDF FILES")

for path in [OUT_MASK_NC, OUT_THRESHOLD_NC]:
    if os.path.exists(path):
        print("Removing existing output:")
        print(path)
        os.remove(path)

mask_nc = Dataset(OUT_MASK_NC, "w", format="NETCDF4")
thr_nc = Dataset(OUT_THRESHOLD_NC, "w", format="NETCDF4")

create_time_xy(mask_nc, time_index, x, y)
create_phase_yx(thr_nc, x, y)

phase_var = mask_nc.createVariable(
    "phase8",
    "i1",
    ("time",),
    zlib=True,
    complevel=COMPLEVEL
)
phase_var[:] = phase8
phase_var.long_name = "calendar-anchored 8-day phase"
phase_var.valid_min = np.int8(1)
phase_var.valid_max = np.int8(46)

copy_crs_if_present(ds, mask_nc)
copy_crs_if_present(ds, thr_nc)

created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

mask_nc.Conventions = "CF-1.8"
mask_nc.title = "Weihe River Basin L5.0 Extreme Climate Event Masks"
mask_nc.processing_level = "L5.0 Event Masks"
mask_nc.dataset_version = "V3.1"
mask_nc.created_utc = created
mask_nc.source = os.path.basename(L3_NC)
mask_nc.time_coverage_start = str(time_index[0].date())
mask_nc.time_coverage_end = str(time_index[-1].date())
mask_nc.temporal_resolution = "8-day master phases"
mask_nc.spatial_resolution = "1000 m master grid"
mask_nc.baseline_period = f"{BASELINE_START}-{BASELINE_END}"
mask_nc.phase_definition = "phase = floor((day_of_year - 1) / 8) + 1; phases 1..46"
mask_nc.CDH_definition = "Tmax >= phase q90 AND RootSM <= phase q10"
mask_nc.HEAT_definition = "Tmax >= phase q90; not a strict CDNH because Tmin is unavailable"
mask_nc.HR_definition = f"Precip >= phase q95 AND Precip >= {PRECIP_MIN_MM:g} mm"
mask_nc.important_note = (
    "HEAT_event is a Tmax-based high-temperature event. It should not be "
    "described as compound day-night heatwave unless Tmin is added."
)

thr_nc.Conventions = "CF-1.8"
thr_nc.title = "Weihe River Basin L5.0 Phase-specific Event Thresholds"
thr_nc.processing_level = "L5.0 Event Threshold Audit"
thr_nc.dataset_version = "V3.1"
thr_nc.created_utc = created
thr_nc.source = os.path.basename(L3_NC)
thr_nc.baseline_period = f"{BASELINE_START}-{BASELINE_END}"
thr_nc.phase_definition = mask_nc.phase_definition

# Output variables.
event_nc_vars = {}

for var in EVENT_VARS:
    outv = mask_nc.createVariable(
        var,
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

    outv.units = "1"
    outv.flag_values = np.asarray([0, 1], dtype=np.int8)
    outv.flag_meanings = "non_event event"

    if var == "CDH_event":
        outv.long_name = "Compound dry-hot event mask"
        outv.definition = mask_nc.CDH_definition
    elif var == "HEAT_event":
        outv.long_name = "Tmax-based high-temperature event mask"
        outv.definition = mask_nc.HEAT_definition
    elif var == "HR_event":
        outv.long_name = "Heavy rainfall event mask"
        outv.definition = mask_nc.HR_definition

    event_nc_vars[var] = outv

thr_vars = {}

for name, long_name, q in [
    ("Tmax_q90", "Phase-specific 90th percentile of Tmax", TMAX_Q),
    ("RootSM_q10", "Phase-specific 10th percentile of RootSM", ROOTSM_Q),
    ("Precip_q95", "Phase-specific 95th percentile of Precip", PRECIP_Q),
]:
    outv = thr_nc.createVariable(
        name,
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
    outv.long_name = long_name
    outv.quantile = q
    outv.baseline_period = f"{BASELINE_START}-{BASELINE_END}"
    thr_vars[name] = outv

print("Event mask output:")
print(OUT_MASK_NC)
print("\nThreshold output:")
print(OUT_THRESHOLD_NC)

# =============================================================================
# 5. BUILD EVENT MASKS BY BLOCK
# =============================================================================

sep("BUILD EVENT MASKS BY PHASE AND SPATIAL BLOCK")

n_blocks = int(np.ceil(ny / BLOCK_Y))

global_counts = {
    "CDH_event": 0,
    "HEAT_event": 0,
    "HR_event": 0,
}

global_valid_climate = 0
global_total_cells = nt * ny * nx

for ib, y0 in enumerate(range(0, ny, BLOCK_Y), start=1):

    y1 = min(y0 + BLOCK_Y, ny)
    byn = y1 - y0

    print(f"Block {ib:02d}/{n_blocks:02d} | y={y0}:{y1}")

    tmax = masked_to_nan(
        ds["Tmax"].isel(y=slice(y0, y1)).transpose("time", "y", "x").values,
        dtype=np.float32
    )

    rootsm = masked_to_nan(
        ds["RootSM"].isel(y=slice(y0, y1)).transpose("time", "y", "x").values,
        dtype=np.float32
    )

    precip = masked_to_nan(
        ds["Precip"].isel(y=slice(y0, y1)).transpose("time", "y", "x").values,
        dtype=np.float32
    )

    expected_shape = (nt, byn, nx)

    if tmax.shape != expected_shape or rootsm.shape != expected_shape or precip.shape != expected_shape:
        raise RuntimeError("Unexpected variable block shape.")

    cdh = np.zeros(expected_shape, dtype=np.int8)
    heat = np.zeros(expected_shape, dtype=np.int8)
    hr = np.zeros(expected_shape, dtype=np.int8)

    valid_climate = (
        np.isfinite(tmax)
        &
        np.isfinite(rootsm)
        &
        np.isfinite(precip)
    )

    global_valid_climate += int(valid_climate.sum())

    for p in range(1, N_PHASES + 1):

        idx_base = base_idx_by_phase[p]
        idx_all = all_idx_by_phase[p]

        tmax_base = tmax[idx_base, :, :]
        rootsm_base = rootsm[idx_base, :, :]
        precip_base = precip[idx_base, :, :]

        with np.errstate(invalid="ignore"):
            tmax_q90 = np.nanquantile(tmax_base, TMAX_Q, axis=0).astype(np.float32)
            rootsm_q10 = np.nanquantile(rootsm_base, ROOTSM_Q, axis=0).astype(np.float32)
            precip_q95 = np.nanquantile(precip_base, PRECIP_Q, axis=0).astype(np.float32)

        thr_vars["Tmax_q90"][p - 1, y0:y1, :] = np.ma.masked_invalid(tmax_q90)
        thr_vars["RootSM_q10"][p - 1, y0:y1, :] = np.ma.masked_invalid(rootsm_q10)
        thr_vars["Precip_q95"][p - 1, y0:y1, :] = np.ma.masked_invalid(precip_q95)

        tmax_all = tmax[idx_all, :, :]
        rootsm_all = rootsm[idx_all, :, :]
        precip_all = precip[idx_all, :, :]

        valid_tmax = np.isfinite(tmax_all) & np.isfinite(tmax_q90[None, :, :])
        valid_cdh = (
            np.isfinite(tmax_all)
            &
            np.isfinite(rootsm_all)
            &
            np.isfinite(tmax_q90[None, :, :])
            &
            np.isfinite(rootsm_q10[None, :, :])
        )
        valid_hr = np.isfinite(precip_all) & np.isfinite(precip_q95[None, :, :])

        heat_p = (
            valid_tmax
            &
            (tmax_all >= tmax_q90[None, :, :])
        )

        cdh_p = (
            valid_cdh
            &
            (tmax_all >= tmax_q90[None, :, :])
            &
            (rootsm_all <= rootsm_q10[None, :, :])
        )

        hr_p = (
            valid_hr
            &
            (precip_all >= precip_q95[None, :, :])
            &
            (precip_all >= PRECIP_MIN_MM)
        )

        heat[idx_all, :, :] = heat_p.astype(np.int8)
        cdh[idx_all, :, :] = cdh_p.astype(np.int8)
        hr[idx_all, :, :] = hr_p.astype(np.int8)

        del tmax_base, rootsm_base, precip_base
        del tmax_q90, rootsm_q10, precip_q95
        del tmax_all, rootsm_all, precip_all
        del valid_tmax, valid_cdh, valid_hr
        del heat_p, cdh_p, hr_p

    block_counts = {
        "CDH_event": int(cdh.sum()),
        "HEAT_event": int(heat.sum()),
        "HR_event": int(hr.sum()),
    }

    for k, v in block_counts.items():
        global_counts[k] += v

    event_nc_vars["CDH_event"][:, y0:y1, :] = cdh
    event_nc_vars["HEAT_event"][:, y0:y1, :] = heat
    event_nc_vars["HR_event"][:, y0:y1, :] = hr

    print(
        f"  CDH={block_counts['CDH_event']:,} | "
        f"HEAT={block_counts['HEAT_event']:,} | "
        f"HR={block_counts['HR_event']:,}"
    )

    del tmax, rootsm, precip
    del valid_climate, cdh, heat, hr
    gc.collect()

    mask_nc.sync()
    thr_nc.sync()

# =============================================================================
# 6. CLOSE WRITERS
# =============================================================================

sep("FINALIZE NETCDF WRITING")

mask_nc.sync()
thr_nc.sync()
mask_nc.close()
thr_nc.close()
ds.close()

print("WRITE COMPLETE")
print(OUT_MASK_NC)
print(OUT_THRESHOLD_NC)

# =============================================================================
# 7. POST-BUILD QC
# =============================================================================

sep("POST-BUILD QC")

qc = xr.open_dataset(
    OUT_MASK_NC,
    decode_times=True,
    cache=False
)

print("Dimensions:")
print(dict(qc.sizes))

dims_ok = (
    qc.sizes.get("time") == nt
    and qc.sizes.get("y") == ny
    and qc.sizes.get("x") == nx
)

time_ok = np.array_equal(
    pd.DatetimeIndex(qc.time.values).values,
    time_index.values
)

vars_ok = all(
    v in qc.data_vars
    for v in EVENT_VARS
)

counts_rows = []

for var in EVENT_VARS:

    arr_sum = 0

    for y0 in range(0, ny, BLOCK_Y):
        y1 = min(y0 + BLOCK_Y, ny)
        a = np.asarray(qc[var].isel(y=slice(y0, y1)).values, dtype=np.int8)
        arr_sum += int((a == 1).sum())

    pct_total = 100.0 * arr_sum / global_total_cells

    counts_rows.append(
        {
            "event_variable": var,
            "event_cells": arr_sum,
            "percent_of_all_time_space_cells": pct_total,
            "definition": getattr(qc[var], "definition", ""),
        }
    )

    print(
        f"{var:10s}: cells={arr_sum:,} | "
        f"{pct_total:.6f}% of all time-space cells"
    )

counts_df = pd.DataFrame(counts_rows)
counts_df.to_csv(
    OUT_COUNTS_CSV,
    index=False,
    encoding="utf-8-sig"
)

qc.close()

# =============================================================================
# 8. REPORT
# =============================================================================

sep("WRITE REPORT")

event_nonzero_ok = all(
    row["event_cells"] > 0
    for row in counts_rows
)

final_ok = all([
    dims_ok,
    time_ok,
    vars_ok,
    event_nonzero_ok,
])

report_lines = []
report_lines.append("Step L5.0 Event Mask Construction Report")
report_lines.append("=" * 76)
report_lines.append("")
report_lines.append(f"Created UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
report_lines.append("")
report_lines.append("Input:")
report_lines.append(f"  {L3_NC}")
report_lines.append("")
report_lines.append("Outputs:")
report_lines.append(f"  Event masks: {OUT_MASK_NC}")
report_lines.append(f"  Thresholds : {OUT_THRESHOLD_NC}")
report_lines.append(f"  Counts     : {OUT_COUNTS_CSV}")
report_lines.append("")
report_lines.append(f"Baseline: {BASELINE_START}-{BASELINE_END}")
report_lines.append("Phase grouping: calendar-anchored 8-day phases, 1..46")
report_lines.append("")
report_lines.append("Definitions:")
report_lines.append(f"  CDH_event  = Tmax >= phase q90 AND RootSM <= phase q10")
report_lines.append(f"  HEAT_event = Tmax >= phase q90")
report_lines.append(f"  HR_event   = Precip >= phase q95 AND Precip >= {PRECIP_MIN_MM:g} mm")
report_lines.append("")
report_lines.append("Important note:")
report_lines.append(
    "  HEAT_event is NOT a strict CDNH/day-night heatwave because the current "
    "L3 dataset does not include Tmin. Do not label it as CDNH in manuscript "
    "figures or methods unless Tmin is added and the mask is rebuilt."
)
report_lines.append("")
report_lines.append("Event counts:")

for row in counts_rows:
    report_lines.append(
        f"  {row['event_variable']}: "
        f"{int(row['event_cells']):,} cells "
        f"({float(row['percent_of_all_time_space_cells']):.6f}% of all cells)"
    )

report_lines.append("")
report_lines.append("QC:")
report_lines.append(f"  Dimensions PASS : {dims_ok}")
report_lines.append(f"  Time PASS       : {time_ok}")
report_lines.append(f"  Variables PASS  : {vars_ok}")
report_lines.append(f"  Nonzero events  : {event_nonzero_ok}")
report_lines.append("")
report_lines.append(f"Final PASS: {final_ok}")

report_text = "\n".join(report_lines)

with open(
    OUT_REPORT_TXT,
    "w",
    encoding="utf-8"
) as f:
    f.write(report_text)

print(report_text)

print("\nSaved:")
print(OUT_REPORT_TXT)

# =============================================================================
# 9. FINAL RESULT
# =============================================================================

sep("FINAL RESULT")

if final_ok:
    print("PASS: Step L5.0 event masks constructed successfully.")
else:
    print("WARNING: Step L5.0 completed, but one or more QC checks failed.")

print(f"Dimensions PASS : {dims_ok}")
print(f"Time PASS       : {time_ok}")
print(f"Variables PASS  : {vars_ok}")
print(f"Nonzero PASS    : {event_nonzero_ok}")

print("\nFINAL EVENT MASK FILE:")
print(OUT_MASK_NC)

print("\nFINAL THRESHOLD FILE:")
print(OUT_THRESHOLD_NC)

print("\nStep L5.0 completed.")
