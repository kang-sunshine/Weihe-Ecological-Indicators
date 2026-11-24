# -*- coding: utf-8 -*-
"""
Build_Weihe_L3_V3_1_Final_NetCDF.py

Build the final NetCDF4 dataset from GEE V3.1 L3 annual GeoTIFFs.
Input directory: .\\Weihe database V3
Output: Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc

Expected structure:
- 2000: 38 phases, 266 bands, 2000-03-05 -> 2000-12-26
- 2001-2024: 46 phases/year, 322 bands/year
- total time: 1142
- variables: Tmax, Temp, Precip, VPD, RootSM, DeepSM, RAD
- NetCDF variable dimensions: (time, y, x)
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import rasterio

try:
    from netCDF4 import Dataset, date2num, num2date
except ImportError:
    raise SystemExit(
        "Missing package netCDF4. Install it first with:\n"
        "pip install netCDF4"
    )

# =============================
# USER SETTINGS
# =============================
DATA_DIR = r".\Weihe database V3"
START_YEAR = 2000
END_YEAR = 2024
FILE_TEMPLATE = "Weihe_L3_ClimateStress_1000m_{year}.tif"
OUTPUT_NC = os.path.join(
    DATA_DIR,
    "Weihe_L3_ClimateStress_1000m_2000_2024_V3_1_FINAL.nc"
)

# False prevents accidental overwrite.
# Change to True only when you intentionally want to rebuild the NC.
OVERWRITE = False
COMPRESSION_LEVEL = 4
CHUNK_Y = 256
CHUNK_X = 256

VARIABLES = ["Tmax", "Temp", "Precip", "VPD", "RootSM", "DeepSM", "RAD"]
NVAR = len(VARIABLES)

EXPECTED_PHASES = {2000: 38}
for y in range(2001, END_YEAR + 1):
    EXPECTED_PHASES[y] = 46
EXPECTED_TOTAL_TIME = sum(EXPECTED_PHASES.values())
assert EXPECTED_TOTAL_TIME == 1142

VAR_META = {
    "Tmax":   ("8-day maximum daily maximum air temperature", "degree_Celsius"),
    "Temp":   ("8-day mean air temperature", "degree_Celsius"),
    "Precip": ("8-day accumulated precipitation", "mm"),
    "VPD":    ("8-day vapor pressure deficit", "kPa"),
    "RootSM": ("root-zone volumetric soil moisture", "m3 m-3"),
    "DeepSM": ("deep-layer volumetric soil moisture", "m3 m-3"),
    "RAD":    ("8-day accumulated surface solar radiation", "MJ m-2"),
}

HARD_RANGES = {
    "Tmax": (-60.0, 70.0),
    "Temp": (-60.0, 60.0),
    "Precip": (0.0, 1500.0),
    "VPD": (0.0, 15.0),
    "RootSM": (0.0, 1.0),
    "DeepSM": (0.0, 1.0),
    "RAD": (0.0, 500.0),
}


def parse_band(desc):
    if desc is None:
        return None, None
    desc = str(desc)
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", desc)
    date_str = m.group(1) if m else None
    var = None
    for name in VARIABLES:
        if re.search(rf"(^|_){re.escape(name)}($|_)", desc, flags=re.I):
            var = name
            break
    return date_str, var


def expected_dates(year):
    if year == 2000:
        start, n = datetime(2000, 3, 5), 38
    else:
        start, n = datetime(year, 1, 1), 46
    return [start + timedelta(days=8 * i) for i in range(n)]


def input_files():
    out = []
    for year in range(START_YEAR, END_YEAR + 1):
        p = os.path.join(DATA_DIR, FILE_TEMPLATE.format(year=year))
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input file: {p}")
        out.append((year, p))
    return out


def close_tuple(a, b, atol=1e-10):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=atol)


def verify_inputs(files):
    print("=" * 96)
    print("PRE-BUILD INPUT VERIFICATION")
    print("=" * 96)

    ref = None
    all_dates = []

    for year, path in files:
        with rasterio.open(path) as src:
            exp_nphase = EXPECTED_PHASES[year]
            exp_nband = exp_nphase * NVAR
            if src.count != exp_nband:
                raise RuntimeError(f"{year}: expected {exp_nband} bands, got {src.count}")

            geom = {
                "width": src.width,
                "height": src.height,
                "crs": str(src.crs),
                "transform": tuple(src.transform),
                "bounds": tuple(src.bounds),
                "res": tuple(src.res),
            }
            if ref is None:
                ref = geom
            else:
                if geom["width"] != ref["width"] or geom["height"] != ref["height"]:
                    raise RuntimeError(f"{year}: raster size mismatch")
                if geom["crs"] != ref["crs"]:
                    raise RuntimeError(f"{year}: CRS mismatch")
                if not close_tuple(geom["transform"], ref["transform"]):
                    raise RuntimeError(f"{year}: transform mismatch")
                if not close_tuple(geom["bounds"], ref["bounds"]):
                    raise RuntimeError(f"{year}: bounds mismatch")
                if not close_tuple(geom["res"], ref["res"]):
                    raise RuntimeError(f"{year}: resolution mismatch")

            observed = []
            for phase in range(exp_nphase):
                base = phase * NVAR
                dates, vars_ = [], []
                for k in range(NVAR):
                    d, v = parse_band(src.descriptions[base + k])
                    dates.append(d)
                    vars_.append(v)
                if vars_ != VARIABLES:
                    raise RuntimeError(f"{year}, phase {phase+1}: variable order error: {vars_}")
                if len(set(dates)) != 1 or dates[0] is None:
                    raise RuntimeError(f"{year}, phase {phase+1}: date group error: {dates}")
                observed.append(datetime.strptime(dates[0], "%Y-%m-%d"))

            exp_dates = expected_dates(year)
            if observed != exp_dates:
                raise RuntimeError(
                    f"{year}: time axis mismatch. "
                    f"Observed {observed[0].date()}..{observed[-1].date()}, "
                    f"expected {exp_dates[0].date()}..{exp_dates[-1].date()}"
                )

            all_dates.extend(observed)
            print(f"PASS {year}: {src.count} bands | {len(observed)} phases | "
                  f"{observed[0].date()} -> {observed[-1].date()}")

    if len(all_dates) != EXPECTED_TOTAL_TIME:
        raise RuntimeError(f"Expected 1142 phases, got {len(all_dates)}")
    if not all(all_dates[i] > all_dates[i-1] for i in range(1, len(all_dates))):
        raise RuntimeError("Combined time axis is not strictly increasing")

    print("-" * 96)
    print(f"INPUT VERIFIED: 25 years | {len(all_dates)} total phases")
    return ref, all_dates


def xy_coords(transform, width, height):
    x = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    y = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    return x, y


def build_netcdf(files, ref, all_dates):
    if os.path.exists(OUTPUT_NC):
        if not OVERWRITE:
            raise FileExistsError(
                f"Output already exists:\n{OUTPUT_NC}\n"
                "Set OVERWRITE = True only if you intentionally want to rebuild it."
            )
        os.remove(OUTPUT_NC)

    width, height = ref["width"], ref["height"]
    with rasterio.open(files[0][1]) as s0:
        transform = s0.transform
        crs_wkt = s0.crs.to_wkt() if s0.crs else ""
    x, y = xy_coords(transform, width, height)

    print("\n" + "=" * 96)
    print("BUILDING FINAL NETCDF")
    print("=" * 96)
    print(f"Output: {OUTPUT_NC}")
    print(f"Shape per variable: ({EXPECTED_TOTAL_TIME}, {height}, {width})")

    nc = Dataset(OUTPUT_NC, "w", format="NETCDF4")
    try:
        nc.Conventions = "CF-1.8"
        nc.title = "Weihe River Basin GEE V3.1 L3 Climate Stress Dataset"
        nc.source = "Google Earth Engine V3.1 L3-only annual GeoTIFF exports"
        nc.processing_level = "L3"
        nc.dataset_version = "V3.1 FINAL"
        nc.qc_status = "PASS: 25 years, 0 warnings, 0 failures"
        nc.time_coverage_start = all_dates[0].strftime("%Y-%m-%d")
        nc.time_coverage_end = all_dates[-1].strftime("%Y-%m-%d")
        nc.temporal_resolution = "8-day master phases"
        nc.geospatial_crs = "EPSG:4326"
        nc.history = f"Created {datetime.now():%Y-%m-%d %H:%M:%S} from V3.1 annual L3 GeoTIFFs"
        nc.comment = "Master timeline starts 2000-03-05; 2000 has 38 phases; 2001-2024 have 46 phases/year."

        nc.createDimension("time", EXPECTED_TOTAL_TIME)
        nc.createDimension("y", height)
        nc.createDimension("x", width)

        time_units = "days since 1970-01-01 00:00:00"
        tv = nc.createVariable("time", "f8", ("time",))
        tv.standard_name = "time"
        tv.long_name = "master 8-day phase start date"
        tv.units = time_units
        tv.calendar = "standard"
        tv.axis = "T"
        tv[:] = date2num(all_dates, units=time_units, calendar="standard")

        xv = nc.createVariable("x", "f8", ("x",))
        xv.standard_name = "longitude"
        xv.long_name = "longitude"
        xv.units = "degrees_east"
        xv.axis = "X"
        xv[:] = x

        yv = nc.createVariable("y", "f8", ("y",))
        yv.standard_name = "latitude"
        yv.long_name = "latitude"
        yv.units = "degrees_north"
        yv.axis = "Y"
        yv[:] = y

        crs = nc.createVariable("crs", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        crs.epsg_code = "EPSG:4326"
        crs.spatial_ref = crs_wkt
        crs.crs_wkt = crs_wkt

        chunks = (1, min(CHUNK_Y, height), min(CHUNK_X, width))
        ncv = {}
        for name in VARIABLES:
            v = nc.createVariable(
                name, "f4", ("time", "y", "x"),
                zlib=True, complevel=COMPRESSION_LEVEL, shuffle=True,
                chunksizes=chunks, fill_value=np.float32(np.nan)
            )
            v.long_name, v.units = VAR_META[name]
            v.grid_mapping = "crs"
            v.coordinates = "y x"
            v.valid_min = np.float32(HARD_RANGES[name][0])
            v.valid_max = np.float32(HARD_RANGES[name][1])
            ncv[name] = v

        global_t = 0
        for year, path in files:
            nphase = EXPECTED_PHASES[year]
            print(f"\nWriting {year}: {nphase} phases")
            with rasterio.open(path) as src:
                for phase in range(nphase):
                    base = phase * NVAR
                    for k, name in enumerate(VARIABLES):
                        arr = src.read(base + k + 1, masked=True)
                        data = np.asarray(arr.filled(np.nan), dtype=np.float32)
                        data[~np.isfinite(data)] = np.nan
                        ncv[name][global_t, :, :] = data
                    global_t += 1
                    if phase == 0 or (phase + 1) % 10 == 0 or phase == nphase - 1:
                        print(f"  phase {phase+1:02d}/{nphase} | global {global_t}/{EXPECTED_TOTAL_TIME}")
            nc.sync()
            print(f"PASS {year} written")

        if global_t != EXPECTED_TOTAL_TIME:
            raise RuntimeError(f"Final written time count = {global_t}, expected 1142")
        nc.sync()
    finally:
        nc.close()

    size_gb = os.path.getsize(OUTPUT_NC) / (1024**3)
    print("\nNETCDF BUILD FINISHED")
    print(f"File size: {size_gb:.3f} GB")


def post_qc():
    print("\n" + "=" * 96)
    print("POST-BUILD NETCDF QC")
    print("=" * 96)

    with Dataset(OUTPUT_NC, "r") as nc:
        dims = {k: len(v) for k, v in nc.dimensions.items()}
        print("Dimensions:", dims)
        if dims.get("time") != 1142 or dims.get("y") != 414 or dims.get("x") != 703:
            raise RuntimeError(f"Dimension QC failed: {dims}")

        tv = nc.variables["time"]
        dt = num2date(tv[:], tv.units, tv.calendar,
                      only_use_cftime_datetimes=False,
                      only_use_python_datetimes=True)
        first = dt[0].strftime("%Y-%m-%d")
        last = dt[-1].strftime("%Y-%m-%d")
        print(f"Time: {len(dt)} | {first} -> {last}")
        if first != "2000-03-05" or last != "2024-12-26":
            raise RuntimeError("Time endpoint QC failed")

        print("Data variables:")
        for name in VARIABLES:
            if name not in nc.variables:
                raise RuntimeError(f"Missing variable {name}")
            v = nc.variables[name]
            print(f"  {name:8s}: shape={v.shape}, dtype={v.dtype}, units={getattr(v, 'units', '')}")
            if v.shape != (1142, 414, 703):
                raise RuntimeError(f"{name} shape failed: {v.shape}")

        # Spot-check start, middle, end timestamps.
        for idx in (0, 571, 1141):
            d = dt[idx].strftime("%Y-%m-%d")
            print(f"\nSpot-check time[{idx}] = {d}")
            for name in VARIABLES:
                a = nc.variables[name][idx, :, :]
                if np.ma.isMaskedArray(a):
                    a = a.filled(np.nan)
                a = np.asarray(a, dtype=np.float32)
                finite = a[np.isfinite(a)]
                if finite.size == 0:
                    raise RuntimeError(f"{name} @ {d}: no valid pixels")
                mn, mx, mean = float(finite.min()), float(finite.max()), float(finite.mean())
                lo, hi = HARD_RANGES[name]
                ok = mn >= lo and mx <= hi
                print(f"  {name:8s} min={mn:10.4f} max={mx:10.4f} mean={mean:10.4f} {'OK' if ok else 'FAIL'}")
                if not ok:
                    raise RuntimeError(f"{name} @ {d}: hard-range QC failed")

    print("\n" + "=" * 96)
    print("FINAL RESULT")
    print("=" * 96)
    print("PASS: NetCDF reconstruction completed successfully")
    print("PASS: time = 1142")
    print("PASS: y = 414")
    print("PASS: x = 703")
    print("PASS: 7 variables")
    print("PASS: first time = 2000-03-05")
    print("PASS: last time  = 2024-12-26")
    print(f"FINAL FILE: {OUTPUT_NC}")


def main():
    files = input_files()
    ref, dates = verify_inputs(files)
    print("\nReference geometry:")
    print(ref)
    build_netcdf(files, ref, dates)
    post_qc()


if __name__ == "__main__":
    main()
