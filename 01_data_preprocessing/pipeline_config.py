# -*- coding: utf-8 -*-
from pathlib import Path

DATA_DIR = Path(r".\Weihe database V2")
OUT_DIR = DATA_DIR / "Weihe_Processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L2_PATTERN = "Weihe_L2_EcolFunction_500m_*.tif"
L3_PATTERN = "Weihe_L3_ClimateStress_1000m_*.tif"

L2_BAND_VARS = ["LAI","EVI","NIRv","GPP","ET","iWUE_source","uWUE_source"]
L3_BAND_VARS = ["Tmax","Temp","Precip","VPD","RootSM","DeepSM","RAD"]
RAW_L2_KEEP = ["LAI","EVI","NIRv","GPP","ET"]
RAW_L3_KEEP = ["Tmax","Temp","Precip","VPD","RootSM","DeepSM","RAD"]

START_YEAR=2000
END_YEAR=2024
BASELINE_START=2001
BASELINE_END=2020
FULL_YEAR_STEPS=46

# Actual QC: year 2000 has 38 steps in BOTH L2 and L3.
# Initial explicit mapping: standard 8-day phases 9..46.
# Verify using Step1b_inspect_2000_band_metadata.py before Step 2.
YEAR_PHASES={2000:list(range(9,47))}
def phases_for_year(year):
    return YEAR_PHASES.get(year, list(range(1,FULL_YEAR_STEPS+1)))

EVENT_MONTHS=(4,5,6,7,8,9,10)
PERCENTILE_HALF_WINDOW=2
TMAX_Q=90.0
ROOTSM_Q=10.0
MIN_EVENT_STEPS=1
MAX_GAP_STEPS=0
PRE_STEPS=5
POST_STEPS=10
RECOVERY_SEARCH_STEPS=12
DAMAGE_THRESHOLD=-0.5
PHYS_SEVERE_THRESHOLD=-1.0
RECOVERY_THRESHOLD=-0.3
SUSTAINED_STEPS=2
ALPHA_RT=0.5
BETA_DAMAGE=0.5
MAX_TRAJECTORY_SAMPLE=20000
BOOTSTRAP_N=1000
RANDOM_SEED=42

MANIFEST_CSV=OUT_DIR/"Step1_Input_Manifest.csv"
AUDIT_TXT=OUT_DIR/"Step1_QC_Audit.txt"
BAND_AUDIT_TXT=OUT_DIR/"Step1_2000_BandMetadata_Audit.txt"
CUBE_NC=OUT_DIR/"Weihe_Cube_1km_8Day_formal.nc"
CLIM_NC=OUT_DIR/"Step3_Climatology_2001_2020.nc"
ANOM_NC=OUT_DIR/"Step3_AnomalyCube_2001_2024.nc"
EVENT_CSV=OUT_DIR/"Step4_CDHE_EventCatalog.csv"
EVENT_SUMMARY_NC=OUT_DIR/"Step4_CDHE_SpatialSummary.nc"
METRICS_NC=OUT_DIR/"Step5_StructureFunction_Metrics.nc"
TRAJECTORY_CSV=OUT_DIR/"Step5_EventAligned_Trajectories.csv"
EVENT_METRICS_SAMPLE_CSV=OUT_DIR/"Step5_EventMetrics_Sample.csv"
FIG4_PNG=OUT_DIR/"Step5_Fig4_EventScale_FunctionalCascade.png"
