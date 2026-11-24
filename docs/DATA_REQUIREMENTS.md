# Data requirements
| Stage | Required input and support | Derived output |
|---|---|---|
| Climate | Annual L3 multiband GeoTIFF, nominal 8-day phases, Tmax, Temp, Precip (phase total in mm), VPD (kPa), RootSM and DeepSM (volume fraction), RAD. User basin boundary and projection. Reference calendar projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a. | Aligned climate NetCDF and pixel-by-phase quantiles |
| Ecological functions | Annual L2 ecological stacks at nominal 500 m with ordered LAI, EVI, NIRv, GPP, ET, iWUE_source, uWUE_source. Supply archived analysis-scale flux values. The retained local code screens GPP >= 0.01 and ET >= 0.05. The production exporter, historical flux scale and optical temporal matcher have not been independently verified. | Analysis grid at nominal 1 km, time/x/y and phase coordinates, ZSync, ZuWUE, ZNIRv, Acomp |
| Events | Climate phase masks, dated indicator cubes and common row/column identifiers. Keep missing phases, including the first eight in 2000. | Per-pixel onset, termination, active phases, hazard severity, damage and observed legacy in anomaly-days |
| Recovery | Event keys, pre-event baseline, peak and confirmation dates, next catalogue onset, observed follow-up days and outcome codes, spatial block IDs. | KM net recovery, RMPT120 in days, competing-risk CIF and spatial intervals |
| Spatial/greening | Projected block coordinates, severity-adjusted CDH pixel response, NIRv trend and seven hydroclimate predictors. Seasonal NIRv trends use 2001–2024 and climate backgrounds 2001–2020. | OOF predictions, conditional damage contrasts and ALE |
| Vegetation and terrain | MCD12Q1 Collection 6.1 LC_Type1 fractions for 2001, 2010, 2020; aligned elevation, slope and aspect. | Type-specific candidates, means, terrain validation |
| Plotting | External frozen figure tables, matrices, coordinates and basin outlines at each script's declared paths. | Presentation figures |

Earth Engine account access and user-owned assets must be configured by the user. Product citations in the paper describe the data context. Calendar identity alone does not establish the origin of the archived ecological fluxes. Historical file names, field spellings and file formats consumed by each stage appear directly in its input declarations. Keep a separate working copy of the input hierarchy because archived scripts may overwrite stage outputs.
