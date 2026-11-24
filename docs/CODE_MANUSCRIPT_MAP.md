# Implementation index

| Implementation | Methods | Results | Figure context | Supplement context |
|---|---|---|---|---|
| 01_data_preprocessing/pipeline_config.py | 2.2, 3.1 | 4.1 | 1–3 | S1 |
| 01_data_preprocessing/Step1_QC_inventory.py | 2.2, 3.1 | 4.1 | 1–3 | S1 |
| 01_data_preprocessing/Step2_build_common_grid_cube.py | 2.2, 3.1 | 4.1 | 1–3 | S1 |
| 01_data_preprocessing/Build_Weihe_L3_V3_1_Final_NetCDF.py | 2.2, 3.1 | 4.1 | 1–3 | S1 |
| 02_extreme_event_detection/Step_L5.0_-_Build_Event_Masks_from_L3.py | 3.1 | 4.1, 4.3 | 3, 6 | S1–S2 |
| 02_extreme_event_detection/Step_L5_2_Reconstruct_Event_Lifecycles_Robust_V1.py | 3.1 | 4.1, 4.3 | 3, 6 | S1–S2 |
| 03_ecological_indicators/Step_L4_1_build_EcoFunction_Robust_V3.py | 3.2 | 4.2 | 4–5 | S1, S4 |
| 03_ecological_indicators/Step_L4_2_build_Phenology_Zscore_Robust_V1.py | 3.2 | 4.2 | 4–5 | S1, S4 |
| 03_ecological_indicators/Step_L4_3_build_EcoFunction_StateSpace_Robust_V1.py | 3.2 | 4.2 | 4–5 | S1, S4 |
| 03_ecological_indicators/Step_L5_1_extract_EventEcoFunction_Response_Robust_V2.py | 3.2 | 4.2 | 4–5 | S1, S4 |
| 04_event_lifecycle_recovery/Step_L5_3_Build_Event_Severity_Resilience_Metrics_Robust_V2_FAST.py | 3.3 | 4.2–4.3 | 4–6 | S2–S5, Fig. S1 |
| 05_severity_response/Step_L5_4A_Severity_Response_Thresholds_FAST.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 05_severity_response/Step_L5_5B5_BasinAnchored_SpatialSensitivity_20km_FINAL.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 05_severity_response/Step_L5_5C_Final_SpatialSensitivity_Mapping_20km.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 06_greening_hydroclimate/Step_L6.0_FINAL_V2.py | 3.4 | 4.5–4.6 | 8–10 | S6 |
| 06_greening_hydroclimate/empirical_greening.py | 3.4 | 4.5–4.6 | 8–10 | S6 |
| 06_greening_hydroclimate/greening_model.py | 3.4 | 4.5–4.6 | 8–10 | S6 |
| 06_greening_hydroclimate/hydroclimatic_ale.py | 3.4 | 4.5–4.6 | 8–10 | S6 |
| 05_severity_response/m1/model_utils.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 05_severity_response/m1/prepare_cache.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 05_severity_response/m1/historical_fit.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 05_severity_response/m1/independent_breakpoints.py | 3.3 | 4.3–4.4 | 6–7 | S2 |
| 02_extreme_event_detection/m2/catalog_overlap_m2.py | 3.1 | 4.1, 4.3 | 3, 6 | S1–S2 |
| 02_extreme_event_detection/m2/heat_compare_m2.py | 3.1 | 4.1, 4.3 | 3, 6 | S1–S2 |
| 02_extreme_event_detection/m2/audit_phase_m2.py | 3.1 | 4.1, 4.3 | 3, 6 | S1–S2 |
| 07_robustness_analysis/m3/m3_extract.py | 3.2, 3.4 | 4.2, 4.5 | 4, 9 | S4, S6 |
| 07_robustness_analysis/m3/m3_greening.py | 3.2, 3.4 | 4.2, 4.5 | 4, 9 | S4, S6 |
| 07_robustness_analysis/m3/m3_strict_sync.py | 3.2, 3.4 | 4.2, 4.5 | 4, 9 | S4, S6 |
| 07_robustness_analysis/m3/m3_trajectory.py | 3.2, 3.4 | 4.2, 4.5 | 4, 9 | S4, S6 |
| 07_robustness_analysis/m3/m3_summarize.py | 3.2, 3.4 | 4.2, 4.5 | 4, 9 | S4, S6 |
| 07_robustness_analysis/m4/m4_prepare.py | 3.3–3.4 | 4.4–4.5 | 7, 9 | S6B |
| 07_robustness_analysis/m4/m4_cv.py | 3.3–3.4 | 4.4–4.5 | 7, 9 | S6B |
| 07_robustness_analysis/m4/m4_nested_full.py | 3.3–3.4 | 4.4–4.5 | 7, 9 | S6B |
| 07_robustness_analysis/m4/m4_bins.py | 3.3–3.4 | 4.4–4.5 | 7, 9 | S6B |
| 04_event_lifecycle_recovery/m5/prepare_m5.py | 3.3 | 4.2–4.3 | 4–6 | S2–S5, Fig. S1 |
| 04_event_lifecycle_recovery/m5/extract_m5.py | 3.3 | 4.2–4.3 | 4–6 | S2–S5, Fig. S1 |
| 04_event_lifecycle_recovery/m5/survival_m5.py | 3.3 | 4.2–4.3 | 4–6 | S2–S5, Fig. S1 |
| 04_event_lifecycle_recovery/m5/analyse_m5.py | 3.3 | 4.2–4.3 | 4–6 | S2–S5, Fig. S1 |
| 07_robustness_analysis/m6/m6_prepare.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m6/m6_extract.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m6/m6_compare.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m6/m6_climate.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m6/m6_breakpoints.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m6/m6_recovery.py | 3.1–3.3 | 4.1–4.3 | 3–6 | S1–S3 |
| 07_robustness_analysis/m8/m8_terrain_models.py | 3.3–3.4 | 4.3–4.6 | 6–10 | S6 |
| 07_robustness_analysis/m8/m8b_landcover.py | 3.3–3.4 | 4.3–4.6 | 6–10 | S6 |
| 07_robustness_analysis/m8/m8b_events.py | 3.3–3.4 | 4.3–4.6 | 6–10 | S6 |
| 07_robustness_analysis/m8/m8b_breakpoints.py | 3.3–3.4 | 4.3–4.6 | 6–10 | S6 |
| 07_robustness_analysis/m8/m8b_greening.py | 3.3–3.4 | 4.3–4.6 | 6–10 | S6 |
| 07_robustness_analysis/single_indicator/minor2_core.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/single_indicator/minor2_prepare_gate.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/single_indicator/minor2_gate_baseline_repair.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/single_indicator/minor2_extract_controls.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/single_indicator/minor2_statistics.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/single_indicator/minor2_severity.py | 3.2–3.3 | 4.3 | 6 | S4A |
| 07_robustness_analysis/components/audit_core.py | 3.2–3.3 | 4.3 | 6 | S4B |
| 07_robustness_analysis/components/extract_components.py | 3.2–3.3 | 4.3 | 6 | S4B |
| 07_robustness_analysis/components/component_statistics.py | 3.2–3.3 | 4.3 | 6 | S4B |
| 07_robustness_analysis/components/recovery_statistics.py | 3.2–3.3 | 4.3 | 6 | S4B |
| 07_robustness_analysis/response_bias/core.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/extract_sync.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/prepare_and_extract.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/statistics_background.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/statistics_legacy.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/statistics_sync.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/common_window_sensitivity.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/repair_stop_ties.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/response_bias/apply_120_support_gate.py | 3.2–3.3 | 4.2–4.3 | 4–6 | S2B, S5, Fig. S2 |
| 07_robustness_analysis/continuous_severity/run_r2.py | 3.3 | 4.3 | 6 | S2–S3 |
| 07_robustness_analysis/hcl/extract_f9.py | 3.4 | 4.5 | 9c | — |
| 01_data_preprocessing/export_climate_calendar.js | 2.2, 3.1 | 4.1 | 1–3 | S1 |
| 08_figures/figure_03.py | 3.1–3.4 | 4.1–4.6 | 3 | main figures |
| 08_figures/figure_04.py | 3.1–3.4 | 4.1–4.6 | 4 | main figures |
| 08_figures/figure_05.py | 3.1–3.4 | 4.1–4.6 | 5 | main figures |
| 08_figures/figure_06.py | 3.1–3.4 | 4.1–4.6 | 6 | main figures |
| 08_figures/figure_07.py | 3.1–3.4 | 4.1–4.6 | 7 | main figures |
| 08_figures/figure_08.py | 3.1–3.4 | 4.1–4.6 | 8 | main figures |
| 08_figures/figure_09.py | 3.1–3.4 | 4.1–4.6 | 9 | main figures |
| 08_figures/figure_10.py | 3.1–3.4 | 4.1–4.6 | 10 | main figures |

Figure entries identify archived rendering implementations. Exact identity with the submitted artwork remains unverified. No verified production renderer was located for Figures 1–2 or Supplementary Figures S1–S2. For S1 see survival_m5 and analyse_m5. For S2 see common_window_sensitivity.

Additional implementations: export_land_cover.js supplies the three-date land-cover stratification (Methods 2.2, 3.4, Table S6). refine_paired_fits.py supports the paired response diagnostics (Methods 3.3, Results 4.3).
