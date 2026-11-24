/**
 * ==============================================================================
 * @Project     : Weihe River Basin Ecological Resilience
 * @Module      : Level III Hydroclimatic Stress
 * @Version     : GEE V3.1 L3-ONLY FINAL
 *
 * @Period      : 2000–2024
 * @Resolution  : 1000 m
 * @Output      : One annual multiband GeoTIFF per year
 *
 * @Bands per 8-day phase:
 *   1. Tmax
 *   2. Temp
 *   3. Precip
 *   4. VPD
 *   5. RootSM
 *   6. DeepSM
 *   7. RAD
 *
 * @Key corrections:
 *   - Precip: total_precipitation_hourly -> 8-day sum
 *   - RAD   : surface_solar_radiation_downwards_hourly -> 8-day sum
 *   - VPD   : calculate hourly first -> 8-day mean
 *   - RootSM: thickness-weighted 0–100 cm soil moisture
 *
 * IMPORTANT:
 *   This script exports ONLY Level III.
 *   Existing Level I and Level II files are NOT regenerated.
 * ==============================================================================
 */


// ==============================================================================
// 0. CONFIGURATION
// ==============================================================================


// ------------------------------------------------------------------------------
// Study area
// ------------------------------------------------------------------------------

var weihe = ee.FeatureCollection(
  'REPLACE_WITH_BASIN_ASSET_ID'
);

var roi = weihe.geometry();

Map.centerObject(weihe, 7);

Map.addLayer(
  weihe.style({
    color: 'red',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Weihe Basin'
);


// ------------------------------------------------------------------------------
// Study period
// ------------------------------------------------------------------------------

var START_YEAR = 2000;
var END_YEAR   = 2024;


// ------------------------------------------------------------------------------
// Target export resolution
// ------------------------------------------------------------------------------

var SCALE_L3 = 1000;


// ------------------------------------------------------------------------------
// Export folder
//
// Strong recommendation:
// first export to a NEW folder.
// Do NOT overwrite the old V2 data before QC.
// ------------------------------------------------------------------------------

var EXPORT_FOLDER = 'GEE_Weihe_V3_1_L3';


// ==============================================================================
// 1. DATA SOURCES
// ==============================================================================


// ------------------------------------------------------------------------------
// PML-V2.2a
//
// IMPORTANT:
// We only use PML here as the master 8-day calendar,
// so the new L3 dates remain exactly synchronized with the existing L2 files.
// ------------------------------------------------------------------------------

var pml = ee.ImageCollection(
  'projects/pml_evapotranspiration/PML/OUTPUT/PML_V22a'
).filterBounds(roi);


// ------------------------------------------------------------------------------
// ERA5-Land Hourly
// ------------------------------------------------------------------------------

var era5 = ee.ImageCollection(
  'ECMWF/ERA5_LAND/HOURLY'
).filterBounds(roi);


// ==============================================================================
// 2. HELPER FUNCTIONS
// ==============================================================================


// ------------------------------------------------------------------------------
// 2.1 Saturation vapor pressure
//
// Input : temperature in degrees Celsius
// Output: kPa
//
// Tetens-type formulation:
//
// es = 0.61078 * exp(17.27*T / (T + 237.3))
// ------------------------------------------------------------------------------

function saturationVaporPressure(tempC) {

  return tempC.expression(

    '0.61078 * exp((17.27 * T) / (T + 237.3))',

    {
      'T': tempC
    }

  );

}


// ------------------------------------------------------------------------------
// 2.2 Hourly VPD
//
// VPD = es(Tair) - ea(Tdew)
//
// Output:
// kPa
//
// The calculation is performed hourly FIRST.
// The resulting hourly VPD is then averaged over each 8-day interval.
// ------------------------------------------------------------------------------

function calculateHourlyVPD(image) {

  var airTempC = image
    .select('temperature_2m')
    .subtract(273.15);

  var dewTempC = image
    .select('dewpoint_temperature_2m')
    .subtract(273.15);


  var es = saturationVaporPressure(
    airTempC
  );

  var ea = saturationVaporPressure(
    dewTempC
  );


  var vpd = es
    .subtract(ea)

    // Prevent tiny negative values caused by numerical precision.
    .max(0)

    .rename('VPD_hourly');


  return vpd.copyProperties(
    image,
    ['system:time_start']
  );

}


// ==============================================================================
// 3. CORE 8-DAY L3 PROCESSING FUNCTION
// ==============================================================================

function buildL3ForOnePMLImage(pmlImage) {


  // ============================================================================
  // 3.1 TIME WINDOW
  // ============================================================================

  var tStart = pmlImage.date();

  var tEnd = tStart.advance(
    8,
    'day'
  );


  // ---------------------------------------------------------------------------
  // Exact source date
  //
  // Example:
  // 2000-03-05
  // ---------------------------------------------------------------------------

  var dateString = tStart.format(
    'YYYY-MM-dd'
  );


  // ---------------------------------------------------------------------------
  // MMDD suffix
  //
  // Example:
  // 0305
  // ---------------------------------------------------------------------------

  var mmdd = tStart.format(
    'MMdd'
  );


  // ---------------------------------------------------------------------------
  // Formal band name
  //
  // Example:
  // 2000-03-05_Tmax_0305
  //
  // This is deliberately compatible with your current Python metadata parser.
  // ---------------------------------------------------------------------------

  function bandName(variableName) {

    return dateString
      .cat('_')
      .cat(variableName)
      .cat('_')
      .cat(mmdd);

  }


  // ============================================================================
  // 3.2 ERA5-LAND 8-DAY COLLECTION
  // ============================================================================

  var era5Window = era5.filterDate(
    tStart,
    tEnd
  );


  // ============================================================================
  // 3.3 TEMPERATURE
  // ============================================================================


  // ---------------------------------------------------------------------------
  // Convert hourly temperature from Kelvin to Celsius
  // ---------------------------------------------------------------------------

  var temperatureHourlyC = era5Window
    .select('temperature_2m')
    .map(function(image) {

      return image
        .subtract(273.15)

        .copyProperties(
          image,
          ['system:time_start']
        );

    });


  // ---------------------------------------------------------------------------
  // Tmax
  //
  // Maximum hourly 2-m air temperature during the 8-day interval.
  //
  // Unit:
  // deg C
  // ---------------------------------------------------------------------------

  var tMax = temperatureHourlyC
    .max()
    .rename(
      bandName('Tmax')
    );


  // ---------------------------------------------------------------------------
  // Temp
  //
  // Mean hourly 2-m air temperature during the 8-day interval.
  //
  // Unit:
  // deg C
  // ---------------------------------------------------------------------------

  var tMean = temperatureHourlyC
    .mean()
    .rename(
      bandName('Temp')
    );


  // ============================================================================
  // 3.4 PRECIPITATION
  //
  // IMPORTANT V3.1 CORRECTION
  //
  // DO NOT use:
  //
  // total_precipitation
  //
  // because it is an accumulated forecast field.
  //
  // Use:
  //
  // total_precipitation_hourly
  //
  // which represents the disaggregated hourly value.
  //
  // Original unit:
  // m water equivalent
  //
  // After *1000:
  // mm
  //
  // After hourly sum:
  // mm / 8-day
  // ============================================================================

  var precip = era5Window

    .select(
      'total_precipitation_hourly'
    )

    .sum()

    .multiply(
      1000.0
    )

    .rename(
      bandName('Precip')
    );


  // ============================================================================
  // 3.5 VPD
  //
  // Hourly VPD -> 8-day mean
  //
  // Unit:
  // kPa
  // ============================================================================

  var vpd = era5Window

    .map(
      calculateHourlyVPD
    )

    .mean()

    .rename(
      bandName('VPD')
    );


  // ============================================================================
  // 3.6 SOIL MOISTURE
  // ============================================================================


  // ---------------------------------------------------------------------------
  // Mean hourly soil moisture over the 8-day interval
  // ---------------------------------------------------------------------------

  var soilMoistureMean = era5Window.mean();


  // ---------------------------------------------------------------------------
  // RootSM
  //
  // ERA5-Land layers:
  //
  // layer 1 :   0–7 cm     thickness =  7 cm
  // layer 2 :   7–28 cm    thickness = 21 cm
  // layer 3 :  28–100 cm   thickness = 72 cm
  //
  // Total depth = 100 cm
  //
  // Thickness weights:
  //
  // 0.07
  // 0.21
  // 0.72
  //
  // RootSM therefore represents:
  //
  // thickness-weighted mean volumetric soil moisture over 0–100 cm
  //
  // Unit:
  // m3 m-3
  //
  // NOTE:
  // It is a root-zone proxy.
  // It is NOT vegetation-specific root-distribution weighting.
  // ---------------------------------------------------------------------------

  var rootSM = soilMoistureMean

    .select(
      'volumetric_soil_water_layer_1'
    )

    .multiply(
      0.07
    )

    .add(

      soilMoistureMean
        .select(
          'volumetric_soil_water_layer_2'
        )
        .multiply(
          0.21
        )

    )

    .add(

      soilMoistureMean
        .select(
          'volumetric_soil_water_layer_3'
        )
        .multiply(
          0.72
        )

    )

    .rename(
      bandName('RootSM')
    );


  // ---------------------------------------------------------------------------
  // DeepSM
  //
  // ERA5-Land layer 4:
  //
  // 100–289 cm
  //
  // Unit:
  // m3 m-3
  // ---------------------------------------------------------------------------

  var deepSM = soilMoistureMean

    .select(
      'volumetric_soil_water_layer_4'
    )

    .rename(
      bandName('DeepSM')
    );


  // ============================================================================
  // 3.7 SOLAR RADIATION
  //
  // IMPORTANT V3.1 CORRECTION
  //
  // DO NOT use:
  //
  // surface_solar_radiation_downwards
  //
  // Use:
  //
  // surface_solar_radiation_downwards_hourly
  //
  // Original hourly unit:
  // J m-2
  //
  // Sum over 8 days:
  // J m-2 / 8-day
  //
  // Divide by 1e6:
  // MJ m-2 / 8-day
  // ============================================================================

  var rad = era5Window

    .select(
      'surface_solar_radiation_downwards_hourly'
    )

    .sum()

    .divide(
      1e6
    )

    .rename(
      bandName('RAD')
    );


  // ============================================================================
  // 3.8 BUILD FINAL L3 STACK
  //
  // CRITICAL:
  // DO NOT change this band order unless Python is changed accordingly.
  // ============================================================================

  var l3Image = ee.Image.cat([

    tMax,

    tMean,

    precip,

    vpd,

    rootSM,

    deepSM,

    rad

  ])


  // ---------------------------------------------------------------------------
  // Standardize numeric type
  // ---------------------------------------------------------------------------

  .toFloat()


  // ---------------------------------------------------------------------------
  // Store original 8-day date metadata
  // ---------------------------------------------------------------------------

  .set({

    'system:time_start':
      tStart.millis(),

    'date_start':
      dateString,

    'date_end':
      tEnd.format('YYYY-MM-dd'),

    'temporal_resolution':
      '8-day',

    'Tmax_unit':
      'degC',

    'Temp_unit':
      'degC',

    'Precip_unit':
      'mm_per_8day',

    'VPD_unit':
      'kPa',

    'RootSM_unit':
      'm3_m-3',

    'DeepSM_unit':
      'm3_m-3',

    'RAD_unit':
      'MJ_m-2_per_8day',

    'RootSM_definition':
      '0-100cm thickness-weighted mean: 0.07*SM1 + 0.21*SM2 + 0.72*SM3'

  });


  return l3Image;

}


// ==============================================================================
// 4. OPTIONAL SINGLE-WINDOW QC
//
// This section does NOT export anything.
// It checks whether the corrected variables have physically plausible values.
// ==============================================================================

var qcStart = ee.Date(
  '2018-07-01'
);

var qcEnd = qcStart.advance(
  8,
  'day'
);

var qcEra = era5.filterDate(
  qcStart,
  qcEnd
);


// ------------------------------------------------------------------------------
// QC precipitation
// ------------------------------------------------------------------------------

var qcPrecip = qcEra

  .select(
    'total_precipitation_hourly'
  )

  .sum()

  .multiply(
    1000
  )

  .rename(
    'Precip'
  );


// ------------------------------------------------------------------------------
// QC radiation
// ------------------------------------------------------------------------------

var qcRad = qcEra

  .select(
    'surface_solar_radiation_downwards_hourly'
  )

  .sum()

  .divide(
    1e6
  )

  .rename(
    'RAD'
  );


// ------------------------------------------------------------------------------
// QC VPD
// ------------------------------------------------------------------------------

var qcVPD = qcEra

  .map(
    calculateHourlyVPD
  )

  .mean()

  .rename(
    'VPD'
  );


// ------------------------------------------------------------------------------
// Print basin-wide QC values
//
// Use mean / min / max so gross scaling problems can be detected immediately.
// ------------------------------------------------------------------------------

print(
  '===== V3.1 QC: 2018-07-01 to 2018-07-09 ====='
);


print(

  'Precip (mm/8day):',

  qcPrecip.reduceRegion({

    reducer:
      ee.Reducer.mean()
        .combine({
          reducer2: ee.Reducer.minMax(),
          sharedInputs: true
        }),

    geometry:
      roi,

    scale:
      10000,

    maxPixels:
      1e9,

    bestEffort:
      true

  })

);


print(

  'RAD (MJ m-2 / 8day):',

  qcRad.reduceRegion({

    reducer:
      ee.Reducer.mean()
        .combine({
          reducer2: ee.Reducer.minMax(),
          sharedInputs: true
        }),

    geometry:
      roi,

    scale:
      10000,

    maxPixels:
      1e9,

    bestEffort:
      true

  })

);


print(

  'VPD (kPa):',

  qcVPD.reduceRegion({

    reducer:
      ee.Reducer.mean()
        .combine({
          reducer2: ee.Reducer.minMax(),
          sharedInputs: true
        }),

    geometry:
      roi,

    scale:
      10000,

    maxPixels:
      1e9,

    bestEffort:
      true

  })

);


// ==============================================================================
// 5. BUILD 2000–2024 YEARLY L3 EXPORT TASKS
// ==============================================================================

var years = ee.List.sequence(
  START_YEAR,
  END_YEAR
);


// ------------------------------------------------------------------------------
// Convert server-side year list to client-side list so Export tasks can be made.
// ------------------------------------------------------------------------------

years.evaluate(function(yearList) {


  yearList.forEach(function(yearValue) {


    // ==========================================================================
    // 5.1 YEAR RANGE
    // ==========================================================================

    var year = Number(
      yearValue
    );


    var yearStart = ee.Date.fromYMD(
      year,
      1,
      1
    );


    // IMPORTANT:
    // filterDate uses [start, end)
    //
    // therefore use Jan 1 of the next year.
    // ==========================================================================

    var yearEnd = yearStart.advance(
      1,
      'year'
    );


    // ==========================================================================
    // 5.2 PML MASTER 8-DAY TIMELINE
    //
    // This guarantees L3 is generated at the exact same 8-day dates as your
    // existing Level-II PML-based dataset.
    // ==========================================================================

    var pmlYear = pml

      .filterDate(
        yearStart,
        yearEnd
      )

      .sort(
        'system:time_start'
      );


    // ==========================================================================
    // 5.3 CHECK NUMBER OF 8-DAY PHASES
    // ==========================================================================

    pmlYear.size().evaluate(function(count) {


      if (count === 0) {

        print(
          '⚠️ SKIP ' +
          year +
          ': no PML images available.'
        );

        return;

      }


      print(
        'Year ' +
        year +
        ': ' +
        count +
        ' master 8-day phases.'
      );


      // ------------------------------------------------------------------------
      // Expected:
      //
      // 2000 -> 38
      // 2001-2024 -> normally 46
      //
      // Your metadata audit already confirmed 2000 begins on 2000-03-05.
      // ------------------------------------------------------------------------

      if (year === 2000 && count !== 38) {

        print(
          '⚠️ WARNING: 2000 phase count is ' +
          count +
          ', expected 38 from previous metadata QC.'
        );

      }


      if (year >= 2001 && count !== 46) {

        print(
          '⚠️ WARNING: ' +
          year +
          ' has ' +
          count +
          ' phases; expected 46.'
        );

      }


      // ==========================================================================
      // 5.4 BUILD L3 8-DAY IMAGE COLLECTION
      // ==========================================================================

      var l3Collection = pmlYear.map(
        buildL3ForOnePMLImage
      );


      // ==========================================================================
      // 5.5 CONVERT TO ANNUAL MULTIBAND IMAGE
      //
      // Band sequence will be:
      //
      // date1_Tmax
      // date1_Temp
      // date1_Precip
      // date1_VPD
      // date1_RootSM
      // date1_DeepSM
      // date1_RAD
      //
      // date2_Tmax
      // ...
      // ==========================================================================

      var l3Yearly = l3Collection

        .toBands()

        .clip(
          roi
        )

        .toFloat();


      // ==========================================================================
      // 5.6 EXPORT
      // ==========================================================================

      Export.image.toDrive({

        image:
          l3Yearly,

        description:
          'Weihe_L3_ClimateStress_1000m_' +
          year,

        folder:
          EXPORT_FOLDER,

        fileNamePrefix:
          'Weihe_L3_ClimateStress_1000m_' +
          year,

        region:
          roi,

        scale:
          SCALE_L3,

        crs:
          'EPSG:4326',

        maxPixels:
          1e13,

        fileFormat:
          'GeoTIFF',

        formatOptions: {

          cloudOptimized:
            true

        }

      });


      print(
        '✅ L3 export task created: ' +
        year
      );


    });


  });


});


// ==============================================================================
// 6. OPTIONAL MAP QC
// ==============================================================================


// ------------------------------------------------------------------------------
// Corrected 8-day precipitation
// ------------------------------------------------------------------------------

Map.addLayer(

  qcPrecip.clip(roi),

  {
    min: 0,
    max: 100
  },

  'QC Precip 2018-07-01 to 07-09',

  false

);


// ------------------------------------------------------------------------------
// Corrected 8-day radiation
// ------------------------------------------------------------------------------

Map.addLayer(

  qcRad.clip(roi),

  {
    min: 50,
    max: 250
  },

  'QC RAD 2018-07-01 to 07-09',

  false

);


// ------------------------------------------------------------------------------
// Corrected mean VPD
// ------------------------------------------------------------------------------

Map.addLayer(

  qcVPD.clip(roi),

  {
    min: 0,
    max: 3
  },

  'QC VPD 2018-07-01 to 07-09',

  false

);


print(
  '============================================================'
);

print(
  'GEE V3.1 L3-ONLY pipeline initialized successfully.'
);

print(
  'Output folder: ' +
  EXPORT_FOLDER
);

print(
  'Only Level III annual GeoTIFFs will be exported.'
);

print(
  'Existing L1 and L2 data are NOT exported or modified.'
);

print(
  '============================================================'
);