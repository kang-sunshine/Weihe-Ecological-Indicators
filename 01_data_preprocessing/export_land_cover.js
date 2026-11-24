// M8 minimal pending export: only3 annual landcover tasks, NO time series.
// NOT EXECUTED LOCALLY: no earthengine-api package or user credential file.
// Open in authenticated Earth Engine Code Editor with an enabled Cloud project.
// Collection required by user: MODIS/061/MCD12Q1, LC_Type1, IGBP1..17.
// Official reference: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1
// No categorical bilinear interpolation. Aggregate native one-hot masks by
// overlap-weighted mean onto EXACT frozen1km grid. Fractions include missing
// area (sum=valid_fraction); mode is maximum fraction, ties lowestIGBPcode.
var roi = ee.FeatureCollection('REPLACE_WITH_BASIN_ASSET_ID').geometry();
var targetCRS='EPSG:4326',targetTransform=[0.008983152841196329, 0.0, 103.97101098399341, 0.0, -0.008983152841196329, 37.41483158357807];
var folder='EI_R1_M8_MINIMAL_LC';
var years=[2001,2010,2020];
years.forEach(function(y){
 var images=ee.ImageCollection('MODIS/061/MCD12Q1').filterDate(y+'-01-01',(y+1)+'-01-01');
 print('Input count',y,images.size());
 var lc=ee.Image(images.first()).select('LC_Type1');
 var valid=lc.gte(1).and(lc.lte(17)).unmask(0);
 var fr=[];
 for(var c=1;c<=17;c++){
  fr.push(lc.eq(c).unmask(0).toFloat().rename('igbp_'+c+'_fraction')
   .reduceResolution({reducer:ee.Reducer.mean(),maxPixels:128,bestEffort:false})
   .reproject({crs:targetCRS,crsTransform:targetTransform}));
 }
 var fractions=ee.Image.cat(fr);
 var vf=fractions.reduce(ee.Reducer.sum()).rename('valid_fraction');
 var mode=fractions.toArray().arrayArgmax().arrayGet([0]).add(1)
   .where(vf.eq(0),0).rename('igbp_mode');
 var purity=fractions.reduce(ee.Reducer.max()).divide(vf)
   .where(vf.eq(0),0).rename('purity_among_valid');
 var missing=ee.Image.constant(1).subtract(vf).rename('missing_fraction');
 var out=mode.addBands(fractions).addBands(vf).addBands(purity).addBands(missing)
  .toFloat().clip(roi).unmask(-9999,false);
 Export.image.toDrive({image:out,description:'M8_LC_'+y+'_IGBP_Fractions_1km',
  fileNamePrefix:'M8_LC_'+y+'_IGBP_Fractions_1km',folder:folder,
  region:roi.bounds(10),crs:targetCRS,crsTransform:targetTransform,
  maxPixels:2e7,fileFormat:'GeoTIFF',formatOptions:{cloudOptimized:true,noData:-9999}});
});
// Code creates3 tasks; author must start/download them. No claim of download.
// Keep8/9savannas,14mosaic,11wetland,13urban,15snow,16barren,17water separate.
// Forest1..5; Grassland/Shrubland6,7,10; pureCropland12. Class14 inclusion only
// as a labelled sensitivity. LC12 does NOT identify irrigation management.
