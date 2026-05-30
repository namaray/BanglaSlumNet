/**
 * GEE Script: Export Sentinel-2 seasonal composites for BanglaSlumNet.
 * Paste into GEE Code Editor (code.earthengine.google.com) and run.
 * Exports land in your Google Drive under the folder set by DRIVE_FOLDER.
 *
 * TODO_VERIFY: Confirm all region bounding boxes before running.
 * TODO_VERIFY: Confirm DRIVE_FOLDER path matches what BanglaSlumNet_Colab.ipynb expects.
 */

// ── Configuration ─────────────────────────────────────────────────────────────
var DRIVE_FOLDER = 'BanglaSlumNet/data/tiles';
var SCALE_M = 10;                    // Sentinel-2 native resolution
var TILE_SIZE_PX = 256;
var TRAINING_YEARS = [2020, 2021, 2022, 2023];
var BANDS = ['B2', 'B3', 'B4', 'B8'];   // + B11, B12 if D7 enabled

// Dry season: Dec–Feb; Wet season: Jun–Aug (Dhaka)
var SEASONS = {
  dry:  { start_month: 12, end_month: 2 },
  wet:  { start_month: 6,  end_month: 8 }
};

// ── Regions (TODO_VERIFY all coordinates) ─────────────────────────────────────
var regions = {
  korail: ee.Geometry.Rectangle([90.4110, 23.7830, 90.4220, 23.7930]),            // TODO_VERIFY
  bhashantek: ee.Geometry.Rectangle([90.3790, 23.8330, 90.3920, 23.8450]),        // TODO_VERIFY
  karail_extension: ee.Geometry.Rectangle([90.4180, 23.7900, 90.4290, 23.7990]), // TODO_VERIFY
  old_dhaka: ee.Geometry.Rectangle([90.3940, 23.7090, 90.4160, 23.7260]),        // TODO_VERIFY
  gulshan_baridhara: ee.Geometry.Rectangle([90.4100, 23.7760, 90.4340, 23.8000]) // TODO_VERIFY
};

// ── Helper: cloud-masked Sentinel-2 collection ────────────────────────────────
function getS2Collection(region, startDate, endDate) {
  return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(region)
    .filterDate(startDate, endDate)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .map(function(img) {
      var scl = img.select('SCL');
      var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
      return img.updateMask(mask).divide(10000).select(BANDS);
    });
}

// ── Helper: best-pixel composite (median) ─────────────────────────────────────
function makeComposite(region, year, seasonName, seasonDef) {
  var startYear = (seasonName === 'dry' && seasonDef.start_month > seasonDef.end_month)
                  ? year - 1 : year;
  var startDate = ee.Date.fromYMD(startYear, seasonDef.start_month, 1);
  var endDate   = ee.Date.fromYMD(year, seasonDef.end_month, 28);

  var col = getS2Collection(region, startDate, endDate);
  return col.median().clip(region);
}

// ── Export loop ───────────────────────────────────────────────────────────────
Object.keys(regions).forEach(function(regionName) {
  var geom = regions[regionName];
  TRAINING_YEARS.forEach(function(year) {
    Object.keys(SEASONS).forEach(function(seasonName) {
      var composite = makeComposite(geom, year, seasonName, SEASONS[seasonName]);
      var taskName = 's2_' + regionName + '_' + year + '_' + seasonName;
      Export.image.toDrive({
        image: composite.toFloat(),
        description: taskName,
        folder: DRIVE_FOLDER,
        fileNamePrefix: taskName,
        region: geom,
        scale: SCALE_M,
        crs: 'EPSG:4326',
        maxPixels: 1e10,
        fileFormat: 'GeoTIFF'
      });
      print('Queued:', taskName);
    });
  });
});
