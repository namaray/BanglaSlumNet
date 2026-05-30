/**
 * GEE Script: Generate weak labels via OSM ∩ GHSL ∩ VIIRS fusion.
 * Exports per-tile noisy label mask and 3-signal HC mask.
 *
 * Rules (§6.1):
 *   slum pixel        = OSM residential ∩ GHSL built-up ∩ VIIRS dark  (≤ city median)
 *   formal-dense pixel= OSM residential ∩ GHSL built-up ∩ VIIRS bright (> city median)
 *   unknown           = anything else
 *
 * TODO_VERIFY: All region bounding boxes.
 * TODO_VERIFY: Confirm GHSL and VIIRS dataset IDs are current in GEE catalog.
 */

var DRIVE_FOLDER  = 'BanglaSlumNet/data/labels';
var SCALE_M       = 10;
var REFERENCE_YEAR = 2022;  // use a mid-range year for the weak label mosaic

// ── Regions (TODO_VERIFY) ─────────────────────────────────────────────────────
var regions = {
  korail:            ee.Geometry.Rectangle([90.4110, 23.7830, 90.4220, 23.7930]),
  bhashantek:        ee.Geometry.Rectangle([90.3790, 23.8330, 90.3920, 23.8450]),
  karail_extension:  ee.Geometry.Rectangle([90.4180, 23.7900, 90.4290, 23.7990]),
  old_dhaka:         ee.Geometry.Rectangle([90.3940, 23.7090, 90.4160, 23.7260]),
  gulshan_baridhara: ee.Geometry.Rectangle([90.4100, 23.7760, 90.4340, 23.8000])
};

var dhakaCity = ee.Geometry.Rectangle([90.3400, 23.6800, 90.5000, 23.9000]);

// ── GHSL built-up layer ───────────────────────────────────────────────────────
// GHS-BUILT-S2 or GHS-SMOD; using GHS-BUILT at 10m
var ghsl = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
             .select('built_surface')
             .gt(0)
             .rename('ghsl_builtup');

// ── OSM residential (via LSIB/OpenStreetMap proxy in GEE) ────────────────────
// GEE does not have a direct OSM layer; use the Dynamic World "built" class as proxy.
// TODO_VERIFY: Replace with a proper OSM-residential raster if available via GEE assets.
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
           .filterDate(ee.Date.fromYMD(REFERENCE_YEAR, 1, 1),
                       ee.Date.fromYMD(REFERENCE_YEAR, 12, 31))
           .filterBounds(dhakaCity)
           .select('label')
           .mode();
var osm_residential = dw.eq(6).rename('osm_residential');  // class 6 = built area

// ── VIIRS nighttime lights ────────────────────────────────────────────────────
var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
              .filterDate(ee.Date.fromYMD(REFERENCE_YEAR, 1, 1),
                          ee.Date.fromYMD(REFERENCE_YEAR, 12, 31))
              .filterBounds(dhakaCity)
              .select('avg_rad')
              .median()
              .rename('viirs_ntl');

// Compute city-median threshold
var cityMedian = viirs.reduceRegion({
  reducer: ee.Reducer.percentile([50]),
  geometry: dhakaCity,
  scale: 500,
  maxPixels: 1e9
}).getNumber('viirs_ntl_p50');

var viirs_dark   = viirs.lt(cityMedian).rename('viirs_dark');
var viirs_bright = viirs.gte(cityMedian).rename('viirs_bright');

// ── Per-pixel signal agreement score ─────────────────────────────────────────
// score = number of signals agreeing (0–3)
var slum_signals   = ghsl.add(osm_residential).add(viirs_dark);    // 0–3
var formal_signals = ghsl.add(osm_residential).add(viirs_bright);  // 0–3

// Class encoding: 0=unknown, 1=slum, 2=formal-dense
var noisy_label = ee.Image(0)
  .where(slum_signals.gte(2),   1)  // ≥2-signal agreement → slum
  .where(formal_signals.gte(2), 2)  // ≥2-signal agreement → formal-dense
  .rename('noisy_label')
  .toByte();

// HC mask: all 3 signals must agree (4-signal HC comes after LocateAnything in Python)
var hc_slum   = slum_signals.eq(3).rename('hc_slum').toByte();
var hc_formal = formal_signals.eq(3).rename('hc_formal').toByte();
var hc_mask   = hc_slum.add(hc_formal.multiply(2)).rename('hc_mask').toByte();

var agreement_score = slum_signals.max(formal_signals).rename('agreement_score').toByte();

// ── Export per region ──────────────────────────────────────────────────────────
Object.keys(regions).forEach(function(regionName) {
  var geom = regions[regionName];
  var composite = noisy_label.addBands(hc_mask).addBands(agreement_score);

  Export.image.toDrive({
    image: composite,
    description: 'weeklabels_' + regionName,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'weeklabels_' + regionName,
    region: geom,
    scale: SCALE_M,
    crs: 'EPSG:4326',
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
  print('Queued weak labels for:', regionName);
});
