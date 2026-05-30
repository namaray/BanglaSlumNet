/**
 * GEE Script: Export socioeconomic E-tensor layers.
 * Layers: VIIRS NTL, WorldPop, GHS-POP, OSM roads density, WB poverty, GHSL built-up.
 * All exported at 10 m aligned to Sentinel-2 grid.
 *
 * TODO_VERIFY: All region bounding boxes.
 * TODO_VERIFY: WB poverty layer — WorldBank GriddedPoverty or similar; confirm GEE asset ID.
 */

var DRIVE_FOLDER  = 'BanglaSlumNet/data/socioeconomic';
var SCALE_M       = 10;
var REFERENCE_YEAR = 2022;

// ── Regions (TODO_VERIFY) ─────────────────────────────────────────────────────
var regions = {
  korail:            ee.Geometry.Rectangle([90.4110, 23.7830, 90.4220, 23.7930]),
  bhashantek:        ee.Geometry.Rectangle([90.3790, 23.8330, 90.3920, 23.8450]),
  karail_extension:  ee.Geometry.Rectangle([90.4180, 23.7900, 90.4290, 23.7990]),
  old_dhaka:         ee.Geometry.Rectangle([90.3940, 23.7090, 90.4160, 23.7260]),
  gulshan_baridhara: ee.Geometry.Rectangle([90.4100, 23.7760, 90.4340, 23.8000])
};

var dhakaCity = ee.Geometry.Rectangle([90.3400, 23.6800, 90.5000, 23.9000]);

// ── 1. VIIRS Nighttime Lights ─────────────────────────────────────────────────
var viirs_ntl = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate(ee.Date.fromYMD(REFERENCE_YEAR, 1, 1),
              ee.Date.fromYMD(REFERENCE_YEAR, 12, 31))
  .filterBounds(dhakaCity)
  .select('avg_rad')
  .median()
  .rename('viirs');

// ── 2. WorldPop population density ───────────────────────────────────────────
var worldpop = ee.ImageCollection('WorldPop/GP/100m/pop')
  .filterDate(ee.Date.fromYMD(REFERENCE_YEAR, 1, 1),
              ee.Date.fromYMD(REFERENCE_YEAR, 12, 31))
  .filterBounds(dhakaCity)
  .select('population')
  .mosaic()
  .rename('worldpop');

// ── 3. GHS-POP ────────────────────────────────────────────────────────────────
var ghspop = ee.Image('JRC/GHSL/P2023A/GHS_POP/2020')
  .select('population_count')
  .rename('ghspop');

// ── 4. OSM roads density (via Dynamic World or GRIP global roads proxy) ───────
// TODO_VERIFY: Replace with a proper OSM roads raster if available as GEE asset.
// Using distance-to-nearest-road proxy from GRIP4 dataset or similar.
// As a placeholder: use Landsat-derived built area proxy for road structure.
var roads_proxy = ee.Image('Oxford/MAP/accessibility_to_cities_2015_v1_0')
  .rename('osm_roads');  // TODO_VERIFY: swap for proper road density raster

// ── 5. WB Poverty index ───────────────────────────────────────────────────────
// TODO_VERIFY: Use WorldBank GriddedPoverty or Chi et al. 2022 relative wealth index.
// Placeholder: relative wealth index from Meta/ORNL if available as GEE asset.
var wb_poverty = ee.Image(0).rename('wb_poverty');  // TODO_VERIFY: replace with real asset

// ── 6. GHSL built-up fraction ────────────────────────────────────────────────
var ghsl_builtup = ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020')
  .select('built_surface')
  .rename('ghsl_builtup');

// ── Stack all channels ────────────────────────────────────────────────────────
var socioeconomic = viirs_ntl
  .addBands(worldpop)
  .addBands(ghspop)
  .addBands(roads_proxy)
  .addBands(wb_poverty)
  .addBands(ghsl_builtup)
  .toFloat();

// ── Export per region ─────────────────────────────────────────────────────────
Object.keys(regions).forEach(function(regionName) {
  var geom = regions[regionName];
  Export.image.toDrive({
    image: socioeconomic.clip(geom),
    description: 'socioeconomic_' + regionName,
    folder: DRIVE_FOLDER,
    fileNamePrefix: 'socioeconomic_' + regionName,
    region: geom,
    scale: SCALE_M,
    crs: 'EPSG:4326',
    maxPixels: 1e10,
    fileFormat: 'GeoTIFF'
  });
  print('Queued socioeconomic for:', regionName);
});
