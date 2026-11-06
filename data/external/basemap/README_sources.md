# Base-map data for Fig. 1 (`scripts/32_location_map.py fetch`)

Downloaded once; `build` uses only these files (offline). Large rasters are git-ignored; this README and `sources.json` are committed. Re-run `fetch` to re-download (it skips files already recorded).

## Copernicus DEM GLO-30 (DGED), COG tile N50_00_E002_00

- Provider: ESA / Copernicus (AWS Open Data Registry)
- Version: tile as served on the AWS Open Data bucket on the download date (release not stated in the file name)
- File: `data/external/basemap/copdem_glo30_N50_00_E002_00.tif`
- URL: https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N50_00_E002_00_DEM/Copernicus_DSM_COG_10_N50_00_E002_00_DEM.tif
- Request parameters: none (whole tile)
- Download date: 2026-10-10
- SHA256: `b68e759f8d16723b146f6ffd27b263beb8403c080b0cf488f768bd6418a3482c`
- Licence: Copernicus DEM licence (free, worldwide; attribution required)
- Required attribution: Copernicus DEM GLO-30, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA

## Copernicus DEM GLO-30 (DGED), COG tile N51_00_E002_00

- Provider: ESA / Copernicus (AWS Open Data Registry)
- Version: tile as served on the AWS Open Data bucket on the download date (release not stated in the file name)
- File: `data/external/basemap/copdem_glo30_N51_00_E002_00.tif`
- URL: https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N51_00_E002_00_DEM/Copernicus_DSM_COG_10_N51_00_E002_00_DEM.tif
- Request parameters: none (whole tile)
- Download date: 2026-10-10
- SHA256: `f9187190ed5f1d8e69decf4c576afc6b5458807543da03d914920d24612afea5`
- Licence: Copernicus DEM licence (free, worldwide; attribution required)
- Required attribution: Copernicus DEM GLO-30, © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA

## EMODnet Digital Bathymetry (DTM 2024), mean depth, coverage emodnet__mean

- Provider: EMODnet Bathymetry Consortium
- Version: DTM 2024 (metadata record cf51df64-56f9-4a99-b1aa-36b8d7b743a1)
- File: `data/external/basemap/emodnet_dtm2024_mean.tif`
- URL: https://ows.emodnet-bathymetry.eu/wcs?service=WCS&version=2.0.1&request=GetCoverage&coverageId=emodnet__mean&format=image%2Ftiff&subset=Lat%2850.95%2C51.3%29&subset=Long%282.15%2C2.95%29
- Request parameters: WCS 2.0.1 GetCoverage, coverageId=emodnet__mean, format=image/tiff, subset=Lat(50.95,51.3), subset=Long(2.15,2.95)
- Download date: 2026-10-10
- SHA256: `80768ee65dc8223b388279ab2ca4fcd66a611d1a084cf2c2d3d95f480ca99ec7`
- Licence: CC-BY 4.0
- Required attribution: EMODnet Bathymetry Consortium (2024): EMODnet Digital Bathymetry (DTM 2024), https://doi.org/10.12770/cf51df64-56f9-4a99-b1aa-36b8d7b743a1
- Vertical datum: Lowest Astronomical Tide (LAT)

## Orthofotomozaïek, middenschalig, winteropnamen, kleur, 2025, Vlaanderen (layer OMWRGB25VL; the most recent winter mosaic in the service — the brief's layer name OMWRGBMRVL is not offered)

- Provider: Digitaal Vlaanderen (Informatie Vlaanderen)
- Version: 2025 winter flights; ground resolution 15 cm (capabilities abstract)
- File: `data/external/basemap/ortho_OMWRGB25VL_0.25m.png`
- URL: https://geo.api.vlaanderen.be/OMW/wms
- Request parameters: WMS 1.3.0 GetMap, LAYERS=OMWRGB25VL, CRS=EPSG:31370, BBOX=22690.0,199560.0,23430.0,200300.0 (requested in 2×2 tiles, stitched), 0.25 m/pixel, FORMAT=image/png
- Download date: 2026-10-10
- SHA256: `2ba983e26bbb6cfe08f31523f9a1584b9c6045fa106de35cfae6128a50a4cc52`
- Licence: Modellicentie Gratis Hergebruik (Flemish free re-use licence)
- Required attribution: Orthophoto: Digitaal Vlaanderen (winter mosaic 2025)

## ne_10m_admin_0_boundary_lines_land

- Provider: Natural Earth
- Version: 1:10m (as fetched by cartopy 0.26.0, run_all step NE)
- File: `data/external/natural_earth/ne_10m_admin_0_boundary_lines_land.shp`
- URL: https://www.naturalearthdata.com
- Request parameters: see data/external/natural_earth/README.md
- Download date: (earlier round)
- SHA256: `b6811b9e2f6ed40b31622e6d261210c5e6afded9b7ad71977d3658293f56373f`
- Licence: public domain
- Required attribution: Made with Natural Earth

## ne_10m_coastline

- Provider: Natural Earth
- Version: 1:10m (as fetched by cartopy 0.26.0, run_all step NE)
- File: `data/external/natural_earth/ne_10m_coastline.shp`
- URL: https://www.naturalearthdata.com
- Request parameters: see data/external/natural_earth/README.md
- Download date: (earlier round)
- SHA256: `459a4a97c09db19aadf5244026612de9d43748be27f83a360242b99f7fabb3c1`
- Licence: public domain
- Required attribution: Made with Natural Earth

## ne_10m_land

- Provider: Natural Earth
- Version: 1:10m (as fetched by cartopy 0.26.0, run_all step NE)
- File: `data/external/natural_earth/ne_10m_land.shp`
- URL: https://www.naturalearthdata.com
- Request parameters: see data/external/natural_earth/README.md
- Download date: (earlier round)
- SHA256: `4cad3a49bc75c1a4c2f3d7efae04f2f8e63151c96764b2658effabf524331fa6`
- Licence: public domain
- Required attribution: Made with Natural Earth
