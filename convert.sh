#!/bin/bash

# Make sure you have ogr2ogr, tippecanoe, and pmtiles installed
# This assumes your script output is 'IMD_Domains_2010-2025_England_Trend_Hotspots.geojson'

# 1. Define your input and output filenames
INPUT_GEOJSON="IMD_Domains_2010-2025_England_Trend_Hotspots.geojson"
TEMP_MBTILES="deprivation.mbtiles"
FINAL_PMTILES="deprivation.pmtiles"


# 2. Re-project BNG to WGS84, then pipe to Tippecanoe
echo "Starting OGR2OGR re-projection and piping to Tippecanoe..."
ogr2ogr -t_srs EPSG:4326 -f GeoJSON /dev/stdout $INPUT_GEOJSON | \
tippecanoe -o $TEMP_MBTILES \
    -f \
    -zg \
    -S 10 \
    -pk \
    --no-tiny-polygon-reduction \
    --coalesce-smallest-as-needed \
    --extend-zooms-if-still-dropping \
    -l deprivation && \

# 3. Convert the MBTiles to PMTiles
echo "Tippecanoe finished. Converting to PMTiles..."
pmtiles convert $TEMP_MBTILES $FINAL_PMTILES && \

# 4. Clean up the intermediate file
rm $TEMP_MBTILES

echo "✅ Success! $FINAL_PMTILES has been created."