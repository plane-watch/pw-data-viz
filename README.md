# Plane Watch - Data Visualisation

## Collecting Data from Clickhouse
Sample query:
```
SELECT * FROM plane_watch.location_updates_low WHERE LastMsg >=
  toDateTime('2025-07-01 00:00:00') AND LastMsg < toDateTime('2025-07-02 00:00:00') AND HasLocation
   = true AND geoDistance(tupleElement(LatLon, 1), tupleElement(LatLon, 2), -37.67333, 144.84333) <= 60000 ORDER BY LastMsg INTO OUTFILE '20250701-ymml-60km.csv.gz2' FORMAT csv
```
This will save the output data to a csv.gz file we can pass to the script. 

## Running the script
This will output the visualisation to `aircraft_paths.png`:
```
uv run aircraft_path_visualizer.py 20250701-yssy-65km.csv.gz --high-res --format png --size large
```


