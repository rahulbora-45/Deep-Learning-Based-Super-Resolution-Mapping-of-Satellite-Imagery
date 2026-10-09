import os
import osmnx as ox
import geopandas as gpd

os.makedirs("data/masks", exist_ok=True)

print("Fetching OpenStreetMap roads and buildings for Ludhiana...")
place_name = "Ludhiana, Punjab, India"

# Download buildings and roads via OSMnx
buildings = ox.geometries_from_place(place_name, tags={"building": True})
roads = ox.geometries_from_place(place_name, tags={"highway": True})

buildings.to_file("data/masks/osm_buildings.geojson", driver="GeoJSON")
roads.to_file("data/masks/osm_roads.geojson", driver="GeoJSON")
print("OSM vector masks saved successfully.")