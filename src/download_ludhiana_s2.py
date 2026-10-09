import pystac_client
import planetary_computer
import odc.stac
import rioxarray
import xarray as xr

# Bounding box for Ludhiana, Punjab region approximately
ludhiana_bbox = [75.75, 30.85, 76.00, 31.00] 

catalog = pystac_client.Client.open(
    "[https://planetarycomputer.microsoft.com/api/stac/v1](https://planetarycomputer.microsoft.com/api/stac/v1)",
    modifier=planetary_computer.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=ludhiana_bbox,
    datetime="2025-02-01/2025-04-30", # Clear spring window
    query={"eo:cloud_cover": {"lt": 5}}
)

items = list(search.items())
print(f"Found {len(items)} matching Sentinel-2 tiles.")

if len(items) > 0:
    item = items[0]
    print(f"Downloading tile ID: {item.id}")
    
    # Load bands: B04 (Red), B03 (Green), B02 (Blue), B08 (NIR)
    ds = odc.stac.load(
        [item],
        bands=["red", "green", "blue", "nir"],
        resolution=10,
        bbox=ludhiana_bbox
    )
    
    os.makedirs("data/ludhiana", exist_ok=True)
    ds.to_netcdf("data/ludhiana/ludhiana_s2_tile.nc")
    print("Saved Ludhiana Sentinel-2 subset successfully.")