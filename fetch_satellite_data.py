import json
import ee
import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import folium
from datetime import datetime

# Authenticate and initialize Earth Engine
ee.Authenticate()
ee.Initialize()


# degrees minutes seconds coordinates to decimal
def dms_to_decimal(coordinate_tuple, direction):
    if len(coordinate_tuple) == 1:
        degrees = coordinate_tuple[0]
        minutes = 0
        seconds = 0
    elif len(coordinate_tuple) == 2:
        degrees, minutes = coordinate_tuple
        seconds = 0
    elif len(coordinate_tuple) == 3:
        degrees, minutes, seconds = coordinate_tuple
    else:
        raise ValueError("Invalid coordinate tuple. Must be of length 1, 2, or 3.")
    dd = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if direction in ['S', 'W']:  # South and West are negative
        dd *= -1
    return dd


def get_current_satellite_image(latitude, longitude, radius=1, date=datetime.now().strftime("%Y-%m-%d")):
    latitude = dms_to_decimal(latitude)
    longitude = dms_to_decimal(longitude)


    # Calculate the bounding box coordinates
    top_left = geodesic(kilometers=radius).destination((latitude, longitude), bearing=315)  # NW
    top_right = geodesic(kilometers=radius).destination((latitude, longitude), bearing=45)  # NE
    bottom_right = geodesic(kilometers=radius).destination((latitude, longitude), bearing=135)  # SE
    bottom_left = geodesic(kilometers=radius).destination((latitude, longitude), bearing=225)  # SW


    corner_points = [
        (top_left.longitude, top_left.latitude),  # Top-left
        (top_right.longitude, top_right.latitude), # Top-right
        (bottom_right.longitude, bottom_left.latitude), # Bottom-right
        (top_left.longitude, top_left.latitude)   # Bottom-left
    ]

    # Ensure polygon closes by repeating the first point at the end
    corner_points.append(corner_points[0])


    # Define Area of Interest (AOI)
    aoi = ee.Geometry.Point([corner_points[0][1], corner_points[0][0]]).buffer(1000).bounds()


    # Load Sentinel-2 imagery 
    sentinel2_image = ee.ImageCollection("COPERNICUS/S2") \
        .filterBounds(aoi) \
        .filterDate(date) \
        .median()  

    # Select Bands 3 (Green) and 4 (Red)
    green_band = sentinel2_image.select('B3')  # Green band (560 nm)
    red_band = sentinel2_image.select('B4')    # Red band (665 nm)

    # Calculate Chlorophyll Index (CI)
    chlorophyll_index = green_band.subtract(red_band).divide(green_band.add(red_band))

    # Visualize Chlorophyll Index
    chlorophyll_index_vis = chlorophyll_index.clip(aoi)
    chlorophyll_index_map = chlorophyll_index_vis.getMapId()



    # Define visualization parameters
    vis_params = {
        'min': -0.5,
        'max': 0.5,
        'palette': ['blue', 'green', 'yellow', 'red']
    }

    # Get map visualization ID and token for Chlorophyll Index (CI)
    map_id = chlorophyll_index.getMapId(vis_params)

    # Create a folium map to display the result
    my_map = folium.Map(location=[corner_points[0][0], corner_points[0][1]], zoom_start=12)
    folium.TileLayer(
        tiles=map_id['tileUrl'],
        attr='Google Earth Engine',
        overlay=True,
        control=True
    ).add_to(my_map)

    return my_map

