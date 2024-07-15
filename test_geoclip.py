import math
import time

import torch

import read_write

# Import GeoCLIP class from GeoCLIP.py (assuming it's in the same folder)
from GeoCLIP import GeoCLIP

start_time = time.time()
# Initialize the model
model = GeoCLIP(from_pretrained=True)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

print(torch.cuda.is_available())
print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))

model.ready_gps_gallery = model.gps_gallery.to(model.device)


end_time = time.time()
load_time = end_time - start_time

print(f"Model loading took {load_time:.2f} seconds\n")


def geoclipmodel(image_path):

    texts = [""]
    top_k = 1  # Number of top predictions to return
    top_pred_gps, top_pred_prob = model.predict(image_path, top_k, texts=texts)

    return top_pred_gps.tolist()[0]


def calculate_spherical_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the Earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371

    # Calculate the result
    return c * r


# MAIN FUNCTION

image_names = read_write.read_column_from_csv("reviews.csv", "image_path")
real_lats = read_write.read_column_from_csv("reviews.csv", "lat")
real_lons = read_write.read_column_from_csv("reviews.csv", "lon")

lat_coordinates = []
lon_coordinates = []
distances = []
total_time = 0

for i, image in enumerate(image_names, 1):
    start_time = time.time()

    coordinates = geoclipmodel(image)
    lat, lon = coordinates[0], coordinates[1]
    lat_coordinates.append(lat)
    lon_coordinates.append(lon)

    dist = calculate_spherical_distance(
        float(real_lats[i - 1]), float(real_lons[i - 1]), lat, lon
    )
    distances.append(dist)

    end_time = time.time()
    iteration_time = end_time - start_time
    total_time += iteration_time

    print(f"Iteration {i} took {iteration_time:.2f} seconds")
    print(f"Average time per iteration: {total_time/i:.2f} seconds")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print("--------------------")

read_write.add_column_to_csv("reviews.csv", "pred_lat", lat_coordinates)
read_write.add_column_to_csv("reviews.csv", "pred_lon", lon_coordinates)
read_write.add_column_to_csv("reviews.csv", "spherical_distance", distances)


print(f"Total time for all iterations: {total_time:.2f} seconds")
print(
    f"Average time per iteration: {total_time/len(image_names):.2f} seconds\n\nTASK COMPLETE"
)
