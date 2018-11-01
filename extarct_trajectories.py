import numpy as np
import pandas as pd
from haversine import haversine
from datetime import datetime
from geohelper import bearing
import timeit

# import the geolife data
data = pd.read_csv("geolife_raw.csv")
sub_data = data
sub_data = sub_data.set_index([range(0, len(sub_data))])

np.set_printoptions(suppress=True)

# Extraction of the trajectories form geolife data
distances = []
time = []
speed = []
bearing_values = []
x = []
x.append(datetime.strptime(sub_data.collected_time[0], "%Y-%m-%d %H:%M:%S-%f"))
start = timeit.default_timer()
for i in range(0, len(sub_data)-1):
    print i
    x.append(datetime.strptime(sub_data.collected_time[i+1], "%Y-%m-%d %H:%M:%S-%f"))
    distances.append(haversine((sub_data.latitude[i], sub_data.longitude[i]),
                                (sub_data.latitude[i+1], sub_data.longitude[i+1]))*1000)
    # extract time
    # np.set_printoptions(suppress=True)
    time.append((x[i+1] - x[i]).total_seconds())

    # calculate speed
    if time[i] == 0:
        speed.append(0)
    else:
        speed.append(distances[i]/time[i])

    # calculate bearing
    bearing_values.append(bearing.initial_compass_bearing(sub_data.latitude[i], sub_data.longitude[i],
                                                          sub_data.latitude[i+1], sub_data.longitude[i+1]))
end = timeit.default_timer()-start
print end

# aggregating the collected trajectory features in a data frame
Trajectories = pd.DataFrame({"distance": distances,
                             "time": time,
                             "speed": speed,
                             "bearing": bearing_values})

# Calculating acceleration
acceleration = []
acceleration.append(np.nan)
Trajectories.index.name = "Traj."
Trajectories.set_index([range(1,len(Trajectories)+1)])
Trajectories["id"] = sub_data["t_user_id"]
# Trajectories["day"] = 0
x = []
for i in range(0,len(sub_data)-1):
    x.append(sub_data.collected_time[i][0:10])
    if i == len(sub_data)-1:
        continue
    acceleration.append((Trajectories.speed[i + 1] - Trajectories.speed[i]) / Trajectories.time[i])
    print i

Trajectories["day"] = x
Trajectories["Class"] = sub_data.transportation_mode

Trajectories["acceleration"] = acceleration

# writing all the extracted trajectories in a csv
Trajectories.to_csv(path_or_buf="Trajectories.csv")
