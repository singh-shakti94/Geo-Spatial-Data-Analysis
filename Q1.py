from re import sub

import numpy as np
import pandas as pd
from haversine import haversine
from datetime import datetime
from geohelper import bearing

data = pd.read_csv("geolife_raw.csv")
sub_data = data
# sub_data = sub_data.append(data[634576:634676])
sub_data = sub_data.set_index([range(0, len(sub_data))])

# sub_data = sub_data.append(data[586807:586907])

# s = data.collected_time[1]
# a,b = s.split(" ")

# data["collected_date"] = ""
# for i in range(0, 1000):
#     data.collected_time[i] = data.collected_time[i][:-3]

# grouped_data = data.groupby("t_user_id")
#
# latitudes_first = grouped_data.latitude.first()
# latitudes_first = latitudes_first.unique()
# latitudes_last = grouped_data.latitude.last()
# longitudes_first = grouped_data.longitude.first()
# longitudes_last = grouped_data.longitude.last()
np.set_printoptions(suppress=True)

# convert to date format
# for i in range(0, len(sub_data)):
import timeit

distances = np.array([])
time = np.array([])
speed = np.array([])
bearing_values = []
x = np.array([])
x = np.append(x, datetime.strptime(sub_data.collected_time[0], "%Y-%m-%d %H:%M:%S-%f"))
start = timeit.default_timer()
for i in range(0, len(sub_data)-1):
    print i
    x = np.append(x, datetime.strptime(sub_data.collected_time[i+1], "%Y-%m-%d %H:%M:%S-%f"))
    distances = np.append(distances,haversine((sub_data.latitude[i], sub_data.longitude[i]),
                               (sub_data.latitude[i+1], sub_data.longitude[i+1]))*1000)
    # extract time
    # np.set_printoptions(suppress=True)
    time = np.append(time, (x[i+1] - x[i]).total_seconds())

    # calculate speed
    speed = np.append(speed, distances[i]/time[i])

    # calculate bearing
    bearing_values.append(bearing.initial_compass_bearing(sub_data.latitude[i], sub_data.longitude[i],
                                                          sub_data.latitude[i+1], sub_data.longitude[i+1]))
end = timeit.default_timer()-start
print end

sub_data.collected_time = x
# # calculate acceleration
# acceleration = np.array([])
# acceleration = np.append(acceleration, np.nan)
# for i in range(0, len(speed)-1):
#     acceleration = np.append(acceleration, (speed[i+1]-speed[i])/time[i])


Trajectories = pd.DataFrame({"distance": distances,
                             "time": time,
                             "speed": speed,
                             "bearing": bearing_values})


Trajectories.index.name = "Traj."
Trajectories.set_index([range(1,len(Trajectories)+1)])

Trajectories["id"] = sub_data["t_user_id"]
Trajectories["day"] = sub_data.collected_time.dt.day
Trajectories["Class"] = sub_data.transportation_mode

Trajectories.to_csv(path_or_buf="Trajectories.csv")

# for i in range(0, len(Trajectories)):
#     if Trajectories.acceleration[i] >= 7:
#         Trajectories = Trajectories.drop(i)
# Trajectories = Trajectories.set_index([range(0, len(Trajectories))])
#
#
# for i in range(0, len(Trajectories)):
#     if Trajectories.distance[i] >= 1000:
#         Trajectories = Trajectories.drop(i)
# Trajectories = Trajectories.set_index([range(0, len(Trajectories))])
#
# for i in range(0, len(Trajectories)):
#     if Trajectories.time[i] >= 500:
#         Trajectories = Trajectories.drop(i)
# Trajectories = Trajectories.set_index([range(0, len(Trajectories))])
#
#
# for i in range(0, len(Trajectories)):
#     if Trajectories.Class[i] == "walk" and Trajectories.speed[i] >= 8:
#         Trajectories = Trajectories.drop(i)
# Trajectories = Trajectories.set_index([range(0, len(Trajectories))])
#
# grouped_trajectories = Trajectories.groupby(["id", "day", "Class"])
# grouped_trajectories.mean()
#
#
# total_distance = grouped_trajectories.distance.sum()
# total_time = grouped_trajectories.time.sum()
# total_speed = total_distance/total_time
#
# sub_trajectories = pd.DataFrame({"max_distance":grouped_trajectories.distance.max()})
# sub_trajectories = sub_trajectories.reset_index(level=["day", "Class"])
#
