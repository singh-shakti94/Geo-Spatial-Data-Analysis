import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the trajectories
trajectories = pd.read_csv("Trajectories.csv")

# cleaning the trajectory data
trajectories = trajectories[trajectories.acceleration < 8]
trajectories = trajectories[trajectories.acceleration > -8]
trajectories = trajectories[trajectories.distance < 100]
trajectories = trajectories[trajectories.time > 0]

trajectories = trajectories.reset_index()

#grouping the trajectories according to id, day and class ( making Sub-trajectories)
trajectories_grouped = trajectories.groupby(["id", "day", "Class"])

# Calculating the point features of the sub-trajectories
traj_stats = pd.DataFrame({"max_acceleration":trajectories_grouped.max()["acceleration"]})
traj_stats["max_speed(mps)"] = trajectories_grouped.max()["speed"].values
traj_stats["max_distance(m)"] = trajectories_grouped.max()["distance"].values
traj_stats["max_bearing"] = trajectories_grouped.max()["bearing"].values

traj_stats["min_speed(mps)"] = trajectories_grouped.min()["speed"].values
traj_stats["min_distance(m)"] = trajectories_grouped.min()["distance"].values
traj_stats["min_bearing"] = trajectories_grouped.min()["bearing"].values
traj_stats["min_acceleration"] = trajectories_grouped.min()["acceleration"].values

traj_stats["mean_speed(mps)"] = trajectories_grouped.mean()["speed"].values
traj_stats["mean_distance"] = trajectories_grouped.mean()["distance"].values
traj_stats["mean_bearing"] = trajectories_grouped.mean()["bearing"].values
traj_stats["mean_acceleration"] = trajectories_grouped.mean()["acceleration"].values

traj_stats["median_speed"] = trajectories_grouped.median()["speed"].values
traj_stats["median_distance"] = trajectories_grouped.median()["distance"].values
traj_stats["median_bearing"] = trajectories_grouped.median()["bearing"].values
traj_stats["median_acceleration"] = trajectories_grouped.median()["acceleration"].values

traj_stats["std_speed"] = trajectories_grouped.std()["speed"].values
traj_stats["std_distance"] = trajectories_grouped.std()["distance"].values
traj_stats["std_bearing"] = trajectories_grouped.std()["bearing"].values
traj_stats["std_acceleration"] = trajectories_grouped.std()["acceleration"].values

# removing sub-trajectories with less than 10 trajectories
count = trajectories_grouped.count()[trajectories_grouped.index.count()>10]
index1 = traj_stats.index
index2 = count.index
index_difference = index1.difference(index2)
traj_stats = traj_stats.drop(index_difference)

# moving the classes from index to the dataframe
traj_stats = traj_stats.reset_index(level=['id', 'Class', "day"])

# remove unwanted classes
for i in range(0,len(traj_stats)-1):
    if traj_stats.Class[i]=="run" or traj_stats.Class[i]=="motorcycle":
        traj_stats = traj_stats.drop(i)

traj_stats = traj_stats.reset_index()
traj_stats = traj_stats.drop("index", axis=1)

# writing the sub-trajectories into a csv
traj_stats.to_csv(path_or_buf="sub_trajectories.csv")

# ---------------------------------------------------Data Exploration------------------------------------------------

#box plots

#mean speed
sns.set_style("whitegrid")
ax = sns.boxplot(data=traj_stats, x = "Class", y="mean_speed(mps)")
plt.savefig("mean_speed.png")
plt.clf()

# max acceleartion
ax = sns.boxplot(data=traj_stats, x = "Class", y="max_acceleration")
ax = sns.swarmplot(data=traj_stats, x = "Class", y="max_acceleration", size=2, color=".3")
plt.savefig("max_acceleration.png")
plt.clf()

# mean acceleartion
ax = sns.boxplot(data=traj_stats, x = "Class", y="mean_acceleration")
# ax = sns.swarmplot(data=traj_stats, x = "Class", y="mean_acceleration", size=2, color=".3")
plt.savefig("mean_acceleration.png")
plt.clf()


ax = sns.boxplot(data=traj_stats, x = "Class", y="min_acceleration")
ax = sns.swarmplot(data=traj_stats, x = "Class", y="min_acceleration", size=2, color=".3")
plt.savefig("min_acceleration.png")
plt.clf()

ax = sns.boxplot(data=traj_stats, x = "Class", y="std_acceleration")
# ax = sns.swarmplot(data=traj_stats, x = "Class", y="mean_acceleration", size=2, color=".3")
plt.savefig("std_acceleration.png")
plt.clf()


# sns.distplot(traj_stats["mean_speed(mps)"][traj_stats.Class=="train"], hist=False, color="g", kde_kws={"shade": True})