import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier

# reading the sub-trajectories
traj_stats = pd.read_csv("sub_trajectories.csv", index_col=0)


# binarizing the classes fro making hierarchical classifier
on_road = []
on_foot = []
on_track = []
for i in traj_stats.index:
    if traj_stats.Class[i]=="train" or traj_stats.Class[i]=="subway":
        on_track.append(1)
    else:
        on_track.append(0)
    print i

    if traj_stats.Class[i]=="walk":
        on_foot.append(1)
    else:
        on_foot.append(0)

    if traj_stats.Class[i]=="car" or traj_stats.Class[i]=="bus" or traj_stats.Class[i]=="taxi":
        on_road.append(1)
    else:
        on_road.append(0)

on_road = pd.DataFrame(on_road)
on_foot= pd.DataFrame(on_foot)
on_track = pd.DataFrame(on_track)

# node 1 of the herirarchy will classify the input instance into foot or non_foot classes. where on foot(walk) is 1
# and other non_foot trajectories as 0
X_train, X_test, y_train, y_test = train_test_split(traj_stats.iloc[:,3:], on_foot, test_size= 0.33, random_state=10)
clf_node1 = RandomForestClassifier()
clf_node1.fit(X_train, y_train)

# node 2 will determine if the trajectory if of a road vehicle or track one.
# road_or_track contains a 0 for every non-track transportation (on road) ie 0 for car||bus||taxi while
# 1 for train||subway
road_or_track = on_foot[0][on_foot[0]==0]
road_or_track.loc[on_track[0][on_track[0]==1].index]=1
train_data = traj_stats.loc[road_or_track.index]

X_train, X_test1, y_train, y_test1 = train_test_split(train_data.iloc[:,3:], road_or_track, test_size= 0.33,
                                                      random_state=10)
clf_node2 = RandomForestClassifier()
clf_node2.fit(X_train, y_train)

# node 3
# what_on_road contains three classes taxi, bus and car as 0, 1 and 2 respectively
what_on_road = road_or_track.loc[road_or_track==0]
what_on_road.loc[traj_stats["Class"][traj_stats["Class"]=="bus"].index]=1
what_on_road.loc[traj_stats["Class"][traj_stats["Class"]=="car"].index]=2
train_data = traj_stats.loc[what_on_road.index]

X_train, X_test1, y_train, y_test1 = train_test_split(train_data.iloc[:,3:], what_on_road, test_size= 0.33,
                                                      random_state=10)
clf_node3 = RandomForestClassifier()
clf_node3.fit(X_train, y_train)

# node 4
# what_on_track contains two classes train and subway as 0 and 1 respectively
what_on_track = road_or_track.loc[road_or_track==1]
what_on_track.loc[traj_stats["Class"][traj_stats["Class"]=="train"].index]=0
what_on_track.loc[traj_stats["Class"][traj_stats["Class"]=="subway"].index]=1
train_data = traj_stats.loc[what_on_track.index]

X_train, X_test1, y_train, y_test1 = train_test_split(train_data.iloc[:,3:], what_on_track, test_size= 0.33, random_state=10)
clf_node4 = RandomForestClassifier()
clf_node4.fit(X_train, y_train)


# implementation of the hierarchy
def predict_class(test_data):
    # node1_result = clf_node1.predict(test_tuple)
    predicted = []
    for i in range(0,len(test_data)):
        if clf_node1.predict(test_data[i:i+1])==1:
            print "The guys was walking!!"
            predicted.append("walk")
            continue
        elif clf_node2.predict(test_data[i:i+1])==1:
            if clf_node4.predict(test_data[i:i+1])==0:
                print "The guy was in a train!"
                predicted.append("train")
                continue
            else:
                print "The guy was in a subway!"
                predicted.append("subway")
                continue
        else:
            node3_results = clf_node3.predict(test_data[i:i+1])
            if node3_results==0:
                print "The guy was in a taxi!"
                predicted.append("taxi")
                continue
            elif node3_results==1:
                print "The guy was in a bus!"
                predicted.append("bus")
                continue
            else:
                print "The guy was in a Car!"
                predicted.append("car")
                continue
    return predicted


predictions = predict_class(X_test)
true = traj_stats.loc[X_test.index]["Class"]

print " Accuracy score: ", accuracy_score(true, predictions)

