import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from collections import Counter
from scipy.signal import argrelextrema
from SimpleRNN import train_SimpleRNN
from sklearn.model_selection import train_test_split
import pandas as pd
from get_data import get_data
from sklearn.preprocessing import MinMaxScaler
from SimpleRNN_Dense import train_SimpleRNN_Dense
from LSTM import train_LSTM
from GRU import train_GRU


def calculate_distances(pos) -> list:
    """Calculates the distances of the Nan values

    param pos: the indexes of the Nan values
    :return: a list of tuples with the index of the Nan value and the smallest distance to the next index
    """
    distances = [(pos[0], pos[1] - pos[0])]
    for i in range(1, len(pos) - 1):
        tem_dis = min(pos[i] - pos[i - 1], pos[i + 1] - pos[i])
        distances.append((pos[i], tem_dis))

    distances.append((pos[len(pos) - 1], pos[len(pos) - 1] - pos[len(pos) - 2]))
    return distances


def appearences_of_distances(distance: list) -> dict:
    """Creates a dictionary who´s keys are distances and
        values are how often the keys appear in the data

    param distance: list of tuples with index and the smallest distance of their neighbors
    :return: a dictionary
    """
    dic_distance = {}
    for dis in distance:
        if dis[1] in dic_distance:
            dic_distance[dis[1]] += 1
        else:
            dic_distance[dis[1]] = 1
    return dic_distance


def get_cluster(distance: list) -> dict:
    """Creates a dictionary of distances as key and
    list of indexes who have the same distance as values

    param distance: list of tuples with indexes and distances
    :return: dictionary with all cluster
    """
    cluster = {}
    for dist in distance:
        if dist[1] in cluster:
            cluster[dist[1]].append(dist[0])
        else:
            cluster[dist[1]] = [dist[0]]
    return cluster


def get_split_cluster(cluster: list, for_distance: int) -> list:
    """Creates a list containing lists which have the indexes who are apart a specific distance

    param cluster: list of indexes with the specified distance
    param distance: distance of the indexes
    :return: a list of sublists who the indexes have a specified distance
    """
    split_cluster = [[cluster[0]]]
    index_split = 0
    for index in range(1, len(cluster)):
        if index == len(cluster) - 1:
            if cluster[index] - cluster[index - 1] == for_distance:
                split_cluster[index_split].append(cluster[index])
                break
            else:
                split_cluster.append([cluster[index]])
                break
        if cluster[index] - cluster[index - 1] == for_distance:
            split_cluster[index_split].append(cluster[index])
        else:
            split_cluster.append([cluster[index]])
            index_split += 1
    return split_cluster


def months_peaks(months: list, dataset) -> list:
    # Tuple in list with month and amount of peaks
    max_overflow = []
    for month in months:
        month_n = dataset[dataset['Month'] == month]
        # add nan values
        amount = month_n['max'].value_counts()
        max_overflow.append(max(amount.index))
    return max_overflow


def group_by_max(dataset):
    new_dataset = pd.DataFrame({'Day': [], 'Q': []})
    list_Q = []
    day_of_year = [num for num in range(1, 366)]
    for month in range(1, 13):
        month_year = dataset[dataset['Month'] == month]
        for day in range(1, month_year['Day'].iloc[len(month_year.index) - 1] + 1):
            temp_day_of_year = month_year[month_year['Day'] == day]
            list_Q.append(temp_day_of_year.Q.max())
    new_dataset['Day'] = day_of_year
    new_dataset['Q'] = list_Q
    return new_dataset


def to_sequences(dataset, seq_size=1):
    x = []
    y = []

    for i in range(len(dataset) - seq_size - 1):
        # print(i)
        window = dataset[i:(i + seq_size), 0]
        x.append(window)
        y.append(dataset[i + seq_size, 0])

    return np.array(x), np.array(y)


def main():
    # print(tf.config.list_physical_devices('GPU'))
    # Fragen wie ich es machen soll mit dem doppelten Code
    filepath_pr_hourly = r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\pr_hourly_DWD_ID1550.dat'
    filepath_Q_hourly = r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\Q_hourly_ID16425004.dat'
    try:
        pr_hourly = pd.read_csv(
            r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\pr_hourly_DWD_ID1550.csv')
        Q_hourly = pd.read_csv(
            r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\Q_hourly_ID16425004.csv')
    except FileNotFoundError:
        pr_hourly = get_data(filepath_pr_hourly, '\t')
        Q_hourly = get_data(filepath_Q_hourly, ',')
        pr_hourly.to_csv(r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\pr_hourly_DWD_ID1550.csv',
                         index=False)
        Q_hourly.to_csv(r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\Q_hourly_ID16425004.csv',
                        index=False)

    # how many na values are there => there are no missing data in overflow
    # PR needs closer analysis, but seems valid 0.0039 percent missing
    naQ_count = Q_hourly['Q'].isna().sum()
    naPR_count = pr_hourly['PR'].isna().sum()
    percentage_na = naPR_count / pr_hourly.shape[0]

    # look if the data is complete -> data is complete, +1 because difference does not count the last day
    begin_Q = datetime(day=1, month=11, year=1920)
    end_Q = datetime(day=23, month=2, year=2020)
    diff_Q = end_Q - begin_Q
    # print((diff_Q.days + 1) * 24)
    # print(Q_hourly.shape[0])
    begin_pr = datetime(day=1, month=9, year=1995)
    end_pr = datetime(day=31, month=12, year=2019)
    diff_pr = end_pr - begin_pr
    # print((diff_pr.days + 1) * 24)
    # print(pr_hourly.shape[0])

    # where are the nan values. Are their many at the same time
    nan_values_dataPR = pr_hourly.loc[pr_hourly['PR'].isna()]
    # a lot of cluster of nan values
    pos = nan_values_dataPR.index
    plt.scatter([num for num in range(0, 851)], pos, marker='x', linewidths=0.1)
    plt.title('Grouping of Nan values')
    plt.ylabel('Index of Nan values')
    plt.savefig('Pictures/Dataanalyse/Grouping_nan_values.png', bbox_inches='tight', dpi=150)
    plt.show()

    # differences of ids to find cluster
    distance = calculate_distances(pos)
    # appearence of distance -> 526 nan values are only one id away
    dic_distance = appearences_of_distances(distance)
    # counts how often distances are appearing in the dictionary
    dis_count = Counter(dic_distance)
    # getting the 15 most common appearances in the data -> 15 to see the data better
    highest_counts = dis_count.most_common(15)
    # extracting common distance and the amount for plotting
    distance_common = [count[0] for count in highest_counts]
    amount_distance = [count[1] for count in highest_counts]
    plt.bar(distance_common, amount_distance)
    plt.title('How often a distance appeared')
    plt.xlabel('Distance to next Nan value')
    plt.ylabel('Count of Appearances')
    plt.savefig('Pictures/Dataanalyse/appearance_of_distance.png', bbox_inches='tight', dpi=150)
    plt.show()

    # get cluster
    cluster = get_cluster(distance)
    # check that the ids in the cluster are sorted -> cluster key distance between id, value ids
    for key in cluster:
        test_list = cluster[key]
        if not all(test_list[i] <= test_list[i + 1] for i in range(len(test_list) - 1)):
            print('Lists are not sorted')
    plt.scatter([num for num in range(0, len(cluster[1]))], cluster[1], marker='x', linewidths=0.1)
    plt.title('Cluster of Nan values which are one apart')
    plt.ylabel('Index of Nan value')
    plt.savefig('Pictures/Dataanalyse/Cluster_one_apart.png', bbox_inches='tight', dpi=150)
    plt.show()

    # get subcluster from the cluster with distance 1
    # for_distance = 1
    # split_cluster = get_split_cluster(cluster[for_distance], for_distance)
    # cluster with a specific distance, the longest list
    # print(split_cluster, max(map(len, split_cluster)))

    # delete all cluster with len(4) or greater
    # cluster_with_greater_length4 = [cluster for cluster in split_cluster if len(cluster) > 3]
    # for cluster in cluster_with_greater_length4:
    #    pr_hourly.drop(cluster, axis=0, inplace=True)

    # interpolate cluster with 3 or less Nan values
    # pr_hourly['PR'] = pr_hourly['PR'].interpolate(method='linear', limit=3)
    # pr_hourly_interpolate = pr_hourly

    pr_hourly_interpolate = pd.read_csv(
        r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\pr_hourly_interpolate_DWD_ID1550.csv')

    # round the numbers of the dataset with 2 decimal numbers
    # decimal = 2
    # pr_hourly_interpolate["PR"] = pr_hourly_interpolate["PR"].apply(lambda x: round(x, decimal))

    # overflow is the same, because no data was missing
    # mean  and standard deviation from precipitation before and after interpolation
    # have insignificant differences of 0.001 rounded
    mean_Q = round(Q_hourly['Q'].mean(), 4)
    std_Q = round(Q_hourly['Q'].std(), 4)
    mean_pr = round(pr_hourly['PR'].mean(), 4)
    std_pr = round(pr_hourly['PR'].std(), 4)
    mean_interpolate = round(pr_hourly_interpolate['PR'].mean(), 4)
    std_interpolate = round(pr_hourly_interpolate['PR'].std(), 4)
    # print(mean_Q, std_Q, mean_pr, std_pr)
    # print(mean_interpolate)
    # print(std_interpolate)

    # just one time to save the interpolated data
    # pr_hourly_interpolate.to_csv(
    #    r'C:\Users\Marcel\LRZ Sync+Share\Studium\Bachelorarbeit\Daten\Daten\pr_hourly_interpolate_DWD_ID1550.csv',
    #    index=False)

    #########################################
    # Plots for overflow

    # Plot aller overflow über die gesamte Zeit
    plot1 = Q_hourly.plot(x='Year', y='Q')
    plot1.grid(True)
    plt.title('Overflow from 1920 to 2020')
    plt.ylabel('Overflow in m^3/s')
    plt.show()

    # overflowdifference in 2015, last value deleted, because their is no next value to make a difference
    ndata3 = Q_hourly.loc[Q_hourly['Year'] == 2015]
    ndata3 = group_by_max(ndata3)
    col_Q = np.diff(ndata3['Q'].to_numpy())
    plt.plot(np.array(range(1, 365)), col_Q)
    plt.grid(True)
    plt.ylabel('Overflow in m^3/s')
    plt.title('Overflowdifference in 2015')
    plt.show()

    ndata_2004 = Q_hourly.loc[(Q_hourly['Year'] == 2004) & (Q_hourly['Month'] == 7)]
    ndata_2008 = Q_hourly.loc[(Q_hourly['Year'] == 2008) & (Q_hourly['Month'] == 7)]
    ndata_2010 = Q_hourly.loc[(Q_hourly['Year'] == 2010) & (Q_hourly['Month'] == 7)]
    ndata_2013 = Q_hourly.loc[(Q_hourly['Year'] == 2013) & (Q_hourly['Month'] == 7)]
    plot_2004 = ndata_2004.groupby('Day', as_index=False).Q.mean()
    plot_2008 = ndata_2008.groupby('Day', as_index=False).Q.mean()
    plot_2010 = ndata_2010.groupby('Day', as_index=False).Q.mean()
    plot_2013 = ndata_2013.groupby('Day', as_index=False).Q.mean()
    plt.plot(np.array(range(1, 32)), plot_2004['Q'])
    plt.plot(np.array(range(1, 32)), plot_2008['Q'])
    plt.plot(np.array(range(1, 32)), plot_2010['Q'])
    plt.plot(np.array(range(1, 32)), plot_2013['Q'])
    plt.legend(['July 2004', 'July 2008', 'July 2010', 'July 2013'])
    plt.xlabel('Day of month')
    plt.ylabel('Overflow in m^3/s')
    plt.title('Overflow in July in different years')
    plt.grid(True)
    plt.savefig('Pictures/Dataanalyse/overflow_July', bbox_inches='tight', dpi=150)
    plt.show()

    # peaks of overflow in different months
    n = 75
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    months_name = ['Jan', 'Feb', 'März', 'April', 'Mai', 'Juni', 'Juli', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez']
    ndata4 = Q_hourly.loc[Q_hourly['Year'] == 2015]
    ndata4['min'] = ndata4.iloc[argrelextrema(ndata4.Q.values, np.less_equal, order=n)[0]]['Q']
    ndata4['max'] = ndata4.iloc[argrelextrema(ndata4.Q.values, np.greater_equal, order=n)[0]]['Q']
    ndata7 = Q_hourly.loc[Q_hourly['Year'] >= 1920]
    ndata7['max'] = ndata7.iloc[argrelextrema(ndata7.Q.values, np.greater_equal, order=n)[0]]['Q']
    max_overflow_all = months_peaks(months, ndata7)
    plt.scatter(ndata4.index, ndata4['min'], color='red')
    plt.scatter(ndata4.index, ndata4['max'], color='yellow')
    plt.plot(ndata4.index, ndata4['Q'])
    plt.title('Highs and Lows of overflow in 2015')
    plt.grid(True)
    plt.ylabel('Overflow in m^3/s')
    plt.savefig('Pictures/Dataanalyse/Peaks_Q.png', bbox_inches='tight', dpi=150)
    plt.show()
    plt.bar(months_name, max_overflow_all)
    plt.ylabel('Overflow in m^3/s')
    plt.title('Maximal overflow in every month from 1920 to 2020')
    plt.grid(True)
    plt.savefig('Pictures/Dataanalyse/bar_max_overflow', bbox_inches='tight', dpi=150)
    plt.show()

    ###############################################################
    # Plots for precipitation
    # bar plot of peaks in different years

    # Plot all overflows in the dataset and compare it to interpolate data.
    # mention mean and standard deviation in text of thesis
    pr_hourly.plot(x='Year', y='PR', color='blue', kind='line')
    plt.title('Precipitationdata before Interpolation')
    plt.grid(True)
    plt.show()
    pr_hourly_interpolate.plot(x='Year', y='PR', color='red', kind='line')
    plt.title('Precipitationdata after Interpolation')
    plt.grid(True)
    plt.show()

    pr_hourly_2008 = pr_hourly[pr_hourly['Year'] == 2008]
    pr_hourly_2008 = pr_hourly_2008[pr_hourly_2008['Month'] == 12]
    pr_hourly_2008.plot(x='Day', y='PR', kind='line')
    plt.grid(True)
    plt.show()
    pr_hourly_interpolate_2008 = pr_hourly_interpolate[pr_hourly_interpolate['Year'] == 2008]
    pr_hourly_interpolate_2008 = pr_hourly_interpolate_2008[pr_hourly_interpolate_2008['Month'] == 12]
    pr_hourly_interpolate_2008.plot(x='Day', y='PR', color='red', kind='line')
    plt.grid(True)
    plt.show()

    # precipitationdifference
    ndata = pr_hourly.loc[pr_hourly['Year'] == 2015]
    col_PR = np.diff(ndata['PR'].to_numpy())
    plt.plot(np.array(range(0, 365 * 24 - 1)), col_PR)
    plt.grid(True)
    plt.title('Precipitationdifference in 2015')
    plt.show()

    # peaks of precipitation in different years
    n = 75
    ndata5 = pr_hourly_interpolate.loc[pr_hourly_interpolate['Year'] == 2015]
    ndata5['min'] = ndata5.iloc[argrelextrema(ndata5.PR.values, np.less_equal, order=n)[0]]['PR']
    ndata5['max'] = ndata5.iloc[argrelextrema(ndata5.PR.values, np.greater_equal, order=n)[0]]['PR']
    ndata5['max'] = ndata5['max'].replace(0, np.nan)
    plt.scatter(ndata5.index, ndata5['min'], color='red')
    plt.scatter(ndata5.index, ndata5['max'], color='yellow')
    plt.plot(ndata5.index, ndata5['PR'])
    plt.title('Highs and Lows of precipitation in 2015')
    plt.grid(True)
    # Einheit nachschauen
    plt.ylabel('Precipitation in mm')
    plt.savefig('Pictures/Dataanalyse/Peaks_PR.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Compare of maximal precipitation in all months
    ndata6 = pr_hourly_interpolate[pr_hourly_interpolate['Year'] >= 1995]
    ndata6['max'] = ndata6.iloc[argrelextrema(ndata6.PR.values, np.greater_equal, order=n)[0]]['PR']
    ndata6['max'] = ndata6['max'].replace(0, np.nan)
    max_precipitation_all = months_peaks(months, ndata6)
    plt.bar(months_name, max_precipitation_all)
    plt.ylabel('Precipitation in mm')
    plt.title('Maximal precipitation in every month from 1995 to 2019')
    plt.grid(True)
    plt.savefig('Pictures/Dataanalyse/bar_max_precipitation', bbox_inches='tight', dpi=150)
    plt.show()

    ndata_2004 = pr_hourly_interpolate.loc[
        (pr_hourly_interpolate['Year'] == 2004) & (pr_hourly_interpolate['Month'] == 7)]
    ndata_2008 = pr_hourly_interpolate.loc[
        (pr_hourly_interpolate['Year'] == 2008) & (pr_hourly_interpolate['Month'] == 7)]
    ndata_2010 = pr_hourly_interpolate.loc[
        (pr_hourly_interpolate['Year'] == 2010) & (pr_hourly_interpolate['Month'] == 7)]
    ndata_2013 = pr_hourly_interpolate.loc[
        (pr_hourly_interpolate['Year'] == 2013) & (pr_hourly_interpolate['Month'] == 7)]
    plot_2004 = ndata_2004.groupby('Day', as_index=False).PR.mean()
    plot_2008 = ndata_2008.groupby('Day', as_index=False).PR.mean()
    plot_2010 = ndata_2010.groupby('Day', as_index=False).PR.mean()
    plot_2013 = ndata_2013.groupby('Day', as_index=False).PR.mean()
    plt.plot(np.array(range(1, 32)), plot_2004['PR'])
    plt.plot(np.array(range(1, 32)), plot_2008['PR'])
    plt.plot(np.array(range(1, 32)), plot_2010['PR'])
    plt.plot(np.array(range(1, 32)), plot_2013['PR'])
    plt.legend(['July 2004', 'July 2008', 'July 2010', 'July 2013'])
    plt.xlabel('Day of month')
    plt.ylabel('Precipitation in mm')
    plt.title('precipitation in July in different years')
    plt.grid(True)
    plt.savefig('Pictures/Dataanalyse/precipiatation_July', bbox_inches='tight', dpi=150)
    plt.show()

    ################################
    # Plots who compare Q and PR
    # Plot who compares them, are the peaks from Q and PR in relation to each other

    Q_data = Q_hourly[(Q_hourly['Year'] == 2005) & (Q_hourly['Month'] == 6)]
    PR_data = pr_hourly[(pr_hourly['Year'] == 2005) & (pr_hourly['Month'] == 6)]
    Q_data = Q_data.groupby('Day', as_index=False).Q.max()
    PR_data = PR_data.groupby('Day', as_index=False).PR.max()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Days of month')
    ax1.set_ylabel('Overflow in m^3/s')
    plot_1 = ax1.plot(np.array(range(1, 31)), Q_data['Q'])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation in mm')
    plot_2 = ax2.plot(np.array(range(1, 31)), PR_data['PR'], color='red')
    lns = plot_1 + plot_2
    plt.legend(lns, ['Overflow', 'Precipitation'])
    plt.grid(True)
    plt.title('Maximal Overflow and Precipitation in July 2005')
    plt.savefig('Pictures/Dataanalyse/max_overflow_precipitation_july2005', bbox_inches='tight', dpi=150)
    plt.show()

    Q_data = Q_hourly[(Q_hourly['Year'] == 2010) & (Q_hourly['Month'] == 5)]
    PR_data = pr_hourly[(pr_hourly['Year'] == 2010) & (pr_hourly['Month'] == 5)]
    Q_data = Q_data.groupby('Day', as_index=False).Q.max()
    PR_data = PR_data.groupby('Day', as_index=False).PR.max()
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Days of Month')
    ax1.set_ylabel('Overflow in m^3/s')
    plot_1 = ax1.plot(np.array(range(1, 32)), Q_data['Q'])
    ax2 = ax1.twinx()
    ax2.set_ylabel('Precipitation in mm')
    plot_2 = ax2.plot(np.array(range(1, 32)), PR_data['PR'], color='red')
    lns = plot_1 + plot_2
    plt.grid(True)
    plt.legend(lns, ['Overflow', 'Precipitation'])
    plt.title('Maximal Overflow and Precipitation in May 2010')
    plt.savefig('Pictures/Dataanalyse/max_overflow_precipitation_May2010', bbox_inches='tight', dpi=150)
    plt.show()

    ################################
    # split into training data and test data
    train_data = Q_hourly[Q_hourly['Year'] <= 1990]
    test_data = Q_hourly[Q_hourly['Year'] > 1990]

    # standardise the data in interval [-1, 1], because tanh is definited in [-1,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))

    seq_size = 2
    train = train_data['Q'].to_numpy()
    test = test_data['Q'].to_numpy()
    train = scaler.fit_transform(train.reshape(-1, 1))
    test = scaler.fit_transform(test.reshape(-1, 1))
    Xtrain, Ytrain = to_sequences(train, seq_size)
    Xtest, Ytest = to_sequences(test, seq_size)

    # Reshape train and test data, because the algorithm needs data in form
    # [batch_size, timestep, sequence size]
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], 1, Xtrain.shape[1]))
    Xtest = np.reshape(Xtest, (Xtest.shape[0], 1, Xtest.shape[1]))


    # train different models
    # train_SimpleRNN(Xtrain, Ytrain, Xtest, Ytest, scaler, seq_size)
    # train_SimpleRNN_Dense(Xtrain, Ytrain, Xtest, Ytest, scaler, seq_size)
    # train_LSTM(Xtrain, Ytrain, Xtest, Ytest, scaler, seq_size)
    train_GRU(Xtrain, Ytrain, Xtest, Ytest, scaler, seq_size)


if __name__ == '__main__':
    main()
# Q and PR in Relation (when PR is bigger, than Q gets higher in graph)
# implement a decision tree to better visualize the data
