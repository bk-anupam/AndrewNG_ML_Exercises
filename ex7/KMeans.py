import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from matplotlib.animation import FuncAnimation


def find_closest_centroids(X_df, initial_centroids):
    num_centroids = len(initial_centroids)
    # compute the euclidian distance of each data point from each of the centroids and store it in a column of dataframe
    for i in range(num_centroids):
        col_name = 'K_' + str(i+1)
        X_df[col_name] = ((X_df['x1'] - initial_centroids[i, 0]) ** 2) + ((X_df['x2'] - initial_centroids[i, 1]) ** 2)
    # for each data point ( row in dataframe ) compare the distance from each centroid and assign the data point
    # to centroid with minimum distance
    if 'cluster' in X_df.columns:
        X_df['cluster'] = X_df.iloc[:, 2:-1].idxmin(axis=1)
    else:
        X_df['cluster'] = X_df.iloc[:, 2:].idxmin(axis=1)


def compute_centroids(X_df, cluster_names_colors):
    new_centroids = {}
    for cluster_name_color in cluster_names_colors:
        cluster_df = X_df[X_df['cluster'] == cluster_name_color[0]]
        new_centroids[cluster_name_color[0]] = [cluster_df['x1'].mean(), cluster_df['x2'].mean()]
    return new_centroids


def plot_data(X_df, cluster_names_colors, centroids, iter_count):
    fig, ax = plt.subplots()
    for cluster_name_color in cluster_names_colors:
        cluster_df = X_df[X_df['cluster'] == cluster_name_color[0]]
        ax.scatter(cluster_df['x1'], cluster_df['x2'], marker='o', color=cluster_name_color[1])
        ax.plot(centroids[cluster_name_color[0]][0], centroids[cluster_name_color[0]][1], '-Xk', ms=10, lw=2, mew=2)
    ax.set_title('Data points clustering at iteration: {}'.format(iter_count))
    plt.show()


def run_kmeans(X_df, initial_centroids, num_iterations):
    centroids = initial_centroids
    for iter_count in range(num_iterations):
        find_closest_centroids(X_df, centroids)
        new_centroids = compute_centroids(X_df, cluster_names_colors)
        plot_data(X_df, cluster_names_colors, new_centroids, iter_count)
        centroids = np.array(list(new_centroids.values()))

    # fig = plt.figure()
    # anim = FuncAnimation(fig, plot_data,
    #                      frames=10,
    #                      interval=500,
    #                      repeat_delay=2,
    #                      fargs=(X_df, cluster_name_colors))
    # return anim


os.getcwd()
data = sio.loadmat('./data/ex7data2.mat')
X = data['X']
X_df = pd.DataFrame(X, columns=['x1','x2'])
K = 3
colors = ['red', 'green', 'blue']
cluster_names_colors = [('K_' + str(i+1), colors[i]) for i in range(K)]
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
run_kmeans(X_df, initial_centroids, 10)
#find_closest_centroids(X_df, initial_centroids)
#print(X_df['cluster'][0:3])
#new_centroids = compute_centroids(X_df, cluster_names_colors)
#plot_data(X_df, cluster_names_colors)
print('test')

