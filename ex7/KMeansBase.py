import numpy as np


def find_closest_centroids(X_df, initial_centroids, num_data_cols):
    num_centroids = len(initial_centroids)
    # compute the euclidian distance of each data point from each of the centroids and store it in a column of dataframe
    for i in range(num_centroids):
        col_name = i
        initial_centroid_matrix = np.tile(initial_centroids[i], (len(X_df), 1))
        X_df[col_name] = np.sqrt(np.sum(np.square(X_df.iloc[:, 0:num_data_cols].values - initial_centroid_matrix), axis=1))
    # for each data point ( row in dataframe ) compare the distance from each centroid and assign the data point
    # to centroid with minimum distance
    if 'cluster' in X_df.columns:
        X_df['cluster'] = X_df.iloc[:, num_data_cols:-1].idxmin(axis=1)
    else:
        X_df['cluster'] = X_df.iloc[:, num_data_cols:].idxmin(axis=1)
    return X_df


def compute_centroids(X_df, data_col_names, clusters):
    new_centroids = []
    for cluster in clusters:
        cluster_df = X_df[X_df['cluster'] == cluster]
        new_centroids.append([cluster_df[data_col_name].mean() for data_col_name in data_col_names])
    return np.array(new_centroids).astype('int32')
