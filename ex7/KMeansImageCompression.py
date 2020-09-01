import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import ex7.KMeansBase as base


def display_image(image_data):
    fig, ax = plt.subplots()
    ax.imshow(image_data)


def get_random_initial_centroids(X, K):
    random_indices = np.random.choice(X.shape[0], size=K)
    random_initial_centroids = X[random_indices, :]
    return random_initial_centroids


def run_kmeans(X_df, data_cols, initial_centroids, clusters, num_iterations):
    centroids = initial_centroids
    for iter_count in range(num_iterations):
        X_df = base.find_closest_centroids(X_df, centroids, len(data_cols))
        centroids = base.compute_centroids(X_df, data_cols, clusters)
    # the cluster column contains one of the K cluster values that each data point belongs to. In our
    # image compression example, this means that each pixel can be represent using a 16 bit short int
    # instead of the [R G B] array wherein each element represents a color intensity in 8 bits (0-255)
    # It is this compressed image representation ( a 128 by 128 array) that we return after running the kmeans
    return X_df['cluster'].values.reshape(128, 128), centroids


def uncompress_image_data(X_image_compressed, final_centroids):
    recovered_image = []
    for row in range(X_image_compressed.shape[0]):
        recovered_image_row = []
        for col in range(X_image_compressed.shape[1]):
            index = X_image_compressed[row][col]
            recovered_image_row.append(list(final_centroids[index]))
        recovered_image.append(recovered_image_row)
    return np.array(recovered_image)


image_data = sio.loadmat('./data/bird_small.mat')
X_image = image_data['A']
# The image data is a three-dimensional matrix A whose first two indices identify
# a pixel position and whose last index represents red, green, or blue.
dim = X_image.shape
# Reshape the image data to create a mx3 matrix of pixel colors where m is the number of pixels ( 128 * 128 )
X_image_reshaped = X_image.reshape(-1, 3)
# The mx3 numpy matrix is loaded as a dataframe
X_image_df = pd.DataFrame(X_image_reshaped, columns=['R','G','B'], dtype=np.int32)
data_columns = X_image_df.columns.values
# number of clusters ( here number of colors to paint an image )
K = 16
max_iters = 10
num_data_cols = 3
# select K centroids ( pixels ) randomly as initial centroids
display_image(X_image)
initial_centroids = get_random_initial_centroids(X_image_reshaped, K)
X_image_reshaped = base.find_closest_centroids(X_image_df, initial_centroids, num_data_cols)
X_image_compressed, final_centroids = run_kmeans(X_image_reshaped, data_columns, initial_centroids, np.arange(K), max_iters)
X_image_recovered = uncompress_image_data(X_image_compressed, final_centroids)
display_image(X_image_recovered)
print('test')