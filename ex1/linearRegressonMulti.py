import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regressionBase as base
import sys


def extract_features(df):
    X = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)
    m = len(X)
    return X, y, m


def normalize_features(X):
    for col_num in range(X.shape[1]):
        col_mean = np.mean(X[:, col_num])
        col_std = np.std(X[:, col_num])
        X[:, col_num] = (X[:, col_num] - col_mean) / col_std
    return X


def run_regression_multi(filename):
    file_path = "./data/" + filename
    # Load the input data as dataframe and change the type of all columns to float from int
    input_df = base.load_data_multi(file_path).astype('float64')
    input_features, y, m = extract_features(input_df)
    X = normalize_features(input_features.copy())
    # add vector of all 1's as the first feature vector
    X = base.add_one_vector(X)
    # initialize theta, learning rate alpha and no of iterations
    theta = np.zeros((X.shape[1], 1))
    alpha = 0.01
    num_iters = 400
    theta, J_history = base.gradient_descent(X, y, theta, alpha, num_iters, m)
    base.plot_cost_vs_iteration(J_history, num_iters)
    print("Theta computed from gradient descent: {}".format(theta))
    house = np.array([[1, (1650 - np.mean(input_features[:, 0]))/np.std(input_features[:, 0]),
                       (3 - np.mean(input_features[:, 1]))/np.std(input_features[:, 1])]])
    predicted_price = np.dot(house, theta)
    print("Predicted price of house of size 1650 sqft with 3 bedrooms is {}$".format(predicted_price[0][0]))
    print('Finished')


if __name__ == "__main__":
    input_filename = sys.argv[1]
    run_regression_multi(input_filename)