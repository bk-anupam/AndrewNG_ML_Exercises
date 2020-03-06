import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    input_df = pd.read_csv(file_path, sep=',', header=None, names=['Population', 'Profit'])
    return input_df


def load_data_multi(file_path):
    return pd.read_csv(file_path, sep=',', header=None, names=['Size', 'NoOfBedrooms', 'Price'])


# the vectorized cost function
def compute_cost(X, theta, y, m):
    J = 0
    J = np.dot(np.transpose((np.dot(X, theta) - y)), (np.dot(X, theta) - y))
    return J[0][0] / (2 * m)


def add_one_vector(X):
    one_vector = np.ones(len(X)).reshape(len(X), 1)
    return np.concatenate((one_vector, X), axis=1)


def scatter_plot(inputDF):
    # do an EDA by plotting the data
    fig, ax_scatter = plt.subplots()
    population = inputDF.iloc[:, 0]
    profit = inputDF.iloc[:, 1]
    ax_scatter.scatter(population, profit)
    plt.xlabel('Population of city in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title("Profit vs Population")
    plt.show()
    return fig, ax_scatter
