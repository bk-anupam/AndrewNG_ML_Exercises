import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    input_df = pd.read_csv(file_path, sep=',', header=None, names=['Population', 'Profit'])
    return input_df


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
