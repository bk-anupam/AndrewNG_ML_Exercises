import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import regressionBase as base
from sklearn.linear_model import LinearRegression


def run_regression(input_filename):
    file_path = "./data/" + input_filename
    input_df = base.load_data(file_path)
    fig, ax = base.scatter_plot(input_df)
    population = input_df.iloc[:, 0].values.reshape(-1, 1)
    profit = input_df.iloc[:, 1].values.reshape(-1, 1)
    reg = LinearRegression()
    reg.fit(population, profit)
    prediction_space = np.linspace(min(population), max(population)).reshape(-1, 1)
    predicted_profit = reg.predict(prediction_space)
    ax.plot(prediction_space, predicted_profit, color='red')
    plt.show()
    print("The R2 score of the model is: {}".format(reg.score(population, profit)))


if __name__ == "__main__":
    file_name = sys.argv[1]
    run_regression(file_name)