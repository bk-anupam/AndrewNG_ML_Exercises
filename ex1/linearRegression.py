import sys
import matplotlib.pyplot as plt
import numpy as np
import regressionBase as base
from mpl_toolkits import mplot3d

def gradient_descent(X, y, theta, alpha, no_iters, m):
    J_history = np.zeros((no_iters, 1))
    for iteration in range(no_iters):
        J_history[iteration] = base.compute_cost(X, theta, y, m)
        h = np.dot(X, theta)
        err = h - y
        theta_err = alpha * (np.dot(np.transpose(X), err) / m)
        theta = theta - theta_err
    plot_cost_vs_iteration(J_history, no_iters)
    return theta


def plot_cost_vs_iteration(J_history, no_iters):
    iter_arr = np.arange(1, 1501).reshape(1500,1)
    fig, ax_costiter = plt.subplots()
    ax_costiter.plot(iter_arr, J_history, color='green')
    plt.xlabel('No of iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs No of iterations')
    plt.show()


def line_fit_plot(init_axes, X, result_theta):
    predictedY = np.dot(X, result_theta)
    population = X[:, 1]
    init_axes.plot(population, predictedY, color='red')
    plt.show()


def costfunction_3d_plot(X, y, m):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    t0, t1 = np.meshgrid(theta0_vals, theta1_vals)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            temp_theta = np.array([theta0_vals[i], theta1_vals[j]])
            #temp_theta = np.array([t0[i][j], t1[i][j]])
            J_vals[i, j] = base.compute_cost(X, temp_theta, y, m)
    J_vals = J_vals.T

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis')
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Surface')
    ax_contour = fig.add_subplot(122)
    plt.contour(theta0_vals, theta1_vals, J_vals, cmap='viridis', linewidths=2, levels=np.logspace(-2,3,20))
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.title('Contour')


def test_3dplot():
    def f(x, y):
        return np.sin(np.sqrt(x ** 2 + y ** 2))
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def run_linear_regression(input_filename):
    file_path = "./data/" + input_filename
    inputDF = base.load_data(file_path)
    base.scatter_plot(inputDF)
    # the feature matrix X
    X = inputDF.iloc[:, 0:1].values
    X = base.add_one_vector(X)
    # the vector y
    y = inputDF.iloc[:, 1:2].values
    m = len(X)
    # now initialize theta with all zeros
    theta = np.zeros(2).reshape(2, 1)
    initial_cost = base.compute_cost(X, theta, y, m)
    print("cost with theta values set to {} : {}".format(theta, initial_cost))
    theta2 = np.array([[-1], [2]])
    initial_cost2 = base.compute_cost(X, theta2, y, m)
    print("cost with theta values set to {} : {}".format(theta2, initial_cost2))
    # initialize the learning rate and no of iterations for gradient descent
    alpha = 0.01
    no_iters = 1500
    print('Running gradient descent on training dataset')
    result_theta = gradient_descent(X, y, theta, alpha, no_iters, m)
    print(result_theta)
    fig_fit, ax_fit = base.scatter_plot(inputDF)
    line_fit_plot(ax_fit, X, result_theta)
    population1 = np.array([[1, 3.5]])
    prediction1 = np.dot(population1, result_theta)
    print("For population of 35000 the predicted profit is: {}".format(prediction1))
    costfunction_3d_plot(X, y, m)


if __name__ == "__main__":
    file_name = sys.argv[1]
    run_linear_regression(file_name)