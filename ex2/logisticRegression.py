import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.special import expit
from scipy.optimize import minimize


def add_one_vector(X):
    one_vector = np.ones(len(X)).reshape(len(X), 1)
    return np.concatenate((one_vector, X), axis=1)


def plot_raw_data(X, y):
    admitted = y == 1
    X_admitted = X[admitted]
    X_notadmitted = X[~admitted]
    plt.plot(X_admitted[:, 0], X_admitted[:, 1], 'k+', ms=8, color='black', label='admitted')
    plt.plot(X_notadmitted[:, 0], X_notadmitted[:, 1], 'ko', ms=8, mew=1, mfc='cyan', mec='k',label='not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Scatter plot of training data')
    plt.legend()
    plt.show()


def cost_function(theta, X, y):
    theta = theta.reshape(-1, 1)
    h = expit(np.dot(X, theta))
    J = -(1/m) * (np.dot(np.transpose(y), np.log(h)) + np.dot(np.transpose(1 - y), np.log(1 - h)))
    grad = (1 / m) * np.dot(np.transpose(X), (h - y))
    return J, grad


def gradient(theta, X, y):
    h = expit(np.dot(X, theta))
    grad = (1/m) * np.dot(np.transpose(X), (h - y))
    return grad.flatten()


def predict(theta, X):
    product = np.dot(X, theta)
    classifier = lambda item: 1 if item > 0 else 0
    v_classifier = np.vectorize(classifier)
    return v_classifier(product)


print(os.getcwd())
data = pd.read_csv('./data/ex2data1.txt')
X = data.iloc[:, 0:2].values
y1d = data.iloc[:, 2].values
plot_raw_data(X, y1d)
X = add_one_vector(X)
m, n = X.shape
y = y1d.reshape(-1, 1)

# now initialize theta with all zeros
initial_theta = np.zeros(n)
cost_initial, gradient_initial = cost_function(initial_theta, X, y)
#gradient_initial = gradient(initial_theta, X, y)
print('Cost at initial theta(zeros): {}'.format(cost_initial))
print('Expected cost: 0.693')
print('Gradient at initial theta: {}'.format(gradient_initial))
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

test_theta = np.array([-24, 0.2, 0.2]).reshape(-1, 1)
cost_test, gradient_test = cost_function(test_theta, X, y)
#gradient_test = gradient(test_theta, X, y)
print('Cost at test theta: {}'.format(cost_test))
print('Expected cost: 0.218')
print('Gradient at test theta: {}'.format(gradient_test))
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

options= {'maxiter': 400}
opt = minimize(fun=cost_function, x0=initial_theta, args=(X, y), jac=True, method='TNC', options=options)
optimized_theta = opt.x
cost_optimized = opt.fun
print('Optimized cost: {}'.format(cost_optimized))
print('Expected optimized cost: 0.203')
print('Optimized theta: {}'.format(optimized_theta))

predictions = predict(optimized_theta, X)
is_correct = predictions == y1d
accuracy = len(predictions[is_correct]) / len(is_correct)
print('Accuracy of predictions: {}'.format(round(accuracy, 2)))
print('plotted')