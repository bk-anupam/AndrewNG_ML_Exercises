import numpy as np
import pandas as pd
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC


def plot_data(X, y):
    X_df = pd.DataFrame(data=X)
    # Indices of training examples where output = +1
    pos = y==1
    X_pos_df = X_df[pos]
    # Indices of training examples where output = -1
    neg = y==0
    X_neg_df = X_df[neg]
    plt.plot(X_pos_df.iloc[:, 0], X_pos_df.iloc[:, 1], 'k+', ms=8, color='black')
    plt.plot(X_neg_df.iloc[:, 0], X_neg_df.iloc[:, 1], 'ko', ms=8, mew=1, mfc='cyan', mec='k')
    plt.title('data1 plot')
    plt.show()


def make_meshgrid(x, y):
    h = .02  # step size in the mesh
    # create a mesh to plot in
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_datapoints(X, y, ax):
    X_df = pd.DataFrame(data=X)
    # Indices of training examples where output = +1
    pos = y == 1
    X_pos_df = X_df[pos]
    # Indices of training examples where output = -1
    neg = y == 0
    X_neg_df = X_df[neg]
    ax.scatter(X_pos_df.iloc[:, 0], X_pos_df.iloc[:, 1], c='black', marker='+', cmap=plt.cm.coolwarm, s=30,
               edgecolors='k')
    ax.scatter(X_neg_df.iloc[:, 0], X_neg_df.iloc[:, 1], marker='o', c='cyan', cmap=plt.cm.coolwarm, s=25, edgecolors='k')
    ax.set_ylabel('y label here')
    ax.set_xlabel('x label here')
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


def fit_model_data1(X, y):
    clf = LinearSVC(C=100)
    clf.fit(X, y)
    fig, ax = plt.subplots()
    # title for the plots
    title = ('Decision surface of linear SVC ')
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax = plot_datapoints(X, y, ax)
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_data2(X, y):
    fig, ax = plt.subplots(figsize=(10,6))
    title = ('Decision surface of non-linear SVC ')
    ax = plot_datapoints(X, y, ax)
    ax.set_title(title)
    ax.legend()
    plt.show()


data1 = sio.loadmat('./data/ex6data1.mat')
fit_model_data1(data1['X'], data1['y'])
#plot_data(data1['X'], data1['y'])
print('data1 plotted')
data2 = sio.loadmat('./data/ex6data2.mat')
X_2 = data2['X']
y_2 = data2['y']
plot_data2(X_2, y_2)
print('data2 plotted')