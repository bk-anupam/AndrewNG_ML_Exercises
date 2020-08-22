import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


def plot_data1(X):
    fig1, ax1 = plt.subplots()
    ax1.scatter(X[:, 0], X[:, 1], marker='o')
    ax1.set_title('data1 scatter plot')
    plt.show()
    return fig1, ax1


def pca(X):
    m = len(X)
    cov_matrix = (1/m) * (X.T.dot(X))
    U, S, V = np.linalg.svd(cov_matrix)
    return U, S, V


def draw_eigen_vector(mu, S, U, ax):
    for i in range(2):
        ax.arrow(mu[0], mu[1], 1.5 * S[i] * U[0, i], 1.5 * S[i] * U[1, i],
                 head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

    ax.axis([0.5, 6.5, 2, 8])
    ax.set_aspect('equal')
    ax.grid(False)
    print('Top eigenvector: U[:, 0] = [{:.6f} {:.6f}]'.format(U[0, 0], U[1, 0]))
    print(' (you should expect to see [-0.707107 -0.707107])')


########## Problem Set 1 #############
data1 = sio.loadmat('./data/ex7data1.mat')
X = data1['X']
fig, ax = plot_data1(X)
# perform feature scaling and mean normalization before running pca on input
X_scaled = StandardScaler().fit_transform(X)
mu = np.mean(X, axis=0)
U, S, V = pca(X_scaled)
draw_eigen_vector(mu, S, U, ax)


########## Problem Set 2 #############
def project_data(U, x_i, k):
    # U_reduced has dimension n by k (we choose the first k columns and each column vector has n rows )
    U_reduced = U[:, 0:k]
    # x_i (the ith training example) in vector form has dimensions n by 1
    x_i_projected = U_reduced.T.dot(x_i)
    # x_i_projected has dimensions k by 1 and is the projection of vector x_i onto the principal components of
    # input data
    return x_i_projected


first_projected = project_data(U, X_scaled[0, :], 1)
# X (m by n), U_reduced (n by k)
U_reduced = U[:, 0:1]
X_projected = X_scaled.dot(U_reduced) # X_projected = m by k
X_reconstructed = X_projected.dot(U_reduced.T)
print('Original first example: {}'.format(X_scaled[0, :]))
print('Projected first example: {}'.format(first_projected))
print('Reconstructed first example: {}'.format(X_reconstructed[0, :]))

fig2, ax2 = plt.subplots(figsize=(8,8))
# plot the original data points in blue
ax2.plot(X_scaled[:,0], X_scaled[:, 1], 'bo', ms=8, mec='b', mew=0.5)
ax2.set_aspect('equal')
ax2.grid(False)
ax2.axis([-4,3,-4,3])
# plot the reconstructed data points in red
ax2.plot(X_reconstructed[:, 0], X_reconstructed[:, 1], 'ro', mec='r', mew=2, mfc='none')
# draw dotted lines from each reconstructed data point to corresponding original data point
for i in range(len(X_reconstructed)):
    ax2.plot([X_scaled[i, 0], X_reconstructed[i, 0]], [X_scaled[i, 1], X_reconstructed[i, 1]], '--k')
print('test')