import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')
from gradientDescent import load_house_data, run_gd


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


x_train, y_train = load_house_data()
X_features = ['size(sqfts)', 'bedrooms', 'floors', 'age']
fig, ax = plt.subplots(1, 4, sharey=True, figsize=(12, 3))
for i in range(len(ax)):
    ax[i].scatter(x_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price 1000's")
plt.show()
w, b, hist = run_gd(x_train, y_train, alpha=1e-7, iters=10)


x_norm, mu, sigma = zscore_normalize_features(x_train)
print(f"Mean = {mu}, sigma = {sigma}")
w_norm, b_norm, hist_norm = run_gd(x_norm,y_train,alpha=1.0e-1, iters=1000)
m = x_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(x_norm[i],w_norm) + b_norm
fig, ax = plt.subplots(1,4,sharey=True, figsize=(12, 3))
for i in range(len(ax)):
    ax[i].scatter(x_train[:, i], y_train, label = "target")
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(x_train[:,i], yp, color="orange", label="predict")
ax[0].set_ylabel("Price 1000s")
plt.show()