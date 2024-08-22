import copy
import numpy as np
np.set_printoptions(precision=2)
import matplotlib.pyplot as plt


def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    x = data[:,:4]
    y = data[:, 4]
    return x, y


def gradient_function(x, y, w, b):
    # to calculate dj_db and dj_dw
    m, n = x.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        error = (np.dot(x[i], w) + b - y[i])
        for j in range(n):
            dj_dw[j] += error*x[i][j]
        dj_db += error
    dj_db /= m
    dj_dw /= m
    return dj_db, dj_dw


def cost_function(x,y,w,b):
    m = x.shape[0]
    j = 0
    for i in range(m):
        j += (np.dot(x[i], w) + b - y[i])**2
    j /= 2*m
    return j


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iter):
    m = x.shape[0]
    hist = {}
    hist["cost"] = []; hist["params"] = []; hist["grads"] = []; hist["iter"] = [];
    w = copy.deepcopy(w_in)
    b = b_in
    save_interval = np.ceil(num_iter/10000)
    # print(
    #     f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  ")
    # print(
    #     f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|")

    for i in range(num_iter):
        dj_db, dj_dw = gradient_function(x, y, w, b)
        w = w - alpha*dj_dw
        b = b - alpha*dj_db
        cst = cost_function(x, y, w, b)
        if i == 0 or (i % save_interval ) == 0:
            hist["cost"].append(cst)
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)
        if i%1000 == 0:
            # print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")
            print(f"Iter - {i}, cost = {cst}")
    return w, b, hist


def run_gd(x, y, alpha=1e-6, iters=1000):
    m, n = x.shape
    initial_w = np.zeros(n)
    initial_b = 0
    w, b, hist = gradient_descent(x, y, initial_w, initial_b, cost_function, gradient_function, alpha, iters)
    print(f"w, b found by gradient descent are {w}, {b}")
    return (w, b, hist)