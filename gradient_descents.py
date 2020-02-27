#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy.linalg import eigvalsh
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def getFunc(Aq, bq):
    def func(x):
        return 0.5 * Aq @ x @ x + bq @ x
    return func

def gradient(f, x, alpha=1e-8):
    x = np.array(x, dtype = np.float64) 
    deriv = []
    for i in range(len(x)):
        x_plus_a, x_minus_a = np.copy(x), np.copy(x)
        x_plus_a[i] = x_plus_a[i] + alpha
        x_minus_a[i] = x_minus_a[i] - alpha
        deriv.append( (f(x_plus_a) - f(x_minus_a))/(2*alpha) ) 
    return np.array(deriv)

def gradient_descent(f, x, alpha=1e-1, eps=1e-8, max_iter=50, mod = "simple", beta=0.5, speaks=True):
    if mod == "2nd":
        alpha = 0
    def grad(point):
        return gradient(f, point)
    x = np.array(x, dtype=np.float64)
    prev_value = f(x)*30 - 100
    counter = 0
    while( counter < max_iter and abs(f(x) - prev_value) > eps):
        if mod == "2nd":
            alpha = 1
            while f(x) < f(x - alpha * grad(x)):
                alpha = alpha * beta
        prev_value = f(x)
        if speaks:
            print(counter, ". x:", x, " f(x):", f(x), " alpha: ", alpha)
        x = x - alpha * grad(x)
        counter += 1
    if speaks:
        print(f"minimum {counter}. x:", x, " f(x):", f(x), " alpha: ", alpha)
        print(f"difference = {abs(f(x) - prev_value)}")
    return x

def gradient_descent_min_alpha(f, x, eps=1e-8, max_iter=50, speaks=True):
    
    def grad(point):
        return gradient(f, point)
    x = np.array(x, dtype=np.float64)
    prev_value = f(x)*30 - 100
    counter = 0
    alpha = 0
    while( counter < max_iter and abs(f(x) - prev_value) > eps):
        prev_value = f(x)
        if speaks:
            print(counter, ". x:", x, " f(x):", f(x), " alpha: ", alpha)
        h = grad(x)
        def func_to_optim(alpha):
            return f(x-alpha*h)
        alpha = gradient_descent(func_to_optim, [1,], mod="2nd", speaks = False, eps = 1e-3)
        #alpha = ((A@x+b)@h)/((A@h)@h)
        x = x - alpha * grad(x)
        counter += 1
    if speaks:
        print(f"minimum {counter}. x:", x, " f(x):", f(x), " alpha: ", alpha)
    return x

def gradient_descent_yar(f, v1, v2, eps=1e-11, max_iter=50, t=0.1, speaks=True):
    def grad(point):
        return gradient(f, point)
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    x2 = gradient_descent(f, v1, eps=0.1, max_iter=7, speaks=False, alpha = 0.1)
    prev_value = f(x)*30 - 100
    counter = 0
    while( counter < max_iter and abs(f(x2) - prev_value) > eps):
        prev_value = f(x2)
        x1 = x2
        x2 = gradient_descent(f, v2, eps=0.01, max_iter=7, speaks=False, alpha = 0.1)
        if speaks:
            print(counter, ". x1:", x1, " f(x1):", f(x1), ". x2:", x2, " f(x2):", f(x2))
        v2 = x2 - (x2 - x1)/norm(x2 - x1) * (int)(f(x2) < f(x1))
        counter += 1
    if speaks:
        print(f"minimum {counter}. x:", x2, " f(x):", f(x2))
    return x2


def plot3D(f, rangeX=np.arange(-5, 5, 0.25), rangeY = np.arange(-5, 5, 0.25), max_z=10, min_z=0):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X, Y = np.meshgrid(rangeX, rangeY)
    Z = f([X, Y])
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(min_z, max_z)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


# In[ ]:




