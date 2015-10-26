'''Import the required modules'''
import sys

from numpy import *
import scipy.io as sio

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

'''h(theta) = theta0 + theta1.X1 + theta2.X2......thetaN.XN which really is the
dot product of the parameters and the features n in the input matrix X'''
def hypothesis(X, theta):
    return X.dot(theta)

def computeCostLoop(X, y, theta):
    m = len(y)    #since the output will always be a 1D array
    cumulative_sum = 0
    for i in range(0, m):
        cumulative_sum += (hypothesis(X[i], theta) - y[i]) ** 2
    cumulative_sum = (1.0 / (2 * m) * cumulative_sum
    return cumulative_sum

def computeCost(X, y, theta):
    """vectorized version of computing cost"""
    m = len(y)
    term = hypothesis(X, theta) -y
    #sum(term**2) ~= term.T.dot(term)
    return (term.T.dot(term) / (2 * m))[0,0]


def gradientDescentLoop(X, y, theta, alpha, iteration):
    """loop version"""
    grad = copy(theta)
    m = len(y)
    n = shape(X)[1]   #figure out how to get number of columns from matrix

    for counter in range(0, iterations):
    '''create n which is inner sums'''
    cum_sum = [0 for x in range(0, n)]

        for j in range(0, n):
            for i in range(0, m):
                term = (hypothesis(X[i], grad) - y[i])
                cum_sum[j] +=X[i, j] * (term)

'''Assign new values for each gradient, to get simultaneous update'''
        for j in range(0, n):
            grad[j] = grad[j] - cum_sum[j] * (alpha/m)

    return grad


def gradientDescent(X, y, theta, alpha, iterations):
    '''Vectorized gradient descent'''
    grad = copy(theta)
    m = len(y)

    for counter in range(0, iterations):
        inner_sum = X.T.dot(hypothesis(X, grad) -y)
        grad -= alpha / m * inner_sum

    return grad


def plot(X, y):
    '''Create a plot of X and y data'''
    pyplot.plot(X, y, 'rx', markersize=5 )
    pyplot.ylabel('Profit')
    pyplot.xlabel('Population')

def part2_1():
    data = sio.loadmat('/home/bernard/Desktop/ml_ng/ex1Data.mat')
    X = data['x']
    y = data['y']
    m = len(y)
    y = y.reshape 







































        
































    










