import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from sklearn import linear_model
import matplotlib.pyplot as plt

def readdata(filename = "assign1_data.txt"):
	dataset = pd.read_csv(filename, delimiter = r'\s+')
	return dataset

def costFunction(X, Y, Theta):
	m = len(Y)
	h = X*Theta > 0
	J = np.linalg.norm(h - Y, ord = 1) / float(m)
	return J 


def gradientDescent(X, Y, Theta, epsilon = 1, num_iters = 300000, maxdiff = 10e-8,\
 regularization = 10e-7):
	m = len(Y)
	x = np.matrix(X)
	y = np.matrix(Y).T
	theta = np.matrix(Theta)
	oldJ = costFunction(x, y, theta)
	for i in xrange(num_iters):
		theta = theta - epsilon *(  (x.T)*((x*theta > 0) - y) / m )
		newJ = costFunction(x, y, theta)
		if newJ < maxdiff:
			break
		oldJ = newJ
	return theta

def stochasticGradientDescent(X, Y, Theta, epsilon = 1, num_iters = 1000, batchsize = 1, shuffle = True, maxdiff = 10e-5):
	# this is the so-called stochastic gradient descent
	m = len(Y)
	x = np.matrix(X)
	y = np.matrix(Y).T
	theta = np.matrix(Theta)
	if shuffle:
		xy = np.hstack((x, y))
		np.random.shuffle(xy)
		x = xy[:, 0:-1]
		y = xy[:, -1]	
	for i in xrange(num_iters):
		for j in xrange(len(y) / batchsize):
			start = j * batchsize
			end = (j + 1) * batchsize
			theta = theta - epsilon *(  (x[start:end].T)*((x[start:end]*theta > 0) 
				- y[start:end]) / float( len(y[start:end]) ) )
		if end != len(y) - 1:
			theta = theta - epsilon *(  (x[end:].T)*((x[end:]*theta > 0)
				- y[end:]) / float( len(y[end:]) ) )
		newJ = costFunction(x, y, theta)
		if newJ < maxdiff:
			break
		
	return theta


def main(argv):
	dataset = readdata()
	X = dataset.iloc[:, 0:2]
	Z = dataset.iloc[:, 3]
	X = sm.add_constant(X)
	theta = np.ones((X.shape[1], 1))
	N = int(argv)
	Xtrain = X[0:N]
	Ztrain = Z[0:N]	
	final_theta = gradientDescent(Xtrain, Ztrain, theta, num_iters = 50000)
	Xtest = np.matrix(X[75:])
	Ztest = np.matrix(Z[75:])
	Z_predict = np.matrix([1 if t > 0 else 0 for t in Xtest * final_theta])
	print "the accuracy with test last 25 data with the traininng on first ", N, " data:"
	print np.sum(abs(Ztest - Z_predict) ) / float(Ztest.shape[1])

if __name__ == "__main__":
	main(sys.argv[1])
