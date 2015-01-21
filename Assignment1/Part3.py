import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math
from sklearn import linear_model
import matplotlib.pyplot as plt
import random

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
				- y[start:end]) / m )
			newJ = costFunction(x, y, theta)
			if newJ < maxdiff:
				break
	return theta

	
def main():
	dataset = readdata()
	X = dataset.iloc[:, 0:2]
	Y = dataset.iloc[:, 2]
	Z = dataset.iloc[:, 3]
	X = sm.add_constant(X)
	theta = np.ones((X.shape[1], 1))
	final_theta = gradientDescent(X, Z, theta, num_iters = 50000)
	# call gradient descent
	print final_theta
		
	final_theta_sto = stochasticGradientDescent(X, Z, theta, num_iters = 500, batchsize = 10)
	#call stochastic gradient descent
	print final_theta_sto

	#plot
	x = dataset.iloc[:,0]
	y = dataset.iloc[:,1]
	co = dataset.iloc[:, -1]
	colors = ['r' if t == 0 else 'b' for t in co]
	plt.figure(1)
	plt.scatter(x, y, marker = 'o', c = colors)
	t = np.arange(min(x), max(x), (max(x) - min(x) ) / 100.)
	
	plt.plot(
	t, -final_theta[0,0] / final_theta[2, 0]  - final_theta[1, 0] * t / final_theta[2, 0], \
	'r', label = 'gradient descent')
	plt.plot(t, 
		-final_theta_sto[0,0] / final_theta_sto[2, 0] - final_theta_sto[1, 0] * t / final_theta_sto[2, 0], 
	color = 'b', label = 'stochastic gradient descent')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
	           ncol=2, mode="expand", borderaxespad=0.)
	plt.savefig("logistic_regression.pdf")

if __name__ == "__main__":
	main()
