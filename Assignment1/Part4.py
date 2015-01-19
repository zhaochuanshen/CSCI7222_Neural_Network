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

def logistic(x):
	return 1 / (1 + np.exp(-x))

def costFunction(X, Y, Theta, regulization):
	m = len(Y)
	h = logistic(X*Theta)
	J = -( Y.T * np.log(h)  +  (1-Y).T* np.log(1- h) ) / (m) + regulization * Theta[1:].T * Theta[1:]
	return J 


def gradientDescent(X, Y, Theta, epsilon = 1, num_iters = 300000, maxdiff = 10e-5,\
 regulization = 10e-7):
	m = len(Y)
	x = np.matrix(X)
	y = np.matrix(Y).T
	theta = np.matrix(Theta)
	oldJ = costFunction(x, y, theta, regulization)
	for i in xrange(num_iters):
		#theta = theta - epsilon * (x.T) * (x*theta - y) / m
		theta = theta - epsilon *(  (x.T)*(logistic(x*theta) - y) / m + regulization * theta /m)
		theta[0] = theta[0] - epsilon* regulization * theta[0] /m
		newJ = costFunction(x, y, theta, regulization)
		if abs(newJ - oldJ) / newJ < maxdiff:
			break
		oldJ = newJ
		if np.isnan(oldJ[0,0]):
			raise NameError('NaN Error') 
	print oldJ, newJ
	print i
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
	final_theta = gradientDescent(Xtrain, Ztrain, theta, num_iters = 30000)
	Xtest = np.matrix(X[75:])
	Ztest = np.matrix(Z[75:])
	Z_predict = np.matrix([1 if t > 0 else 0 for t in Xtest * final_theta])
	print Ztest.shape
	print Z_predict.shape
	print np.sum(abs(Ztest - Z_predict) ) / float(Ztest.shape[1])

if __name__ == "__main__":
	main(sys.argv[1])
