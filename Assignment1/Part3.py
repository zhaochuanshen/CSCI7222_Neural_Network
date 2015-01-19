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


def gradientDescent(X, Y, Theta, epsilon = 1, num_iters = 300000, maxdiff = 10e-8,\
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
	print oldJ, newJ
	print i
	return theta
	
def main():
	dataset = readdata()
	X = dataset.iloc[:, 0:2]
	Y = dataset.iloc[:, 2]
	Z = dataset.iloc[:, 3]
	X = sm.add_constant(X)
	theta = np.ones((X.shape[1], 1))
	final_theta = gradientDescent(X, Z, theta, num_iters = 300000)
	print final_theta
	x = dataset.iloc[:,0]
	y = dataset.iloc[:,1]
	co = dataset.iloc[:, -1]
	colors = ['r' if t == 0 else 'b' for t in co]
	plt.figure(1)
	plt.scatter(x, y, marker = 'o', c = colors)
	t = np.arange(min(x), max(x), (max(x) - min(x) ) / 100.)
	plt.plot(t, -final_theta[0,0] / final_theta[2, 0] - final_theta[1, 0] * t / final_theta[2, 0])
	#plt.show()
	plt.savefig("logistic_regression.pdf")
	
	'''
	[[ -6.76082073]
	 [-30.82192608]
	  [ 46.89694015]]
	X = dataset.iloc[:, 0:2]
	Y = dataset.iloc[:, 3]
	x = 1000000000000000000.0 
	logreg = linear_model.LogisticRegression(C = x)
	logreg.fit(X, Y)
	print logreg.coef_
	print logreg.intercept_
	'''


if __name__ == "__main__":
	main()
