import pandas as pd
import numpy as np
import statsmodels.api as sm
import math

def readdata(filename = "assign1_data.txt"):
	dataset = pd.read_csv(filename, delimiter = r'\s+')
	return dataset

def costFunction(X, Y, Theta):
	m = len(Y)
	diff = X * Theta - Y
	J = diff.T * diff /(2*m) 
	return J 

def gradientDescent(X, Y, Theta, epsilon = 1, num_iters = 1000, maxdiff = 10e-30):
	m = len(Y)
	x = np.matrix(X)
	y = np.matrix(Y).T
	theta = np.matrix(Theta)
	costFunction(x, y, theta)	
	oldJ = costFunction(x, y, theta)
	for i in xrange(num_iters):
		theta = theta - epsilon * (x.T) * (x*theta - y) / (2*m)
		newJ = costFunction(x, y, theta)
		if abs(newJ - oldJ) < maxdiff:
			break	
		oldJ = newJ
	return theta


def main():
	dataset = readdata()
	X = dataset.iloc[:, 0:2]
	Y = dataset.iloc[:, 2]
	Z = dataset.iloc[:, 3]
	X = sm.add_constant(X)
	theta = np.ones((X.shape[1], 1))
	final_theta = gradientDescent(X, Y,theta)
	print final_theta


if __name__ == "__main__":
	main()
