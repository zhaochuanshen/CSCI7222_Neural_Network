import numpy as np
import random

def costFunction(X, Y, Theta):
	m = len(Y)
	h = X*Theta > 0 
	J = np.linalg.norm(h - Y, ord = 1) / float(m)
	return J 


def stochasticGradientDescent(X, Y, Theta = None, epsilon = 0.1, num_iters = 1000, batchsize = 1, shuffle = True, maxdiff = 10e-5):
	# this is the so-called stochastic gradient descent
	# X and Y must be np matrix, with the same number of rows
	m = len(Y)
	x = X
	y = Y
	if not Theta:
		Theta =np.zeros((x.shape[1], y.shape[1]))
	theta = np.matrix(Theta)
	for i in xrange(num_iters):
		if shuffle:
			xcolsize = x.shape[1]
			xy = np.hstack((x, y))
			np.random.shuffle(xy)
			x = xy[:, 0:xcolsize]
			y = xy[:, xcolsize:]
		for j in xrange(len(y) / batchsize):
			start = j * batchsize
			end = (j + 1) * batchsize
			theta = theta - epsilon *(  (x[start:end].T)*((x[start:end]*theta > 0) 
				- y[start:end]) / float( len(y[start:end]) ) )
		if end < len(y) :
			print j,  end	
			theta = theta - epsilon *(  (x[end:].T)*((x[end:]*theta > 0)
					- y[end:]) / float(len( y[end:] )) )
		newJ = costFunction(x, y, theta)
		if newJ < maxdiff:
			break
		
	return theta
