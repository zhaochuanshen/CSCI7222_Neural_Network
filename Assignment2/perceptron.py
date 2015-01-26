import numpy as np
import random

def costFunction(X, Y, Theta):
	m = len(Y)
	h = X*Theta > 0 
	J = np.linalg.norm(h - Y, ord = 1) / float(m)
	return J 


def stochasticGradientDescent(X, Y, Theta = None, epsilon = 0.1, num_iters = 1000, batchsize = 1, shuffle = True, maxdiff = 10e-5):
	# this is the so-called stochastic gradient descent
	m = len(Y)
	x = np.matrix(X)
	y = np.matrix(Y).T
	if not Theta:
		Theta =np.zeros((X.shape[1], 1))
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
		if end < len(y) :
			print j,  end	
			theta = theta - epsilon *(  (x[end:].T)*((x[end:]*theta > 0)
					- y[end:]) / float(len( y[end:] )) )
		newJ = costFunction(x, y, theta)
		if newJ < maxdiff:
			break
		
	return theta
