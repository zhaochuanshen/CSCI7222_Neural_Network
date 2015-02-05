import numpy as np
import random


def squreError(t, y):
	# t is the label  and y is the probability
	# in this case, both t and y are 2500 * 10
	# this is a cost function
	delta = y - t
	cost = np.sum(delta * delta)
	return (cost, delta)

def cross_entropy(t, y):
	# this is a cost function
	#t is the label  and y is the probability
	logy = np.log(y)
	cost = -np.sum( t * y)
	gradient = t / y
	return (cost, gradient)

def softmax(x, theta):
	# this is activation function
	temp = np.exp( np.dot(x, theta) )
	row_sum = np.sum(temp, axis = 1)
	g = temp / row_sum[:, None]
	return g

def mytanh(x, theta):
	# this is a activation function
	z = np.dot(x, theta)
	g = 2. / (1 + np.exp(-z)) - 1.
	gradient = (1 + g) * (1 - g) /2. # for this case, still 2500 * 10
	return (g, gradient)

def functionEvaluation(cost_func, activation_func, x, y, theta):
	m = float(x.shape[0])
	if activation_func == softmax and cost_func == cross_entropy:
		# softmax function is different from tanh or 
		g = softmax(x, theta)
		(cost , _) = cost_func(y, g)
		gradtemp = g - y
		grad = np.dot(x.T, gradtemp)
		return (cost/m, grad/m)
	g, acti_grad = activation_func(x, theta)
	cost, cost_grad = cost_func(y, g)
	grad = np.dot(x.T, acti_grad * cost_grad) 
	return (cost / m, grad / m)

def shuffleXYTogether(X, Y):
	xcolsize = X.shape[1]
	xy = np.hstack((X, Y))
	np.random.shuffle(xy)
	x = xy[:, 0:xcolsize]
	y = xy[:, xcolsize:]	
	return (x, y)


def randomInitializeTheta(m, n):
	theta = np.random.normal(0., 1.0, (m,n))
	t = np.absolute(theta).sum(axis = 1)
	theta = 2*theta / t[:, None]
	return theta

def stochasticGradientDescent(x, y, Theta = None, cost_func =squreError,\
 		activation_func=mytanh, alpha = 5, num_iters = 1000, \
		batchsize = 20, shuffle = True, tol = 1e-4, momentum = True, momentumdecay = 0.9):
	if not Theta:
		theta = randomInitializeTheta(x.shape[1], y.shape[1]) 
	else:
		theta = np.array(Theta)	
	if len(y) == batchsize:
		shuffle = False
	velocity = 0.0
	error_history = []
	for i in xrange(num_iters):
		if shuffle:
			(x, y) = shuffleXYTogether(x, y)
		for j in xrange( y.shape[0] / batchsize + 1  ):
			jstart =  j * batchsize
			jend = min(y.shape[0], (j + 1) * batchsize)
			if jstart >= y.shape[0]:
				break
			(cost, gradient) = functionEvaluation(cost_func, activation_func,\
					x[jstart:jend], y[jstart:jend], theta)
			if momentum == False: # no mementum method		
				theta =  theta - alpha * gradient
			else:
				velocity = momentumdecay * velocity - (1. - momentumdecay) * alpha * gradient
				theta = theta + velocity
		(cost,_) = functionEvaluation(cost_func, activation_func,\
					x, y, theta)
		error_history.append(cost)
		if len(error_history) > 2 and np.abs((error_history[-1] - error_history[-2]) / error_history[-1]) < tol:
			break
		print i,cost
	return (theta, error_history)
	
