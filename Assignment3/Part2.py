import readfile
import numpy as np
import learn 
import matplotlib.pyplot as plt


def relabeltraindata(data):
	#here I am try to label all of them,
	#I will return N(=10) columns, each column is essentially similar to assignment 2
	#where label 1(if this is that digit ) or -1 (if not)
	X = data[:, 0:-1]
	symbols =  np.unique(data[:,-1])
	Y =  np.zeros( ( data.shape[0], len(symbols) ) )
	
	d = {}
	numofelementindict = 0	
	# this dictionary records which column of Y it should go
	# this looks reductant, but may be useful if we want to generalize to something else
	for i, row in enumerate(data):
		try:
			Y[i, d[ row[-1]]] = 1
		except KeyError:
			d[ row[-1] ] = numofelementindict
			numofelementindict += 1
			Y[i, d[ row[-1]]] = 1 
	return ( X, Y, d)

def main():
	traindata = readfile.readfile()
	(trainX, trainY, d) = relabeltraindata(data = traindata)
	ones = np.ones((trainX.shape[0], 1))
	trainX = np.hstack((ones, trainX))
	(theta, error_history) = learn.stochasticGradientDescent(
		trainX, trainY, activation_func = learn.softmax, cost_func = learn.cross_entropy,\
		momentum=True, alpha = 1, batchsize = 200)

	plt.plot(error_history)
	plt.xscale('log')
	plt.savefig("error.pdf")
	print error_history

	testdata = readfile.readfile('digits_test.txt')
	testX = testdata[:, 0:-1]
	testY = testdata[:, -1]
	ones = np.ones((testX.shape[0], 1))
	testX = np.hstack((ones, testX))

	intermediateY = np.dot(testX, theta) 
		
	predictY = np.argmax(intermediateY, axis = 1)
	predictY = np.ravel(predictY)
	inv_d = {v: k for k, v in d.items()}
	predictY = [inv_d[i] for i in predictY]
	
	print 'accuracy: %.4f' % ( ( sum( predictY == testY ) / float(len(predictY)))) 		


if __name__ == "__main__":
	main()		
