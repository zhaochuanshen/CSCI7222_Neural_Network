import readfile
import numpy as np
import learn 
import matplotlib.pyplot as plt

def relabel(data, target1 = 0, target2 = 8):
	# as I will try to distinguish two digits, (for example 2 and 8)
	# I don't care other digits at all 
	f = lambda row: True if row[-1] == target1 or row[-1] == target2 else False 
	data = np.array(filter(f, data))
	for row in data:
		if row[-1] == target1:
			row[-1] = 1
		if row[-1] == target2:
			row[-1] = 0	
	X = np.array(data[:, 0:-1])
	Y = np.matrix(data[:, -1]).T
	compY = 1 - Y
	Y= np.hstack( (Y, compY) )
	Y = np.array(Y)
	return (X, Y)



def main():
	traindata = readfile.readfile()
	(trainX, trainY) = relabel(data = traindata)
	print trainX.shape
	print trainY.shape
	(theta, error_history) = learn.neutralnetwork(trainX, trainY,num_iters = 1000)

	plt.plot(error_history)
	plt.xscale('log')
	plt.savefig("error.pdf")
	print error_history


	testdata = readfile.readfile('digits_test.txt')
	testX, testY = relabel(testdata)
	
	z= [None]
	ones = np.ones((testX.shape[0], 1))
	a = [np.hstack((ones, testX))]

	for i in xrange(len(theta) - 1):
		z.append(np.dot(a[-1], theta[i]))
		tempa = learn.sigmoid(z[-1])
		ones = np.ones((tempa.shape[0],1))
		tempa = np.hstack((ones, tempa))
		a.append(tempa)
	intermediateY = np.exp( np.dot(a[-1], theta[-1]) )
		
		
	predictY = np.argmax(intermediateY, axis = 1)
	predictY = np.ravel(predictY)
	
	testY = np.argmax(testY, axis = 1)
	testY = np.ravel(testY)
	
	print 'accuracy: %.4f' % ( ( sum( predictY == testY ) / float(len(predictY)))) 		

if __name__ == "__main__":
	main()		
