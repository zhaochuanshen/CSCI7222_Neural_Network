import readfile
import numpy as np
import learn 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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
	(theta, error_history) = learn.neutralnetwork(
		trainX, trainY)

	plt.plot(error_history)
	plt.xscale('log')
	plt.savefig("error.pdf")
	print error_history


	testdata = readfile.readfile('digits_test.txt')
	testX = testdata[:, 0:-1]
	testY = testdata[:, -1]
	
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
	inv_d = {v: k for k, v in d.items()}
	predictY = [inv_d[i] for i in predictY]
	resultfile = open('confusionmatrix_test_3.txt','w')	
	cm = str(confusion_matrix(testY, predictY))
	print cm
	resultfile.write(str(confusion_matrix(testY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(testY, predictY))
	resultfile.close()		
	
	
		
	print 'accuracy: %.4f' % ( ( sum( predictY == testY ) / float(len(predictY)))) 		

if __name__ == "__main__":
	main()		
