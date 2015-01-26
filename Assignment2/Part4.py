import readfile
import numpy as np
import perceptron
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def relabeltraindata(data):
	#here I am try to label all of them,
	#I will return N(=10) columns, each column is essentially similar to part 2
	#where label 1(if this is that digit ) or 0 (if not)
	X = data[:, 0:-1]
	symbols =  np.unique(data[:,-1])
	Y = np.zeros( ( data.shape[0], len(symbols) ) )
	
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
	return ( np.matrix(X), np.matrix(Y) , d)

def main():
	traindata = readfile.readfile()
	(trainX, trainY, d) = relabeltraindata(data = traindata)
	ones = np.ones((trainX.shape[0], 1))
	trainX = np.hstack((ones, trainX))
	theta = perceptron.stochasticGradientDescent(trainX, trainY, num_iters = 20)
	
	testdata = readfile.readfile("digits_test.txt")
	testX = testdata[:, 0:-1]
	testY = testdata[:, -1]
	ones = np.ones((testX.shape[0], 1))
	testX = np.hstack((ones, testX))
	intermediateY = np.dot(testX, theta) 
	#note: testX and theta are np matrices
		
	predictY = np.argmax(intermediateY, axis = 1)
	predictY = np.asarray(predictY).squeeze()	
	inv_d = {v: k for k, v in d.items()}
	predictY = [inv_d[i] for i in predictY]
	resultfile = open('confusionmatrix_test_4.txt','w')
	cm = str(confusion_matrix(testY, predictY))
	print cm
	resultfile.write(str(confusion_matrix(testY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(testY, predictY))
	resultfile.close()	
	print 'accuracy: %.4f' % ( ( sum( predictY == testY ) / float(len(predictY)))) 	

if __name__ == "__main__":
	main()


