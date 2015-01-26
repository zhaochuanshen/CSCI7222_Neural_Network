import readfile
import numpy as np
import perceptron
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def relabel(data, target = 2):
	# as I will try to distinguish a digit, (for example 2) from all other
	# this function relabel data, label target 2 as 1, and all other as 0
	for row in data:
		if row[-1] == target:
			row[-1] = 1
		else:
			row[-1] = 0	
	X = data[:, 0:-1]
	Y = data[:, -1]
	return (X, Y)


def main():
	traindata = readfile.readfile()
	(trainX, trainY) = relabel(data = traindata)
	ones = np.ones((trainX.shape[0], 1))
	trainX = np.hstack((ones, trainX))
	theta = perceptron.stochasticGradientDescent(trainX, trainY, num_iters = 30)
	
	testdata = readfile.readfile("digits_test.txt")
	(testX, testY) = relabel(data = testdata)	
	ones = np.ones((testX.shape[0], 1))
	testX = np.hstack((ones, testX))
	intermediateY = np.dot(testX, theta) 
	#note: testX is np array, theta is np matrix
	
	predictY = [1 if item > 0 else 0 for item in intermediateY]
	resultfile = open('confusionmatrix_test.txt','w')
	cm = str(confusion_matrix(testY, predictY))
	print cm
	resultfile.write(str(confusion_matrix(testY, predictY)))
	resultfile.write('\n\n\n')
	resultfile.write(metrics.classification_report(testY, predictY))
	resultfile.close()	
	
	print 'accuracy: %.4f' % ( ( sum( predictY == testY ) / float(len(predictY)))) 	

if __name__ == "__main__":
	main()


