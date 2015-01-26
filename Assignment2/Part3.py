import readfile
import numpy as np
import perceptron
from sklearn.metrics import confusion_matrix
from sklearn import metrics

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
	X = data[:, 0:-1]
	Y = data[:, -1]
	return (X, Y)

def main():
	traindata = readfile.readfile()
	(trainX, trainY) = relabel(data = traindata)
	ones = np.ones((trainX.shape[0], 1))
	trainX = np.hstack((ones, trainX))
	theta = perceptron.stochasticGradientDescent(trainX, trainY, num_iters = 20)
	
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


