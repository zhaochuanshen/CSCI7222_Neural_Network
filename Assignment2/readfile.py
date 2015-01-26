import numpy as np
import re

def readfile(filename = "digits_train.txt"):
	f = open(filename)
	i = 0
	result = []
	for line in f:
		if re.search("-", line):
			row = []
			i += 1
		elif re.search('\.', line):
			row += map(float, line.split())
		else:
			label = [j for j, x in enumerate( line.split() ) if x != '0'][0]
			row += [label] 
			result.append(row)
	f.close()
	result = np.array(result)
	#print result.shape
	return result
		
