import pandas as pd
import numpy as np
from sklearn import linear_model
import statsmodels.api as sm

def main():
	dataset = pd.read_csv("assign1_data.txt", delimiter = r'\s+')
	X = dataset.iloc[:, 0:2]
	Y = dataset.iloc[:, 2]
	Z = dataset.iloc[:, 3]

	#I am using the sklearn package
	clf = linear_model.LinearRegression()
	clf.fit (X, Y)
	print "using sklearn, coefficients are "
	print clf.coef_
	print "using sklearn, intercept is "
	print clf.intercept_

	#I am using the pandas to the least square regession
	x = sm.add_constant(X)
	est = sm.OLS(Y, x)
	est = est.fit()
	print "results from statmodels"
	print est.summary()

if __name__ == "__main__":
	main()



