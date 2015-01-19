This is my solution to the first assignment.

Part 1: I used both the sklearn and the statsmodels packages to the least squared

Part 2: My least squared, no regularization for least sqaure

Part 3: My logistic regression, with very week regularization

Part 4: "sort of" cross validation.

===

website of the problem:
http://www.cs.colorado.edu/~mozer/Teaching/syllabi/DeepLearning2015/assignments/assignment1.html


Goal
The goal of this assignment is to introduce neural networks in terms of ideas you are already familiar with:  linear regression and linear-threshold classification.
Part 1
Consider the following table that describes a relationship between two input variables (x1,x2) and an output variable (y).

Part 2
Using the LMS algorithm, write a program that determines the coefficients {w1,w2,b} via incremental updating and multiple passes through the training data. You will need to experiment with updating rules (online, batch, minibatch), step sizes (i.e., learning rates), stopping criteria, etc.

Be prepared to share some of the lessons you learned from these experiments with the class next week.


Part 3
Turn this data set from a regreesion problem into a classification problem simply by using the sign of y (+ or -) as representing one of two classes. In the data set you download, you'll see a variable z that represents this binary (0 or 1) class.  Use the perceptron learning rule to solve for the coefficients {w1, w2, b} of this classification problem. 

Note: if you do this right, your solution to Part 3 will require only a few lines of code added to the code you wrote for Part 2.

Part 4
In machine learning, we really want to train a model based on some data and then expect the model to do well on "out of sample" data. Try this with the code you wrote for Part 3:  Train the model on the first {25, 50, 75} examples in the data set and test the model on the final 25 examples. How does performance on the test set vary with the amount of training data?
