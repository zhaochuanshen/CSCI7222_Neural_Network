problem description


http://www.cs.colorado.edu/~mozer/Teaching/syllabi/DeepLearning2015/assignments/assignment2.html

OR:


Goal
The goal of this assignment is to build your first neural network to process a real data set:  handprinted digits.  You should be able to re-use the code you wrote for the perceptron last week.
Data Set
The data set consists of handprinted digits, originally provided  by Yann Le Cun.  Each digit is described by a 14x14 pixel array. Each pixel has a grey level with value ranging from 0 to 1.   The data is split between two files, a training set  that contains the examples used for training your neural network, and a test set that contains examples you'll use to evaluate the trained network.  Both training and test sets are organized the same way. Each file begins with 250 examples of the digit "0", followed by 250 examples of the digit "1", and so forth up to the digit "9". There are thus 2500 examples in the training set and another 2500 examples in the test set.  

Each digit begins with a label on a line by itself, e.g., "train4-17", where the "4" indicates the target digit, and the "17" indicates the example number. The next 14 lines contain 14 real values specifying pixel intensities for pixels in the corresponding row. Finally, there is a line with 10 integer values indicating the target.  The vector "1 0 0 0 0 0 0 0 0 0" indicates the target 0; the vector "0 0 0 0 0 0 0 0 0 1" indicates the target 9.
Part 1
Write code to read in a file -- either train or test -- and build a data structure containing the input-output examples.  Although the digit pixels lie on a 14x14 array, the input to your network will be a 196-element vector.  The arrangement into rows is to help our eyes see the patterns.  You might also write code to visualize individual examples.

Note: we will use the same data set next week when we implement back propagation, so the utility functions you write in Part 1 will be reused. It's worth writing some nice code to step through examples and visualize the patterns.
Part 2
Train a perceptron to discriminate 2 from not-2. That is, lump together all the examples of the digits {0, 1, 3, 4, 5, 6, 7, 8, 9}.  You will have 250 examples of the digit 2 in your training file and 2250 examples of not-2.  Assess performance on the test set and report false negative and false positive rates.  The false-negative rate is the proportion of not-2's which are classified as 2's. The false-positive rate is the proportion of 2's which are classified as not-2's.

Note: The perceptron algorithm is an "on line" algorithm:  you adjust the weights after each example is presented.  Next week, we're going to change your code to implement back propagation. Back propagation can be run in an "on line" mode, or "batch".  To anticipate next week's work, you might want to set up your code this week to process minibatches of between 1 and 2500 examples. You will compute the summed weight update for all examples in the minibatch, and then update the weights at the end of the minibatch.  (A minibatch with 1 example corresponds to the on-line algorithm; a minibatch with 2500 examples corresponds to the batch algorithm.)

Remember an important property of the perceptron algorithm:  it is guaranteed to converge only if there is a setting of the weights that will classify the training set perfectly.  (The learning rule corrects errors. When all examples are classified correctly, the weights stop changing.)  With a noisy data set like this one, the algorithm will not find an exact solution.  Also remember that the perceptron algorithm is not performing gradient descent. Instead, it will jitter around the solution continually changing the weights from one iteration to the next. The weight changes will have a small effect on performance, so you'll see training set performance jitter a bit as well.
Part 3 (Optional)
Train a perceptron to discriminate 8 from 0. You will have 500 training examples and 500 test examples.
Part 4 (Optional)
Train a perceptron with 10 outputs to classify the digits 0-9 into distinct classes. Using the test set, construct a confusion matrix.  The 10x10 confusion matrix will specify the frequency by which each input digit is assigned to each output class.  The diagonal of this matrix will indicate correct classifications.


