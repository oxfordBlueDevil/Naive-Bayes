# Gaussian Naive Bayes

- Here is a Python class that implements Naive Bayes Classification using a simple
gaussian likelihood function. It should conform to the following specification, and
you should demonstrate this functionality in a Python shell (attach the output) or
IPython notebook:
	1. A new classifier can be instantiated with the function call
	NaiveBayes(some_data_file.csv) , where some_data_file.csv is formatted
	as feature_1, feature_2, … , feature_N, category and contains the initial
	set of observations to classify on . You can assume the features are all
	floating point numbers and the categories are integers.
	2. it has a method called predict which takes an array of length N (where N
	is the number of features) and returns the most likely output category.
	3. it has a method called observe which takes two parameters; an array of
	length N and an integer category. This method should add this data to the
	classifier and revise its future predictions.