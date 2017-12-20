import numpy as np 

def transform(vector, transformations):
	'''
	vector: np.array containing the data
	transformations: a list of functions that will be applied to the data vector

	output: a matrix whose columns are the vector transformed by the functions in the transformations list.
		Note that the dimensions of the output matrix are (N x M) where N is the size of the dataset and M is the number of transformation functions passed.

	Example:

		>>> v = np.array([1,2,3])
		>>> transformations = [power(0), power(1), power(2)]
		>>> transform(v, transformations)
		array([[1, 1, 1],
		       [1, 2, 4],
		       [1, 3, 9]])

	'''
	return np.transpose(np.array([transform(vector) for transform in transformations]))

def power(p):
	return lambda vec: vec ** p

if __name__ == "__main__":
	v = np.array([1,2,3])
	transformations = [power(0), power(1), power(2)]

	print(transform(v, transformations))

