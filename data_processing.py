import numpy as np
import pandas as pd
import seaborn as sns
from transformation import power, transform

def generate_dataset(coeffs, mean=0, var=5, no_of_timesteps=30):
	'''
	coeffs: np.array, represents the coefficients of the original signal
	mean: integer, represents the mean of the noise gaussian
	var: integer, represents the variance of the noise gaussian
	no_of_timesteps: integer, represents the number of measurements that will be out

	Returns a nd.array containing an artificial dataset, i.e. a set of measurements for the original signal
	'''

	transformations = [power(i) for i in range(len(coeffs))]

	original_signal = lambda x: np.dot(transform(x, transformations), coeffs)
	
	noises = np.random.normal(mean, var, no_of_timesteps)

	return np.array([np.array([i, original_signal(i) + noises[i]]) for i in range(no_of_timesteps)])

def plot_data(data, name):
	'''
	data: np.array, Bidimensional dataset where the first column is the timestep and the second is the value of the measurement at that timestep
	name: str, name and extension of the output file

	Plots the data and saves it to a file in the imgs directory 
	'''

	df = pd.DataFrame({"time": data[:,0].tolist(), "measurement": data[:,1].tolist()})
	plt = sns.lmplot("time", "measurement", df, fit_reg=False)
	plt.savefig("imgs/" + name)
