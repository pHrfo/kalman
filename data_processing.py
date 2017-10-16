import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformation import power, transform

def generate_dataset(coeffs, mean=0, var=5, no_of_timesteps=30):
	'''
	coeffs: np.array, represents the coefficients of the original signal
	mean: integer, represents the mean of the noise gaussian
	var: integer, represents the variance of the noise gaussian
	no_of_timesteps: integer, represents the number of measurements that will be out

	Returns a nd.array containing an artificial dataset, i.e. a set of measurements for the original signal
	'''
	# Here we create the array of transformations as specified in transformation.py
	transformations = [power(i) for i in range(len(coeffs))]

	# A function that transforms an integer x into an array [1	x] and then multiplies it by the coefficients of the model
	original_signal = lambda x: np.dot(transform(x, transformations), coeffs)

	# Here we sample a gaussian N(mean, var) to get the error values for each time step
	noises = [np.random.normal(mean, var) for _ in range(no_of_timesteps)]

	# We sum the original signal and the noise to get the corrupted dataset
	return np.array([np.array([i, original_signal(i) + noises[i]]) for i in range(no_of_timesteps)])




def plot_data(data, name, reg_model=None):
	'''
	data: np.array, Bidimensional dataset where the first column is the timestep and the second is the value of the measurement at that timestep
	name: str, name and extension of the output file

	Plots the data and saves it to a file in the imgs directory 
	'''
	df = pd.DataFrame({"time": data[:,0].tolist(), "measurement": data[:,1].tolist()})

	sns.set_style("darkgrid")
	sns.lmplot("time", "measurement", df, fit_reg=False)	

	if reg_model is not None:

		transforms = [power(i) for i in range(len(reg_model))]
		reg_line = [np.dot(transform(i, transforms), reg_model) for i in range(len(data))]
		plt.plot(reg_line)
	
	plt.savefig("imgs/" + name)





if __name__ == "__main__":
	ds = generate_dataset(np.array([0,1,1]), var=15)
	plot_data(ds, 'regplot.png')
	plot_data(ds, 'regplot_w_regression.png', np.array([-1.48680659, 1.30027409, 0.98733384]))
