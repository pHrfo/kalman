import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformation import power, transform

def generate_dataset(coeffs, mean=0, var=5, no_of_timesteps=30, transforms = []):
	'''
	coeffs: np.array, represents the coefficients of the original signal
	mean: integer, represents the mean of the noise gaussian
	var: integer, represents the variance of the noise gaussian
	no_of_timesteps: integer, represents the number of measurements that will be out

	Returns a nd.array containing an artificial dataset, i.e. a set of measurements for the original signal
	'''
	# Here we create the array of transformations as specified in transformation.py
	if transforms:
		transformations = transforms
	else:
		transformations = [power(i) for i in range(len(coeffs))]

	# A function that transforms an integer x into an array [1	x] and then multiplies it by the coefficients of the model
	original_signal = lambda x: np.dot(transform(x, transformations), coeffs)

	# Here we sample a gaussian N(mean, var) to get the error values for each time step
	noises = [np.random.normal(mean, var) for _ in range(no_of_timesteps)]

	# We sum the original signal and the noise to get the corrupted dataset
	return np.array([np.array([i, original_signal(i) + noises[i]]) for i in range(no_of_timesteps)])




def plot_data(data, name, reg_model=None, noise_var=5, arrived=0, drift=False, transforms = []):
	'''
	data: np.array, Bidimensional dataset where the first column is the timestep and the second is the value of the measurement at that timestep
	name: str, name and extension of the output file
	reg_model: np.array, parameters of the regression model calculated
	noise_var: number, the variance of the noise gaussian
	arrived: int, number of data points arrived up until this plot

	Plots the data and saves it to a file in the imgs directory 
	'''
	domain = [x for x in range(len(data))]
	df = pd.DataFrame({"time": data[:,0].tolist(), "measurement": data[:,1].tolist()})
	df['arrived'] =  [True]*arrived + [False]*(len(data) - arrived)
	palette = ['#BABABA', '#C74242']
	if arrived == len(data):
		palette = ['#C74242']

	stds = np.array([(noise_var)]*len(data))
	# print(palette)

	sns.set_style("darkgrid")
	sns.lmplot("time", "measurement", df, hue='arrived', fit_reg=False, palette=palette)	

	if reg_model is not None:
		
		if not transforms:
			transforms = [power(i) for i in range(len(data[0]))]
		reg_line = []
		
		if drift:
			# print(reg_model[11], transform(10, transforms))
			for i in range(len(data)):
				print(data[i,0], data[i,1] - np.dot(transform(i, transforms), reg_model[i]))
			reg_line = np.array([np.dot(transform(i, transforms), reg_model[i+1]) for i in range(len(data))])
			#print(reg_line)
		else:
			reg_line = np.array([np.dot(transform(i, transforms), reg_model) for i in range(len(data))])


		plt.fill_between(domain, reg_line - 2*stds/len(domain), reg_line + 2*stds/len(domain), facecolor='red', alpha=.1)
		plt.plot(domain,reg_line)
		plt.show()
		

	
	plt.savefig("imgs/" + name)
	plt.close()





if __name__ == "__main__":
	ds = generate_dataset(np.array([0,1,1]), var=15)
	# plot_data(ds, 'regplot.png')
	plot_data(ds, 'regplot_w_regression.png', np.array([-1.48680659, 1.30027409, 0.98733384]), noise_var=15, arrived=len(ds)-8)
