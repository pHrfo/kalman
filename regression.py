import numpy as np
import data_processing as dp
import transformation as tr
import math

def batch_regression(H, y, m_0, P_0, sq_sigma):
	'''
	H: np.array
	y: np.array
	m_0: np.array
	P_0: np.array
	sq_sigma: float

	Performs batch linear regression for the data H with targets y, as specified in the book "Bayesian Filtering & Smoothing" by Simo Särkkä.
	Check the book for details on the formulas and the specification of the variables.
	'''

	P_T = np.linalg.inv(np.linalg.inv(P_0) + \
			np.multiply(1/sq_sigma, np.dot(np.transpose(H), H)))

	aux = np.multiply(1/sq_sigma, np.dot(np.transpose(H),y)) + np.dot(np.linalg.inv(P_0), m_0)

	m_t = np.dot(P_T, aux)

	return m_t, P_T

def recursive_regression(H, y, m_0, P_0, sq_sigma):
	m_prior, P_prior = m_0, P_0

	output = [(m_prior, P_prior)]
	
	for H_k, y_k in zip(H, y):
		m_prior, P_prior = recursive_update(H_k, y_k, m_prior, P_prior, sq_sigma)
		output.append((m_prior, P_prior))

	return output

def recursive_regression_with_drift(H, y, m_0, P_0, sq_sigma, Q):
	m_prior, P_prior = m_0, P_0

	output = [(m_prior, P_prior)]

	for H_k, y_k in zip(H, y):
		m_params = m_prior
		P_params = P_prior + Q
		m_prior, P_prior = recursive_update(H_k, y_k, m_params, P_params, sq_sigma)
		output.append((m_prior, P_prior))

	return output


def recursive_update(H_k, y_k, m_prior, P_prior, sq_sigma):
	s_k = np.dot(np.dot(H_k, P_prior), np.transpose(H_k)) + sq_sigma
	K_k = np.dot(np.dot(P_prior, np.transpose(H_k)), 1/s_k)
	L_k = y_k - np.dot(H_k, m_prior)
	m_k = m_prior + np.dot(K_k, L_k)
	P_k = P_prior - np.outer(np.multiply(K_k, s_k), np.transpose(K_k))
		
	return m_k, P_k

if __name__ == "__main__":
	# transforms = [tr.power(0), tr.power(1)]
	# data = dp.generate_dataset(np.array([0, 1]), var=20, no_of_timesteps=50, transforms=transforms)
	
	transforms = [lambda vec: np.sin(vec*2*math.pi)]
	data = dp.generate_dataset(np.array([1]), var=0.5, no_of_timesteps=50, transforms=transforms)

	transforms = [tr.power(0), tr.power(1)]
	H = tr.transform(data[:,0], transforms)
	y = data[:,1]

	m_0 = np.array([0, 0])
	p_0 = np.multiply(np.eye(2), 5)
	Q = np.multiply(np.eye(2), 1)
	sq_sigma = 1

	rec_models = recursive_regression_with_drift(H, y, m_0, p_0, sq_sigma, Q)
	# for i, mod in enumerate(rec_models):
	# 	dp.plot_data(data, "reg_recursive_" + str(i) + ".png", mod[0], arrived=i, noise_var=15)

	models = []
	for rec_model in rec_models:

		model = rec_model[0]
		models += [model]

	print(data)
	dp.plot_data(data, 
				"reg_drift.pdf",
				models, 
				arrived=len(data), 
				noise_var=15, 
				drift=True, 
				transforms=transforms)

	#m_t, p_t = batch_regression(H, y, m_0, p_0, sq_sigma) 

	#dp.plot_data(data, "reg_output.png", m_t)


