import numpy as np
import data_processing as dp
import transformation as tr
import math
from matplotlib import pyplot as plt

def kalman_filter(Y, m_0, P_0, process_noise, measurement_noise, prediction_step, update_step, make_A=None, make_H=None, delta_t=None):
	'''
	Y = np.array, the array of measurements
	m_0 = np.array, the mean of the prior distribution over the states
	P_0 = np.array, the covariance matrix of the prior distribution over the states
	process_noise = np.array, the noise for the transition phase of the HMM (the transition of states)
	measurement_noise = np.array, the noise for the emission phase of the HMM (the emission of a measurement from a given hidden state)
	make_A = function, a function that creates the transition matrix of the dynamic model
	make_H = function, a function that creates the measurement model matrix
	prediction_step = a function that performs the prediction step for the problem, given parameters make_A, P_prior, m_prior and process noise
	update_step = a function that performs the update step for the problem, given parameters y_k, make_H, m_pred, P_pred and measurement_noise
	'''
	m_prior, P_prior = m_0, P_0
	output = [(m_prior, P_prior)]

	for i, y_k in enumerate(Y):

		m_pred, P_pred = prediction_step(m_prior, P_prior, process_noise, make_A, delta_t)
		m_post, P_post = update_step(y_k, m_pred, P_pred, measurement_noise, make_H)

		output += [(m_post, P_post)]

		m_prior, P_prior = m_post, P_post

	return output

############################## GENERAL KALMAN FILTER ###################################


def general_kalman_prediction(m_prior, P_prior, process_noise, make_A=None, delta_t=None,):
	A = make_A(delta_t)

	m_pred = np.dot(A, m_prior)
	P_pred = np.dot(np.dot(A, P_prior), np.transpose(A)) + process_noise

	return m_pred, P_pred

def general_kalman_update(y_k, m_pred, P_pred, measurement_noise, make_H=None):
	H_k = make_H()

	v_k = y_k - np.dot(H_k, m_pred)
	S_k = np.dot(np.dot(H_k, P_pred), np.transpose(H_k)) + measurement_noise

	K_k = np.dot(P_pred, np.dot(H_k, np.linalg.inv(S_k)))
	m_post = m_pred + np.dot(K_k, v_k)
	P_post = P_pred - np.outer(np.dot(K_k, S_k), np.transpose(K_k))

	return m_post, P_post

################################# GAUSSIAN RANDOM WALK #################################

def random_walk_prediction(m_prior, P_prior, process_noise, make_A=None, delta_t=None):
	m_pred = m_prior
	P_pred = P_prior + process_noise

	return m_pred, P_pred

def random_walk_update(y_k, m_pred, P_pred, measurement_noise, make_H=None):
	m_post = m_pred + P_pred*(y_k - m_pred)/(P_pred + measurement_noise)
	P_post = P_pred - P_pred ** 2 / (P_pred + measurement_noise)

	return m_post, P_post

def generate_random_walk(measurement_noise, process_noise, no_of_timesteps=50):
	prior = 0
	data = []
	for k in range(no_of_timesteps):
		x_k = prior + np.random.normal(0, process_noise)
		y_k = x_k + np.random.normal(0, measurement_noise)

		data.append(y_k)
		prior = x_k

	return data

def make_A_random_walk(delta_t=None):
	return 1

def make_H_random_walk():
	return 1


if __name__ == "__main__":
	y = generate_random_walk(measurement_noise = 1, process_noise = 1, no_of_timesteps = 156)

	plt.plot(y, 'r.')

	posteriors = kalman_filter(y, [[5]], [[0.5]], [[1]], [[1]],
								 general_kalman_prediction,
								 general_kalman_update,
								 make_A=make_A_random_walk,
								 make_H=make_H_random_walk)
	x = [model[0][0][0] for model in posteriors]
	
	# posteriors = kalman_filter(y, 0, 0.5, 1, 1,
	# 							 random_walk_prediction,
	# 							 random_walk_update)

	# x = [model[0] for model in posteriors]

	plt.plot(x)
	plt.savefig('imgs/kalman.pdf')



################################ WASTEWATER PLANT DATA ##########################3

def wastewater_predict(m_prior, P_prior, process_noise, make_A=None, delta_t=None):
	A = make_A_wastewater(1)
	m_pred = np.dot(A, m_prior)

	sum_fac = np.array([[process_noise[0][0]/3, process_noise[0][0]/2], [process_noise[0][0]/3, process_noise[0][0]]])
	P_pred = np.dot(np.dot(A, P_prior), np.transpose(A)) + sum_fac

	return m_pred, P_pred

def wastewater_update(y_k, m_pred, P_pred, measurement_noise, make_H=None):
	H_k = make_H_wastewater()
	v_k = y_k - np.dot(H_k, m_pred)
	# print(v_k)
	S_k = np.dot(np.dot(H_k, P_pred), np.transpose(H_k)) + measurement_noise
	K_k = np.multiply(np.dot(P_pred, H_k), 1/S_k)

	m_post = m_pred + np.dot(K_k, v_k)
	P_post = P_pred - np.outer(np.dot(K_k, S_k), np.transpose(K_k))

	return m_post, P_post

def make_A_wastewater(delta_t=1):
	return np.array([[1, delta_t], [0, 1]])

def make_H_wastewater():
	return np.array([1, 0])