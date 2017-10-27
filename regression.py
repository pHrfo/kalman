import numpy as np
import data_processing as dp
import transformation as tr

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


def recursive_update(H_k, y_k, m_prior, P_prior, sq_sigma):
	s_k = np.dot(np.dot(H_k, P_prior), np.transpose(H_k)) + sq_sigma
	K_k = np.dot(np.dot(P_prior, np.transpose(H_k)), s_k)
	L_k = y_k - np.dot(H_k, m_prior)
	m_k = m_prior + np.dot(K_k, L_k)
	P_k = P_prior - np.dot(np.dot(K_k, s_k), np.transpose(K_k))
		
	return m_k, P_k

if __name__ == "__main__":
	data = dp.generate_dataset(np.array([4.75, 10.4]), var=15)
	transforms = [tr.power(0), tr.power(1)]
	H = tr.transform(data[:,0], transforms)
	y = data[:,1]

	m_0 = np.array([0, 0])
	p_0 = np.multiply(np.eye(2), 0.01)
	sq_sigma = 1

	rec_models = recursive_regression(H, y, m_0, p_0, sq_sigma)
	# for i, mod in enumerate(rec_models):
	# 	dp.plot_data(data, "reg_recursive_" + str(i) + ".png", mod[0], arrived=i, noise_var=15)

	for i in range(10):
		model = np.random.multivariate_normal(rec_models[-1][0], rec_models[-1][1])
		dp.plot_data(data, "sample_" + str(i) + ".png", model, arrived=len(data), noise_var=15)

	#m_t, p_t = batch_regression(H, y, m_0, p_0, sq_sigma) 

	#dp.plot_data(data, "reg_output.png", m_t)


