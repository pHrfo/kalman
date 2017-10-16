import numpy as np
import data_processing as dp
import transformation as tr

def batch_regression(H, y, m_0, P_0, sq_sigma):
	'''
	'''

	P_T = np.linalg.inv(np.linalg.inv(P_0) + \
			np.multiply(1/sq_sigma, np.dot(np.transpose(H), H)))

	aux = np.multiply(1/sq_sigma, np.dot(np.transpose(H),y)) + np.dot(np.linalg.inv(P_0), m_0)

	m_t = np.dot(P_T, aux)

	return m_t, P_T

if __name__ == "__main__":
	data = dp.generate_dataset(np.array([0, 1, 1]), var=80)
	transforms = [tr.power(0), tr.power(1), tr.power(2)]
	H = tr.transform(data[:,0], transforms)
	y = data[:,1]

	m_0 = np.array([0, 0, 0])
	p_0 = np.eye(3)
	sq_sigma = 80

	m_t, p_t = batch_regression(H, y, m_0, p_0, sq_sigma) 

	dp.plot_data(data, "reg_output.png", m_t)


