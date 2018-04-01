import numpy as np
import kalman_filtering as klm
import math
from matplotlib import pyplot as plt
from scipy.io import loadmat

data = loadmat("data/wastewater_data")
X = data['T']
zeros_lim = 200
zeros_qtd = 100
Y = [y[0] for y in data['Y']]
print(Y[zeros_lim:zeros_lim+zeros_qtd])
Y = [y[0] for y in data['Y'][:zeros_lim]] + [None for x in range(zeros_qtd)] + [y[0] for y in data['Y'][zeros_lim + zeros_qtd:]]
print(Y[zeros_lim:zeros_lim+zeros_qtd])

models = klm.kalman_filter(Y=np.array(Y), 
						 m_0=np.array([59.5080566406,0]),
						 P_0=np.multiply(np.eye(2), 0.001),
						 process_noise=np.multiply(np.eye(2), 0.001), 
						 measurement_noise=4,
						 prediction_step=klm.wastewater_predict,
						 update_step=klm.wastewater_update,
						 make_A=klm.make_A_wastewater,
						 make_H=klm.make_H_wastewater,
						 delta_t=1)

range_ = [0,len(Y)]
# range_ = [zeros_lim-15, zeros_lim + zeros_qtd + 50]
X = [model[0][0] for model in models]
plt.plot([x for x in range(len(Y[range_[0]:range_[1]]))], Y[range_[0]:range_[1]], 'r.')
plt.plot([x for x in range(len(X[range_[0]:range_[1]]))], X[range_[0]+1:range_[1]+1] , '.-')
plt.savefig('imgs/wastewater.pdf')