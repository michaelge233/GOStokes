import numpy as np

seed=324
np.random.seed(seed)
n_point=200

alpha_array_raw=np.random.rand(n_point)*90
beta_array_raw=np.random.rand(n_point)*180
alpha_array=np.copy(alpha_array_raw)
beta_array=np.copy(beta_array_raw)
idx=np.zeros(n_point,dtype=int)

cur_a=0
cur_b=0
for i in range(n_point):
    idx[i]=np.argmin(np.abs(alpha_array_raw-cur_a)+np.abs(beta_array_raw-cur_b))
    cur_a=alpha_array[idx[i]]
    cur_b=beta_array[idx[i]]
    alpha_array_raw[idx[i]]=np.inf
    beta_array_raw[idx[i]]=np.inf
alpha_array=alpha_array[idx]
beta_array=beta_array[idx]
np.savez("./%dpoints_seed%d.npz"%(n_point,seed),alpha_array,beta_array)