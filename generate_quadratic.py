import numpy as np
import torch

np.random.seed(2)

L = 100
N = 100

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4, 4, N).reshape(N, 1)
data = np.pow(x, 2).astype('float64')
torch.save(data, open('traindata.pt', 'wb'))
