import numpy as np
import torch

def generate_polynomials(configs, save_file=True):
    # Generate input data: a base unshifted set, and many randomly shifted sets
    stop = configs['start'] + configs['samples'] * configs['separation']
    shift = np.concat([np.array([0]), (np.random.randint(-configs['shift'], configs['shift'] + 1, configs['datasets'] - 1))])
    x = np.array(range(configs['start'], stop, configs['separation'])) + shift.reshape(configs['datasets'], 1)
    # Generate output data: a base unscaled set, and many randomly scaled sets
    equation_scale = np.concat([np.array([0]), (configs['equation_scale'] * np.random.rand(configs['datasets'] - 1))])
    data = np.dstack([x, np.zeros(x.shape)])
    for k, c in enumerate(configs['coefficients']):
        term_scale = np.concat([np.array([0]), (configs['term_scale'] * np.random.rand(configs['datasets'] - 1))])
        scale = configs['base_scale'] + equation_scale + term_scale
        data[:, :, 1] += c * np.pow(x, k) * scale.reshape(configs['datasets'], 1)
    if save_file:
        torch.save(data.astype('float64'), open('polynomial_data.pt', 'wb'))
    return data

def import_polynomials():
    return torch.load('polynomial_data.pt')