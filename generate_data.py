import numpy as np
import torch

def generate_inputs_old(configs):
    # Make the first dataset a base unshifted set, and make the rest be randomly shifted sets
    stop = configs['start'] + configs['samples'] * configs['separation']
    shift = np.concat([np.array([0]), (np.random.randint(-configs['shift'], configs['shift'] + 1, configs['datasets'] - 1))])
    return np.array(range(configs['start'], stop, configs['separation'])) + shift.reshape(configs['datasets'], 1)

def generate_polynomials_old(configs, save_file=True):
    x = generate_inputs_old(configs)
    data = np.dstack([x, np.zeros(x.shape)])
    # Make the first dataset a base unscaled set, and make the rest be randomly scaled sets
    equation_scale = np.concat([np.array([0]), (configs['equation_scale'] * np.random.rand(configs['datasets'] - 1))])
    for k, c in enumerate(configs['coefficients']):
        term_scale = np.concat([np.array([0]), (configs['term_scale'] * np.random.rand(configs['datasets'] - 1))])
        scale = configs['base_scale'] + equation_scale + term_scale
        data[:, :, 1] += c * np.pow(x, k) * scale.reshape(configs['datasets'], 1)
    if save_file:
        torch.save(data.astype('float64'), open('polynomial_data.pt', 'wb'))
    return data

def generate_inputs(configs):
    stop_interp = configs['start'] + configs['specific_samples'] * configs['separation']
    specific = np.array(range(configs['start'], stop_interp, configs['separation']))
    # Make the first dataset a base unshifted set, and make the rest be randomly shifted sets
    start_extrap = configs['start']
    stop_extrap = start_extrap + (configs['general_samples']) * configs['separation']
    shift_extrap = np.concat([np.array([0]), (np.random.randint(-configs['shift'], configs['shift'] + 1, configs['datasets'] - 1))])
    general = np.array(range(start_extrap, stop_extrap, configs['separation'])) + shift_extrap.reshape(configs['datasets'], 1)
    return specific, general

def generate_polynomials(configs, save_file=True):
    x_s, x_g = generate_inputs(configs)
    data_s = np.vstack([x_s, np.zeros(x_s.shape)]).T
    data_g = np.dstack([x_g, np.zeros(x_g.shape)])
    # Make the first dataset a base unscaled set, and make the rest be randomly scaled sets
    equation_scale = np.concat([np.array([0]), (configs['equation_scale'] * np.random.rand(configs['datasets'] - 1))])
    for k, c in enumerate(configs['coefficients']):
        data_s[:, 1] += c * np.pow(x_s, k)
        term_scale = np.concat([np.array([0]), (configs['term_scale'] * np.random.rand(configs['datasets'] - 1))])
        scale = configs['base_scale'] + equation_scale + term_scale
        data_g[:, :, 1] += c * np.pow(x_g, k) * scale.reshape(configs['datasets'], 1)
    if save_file:
        torch.save(data_g.astype('float64'), open('polynomial_data.pt', 'wb'))
    return data_s, data_g

def import_polynomials():
    return torch.load('polynomial_data.pt')