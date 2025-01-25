import numpy as np
import torch
import argparse

# TODO: move generate_polynomial into ufie.py

np.random.seed(2)

def generate_polynomial(opt, coefficients):
    # Generate input data
    stop = opt.start + opt.length * opt.separation
    shift = np.random.randint(-opt.shift, opt.shift + 1, opt.number).reshape(opt.number, 1)
    x = np.array(range(opt.start, stop, opt.separation)) + shift
    # Generate output data
    equation_scale = opt.equation_scale * np.random.rand(opt.number).reshape(opt.number, 1)
    data = np.zeros(x.shape)
    for k, c in enumerate(coefficients):
        term_scale = opt.term_scale * np.random.rand(opt.number).reshape(opt.number, 1)
        data += (opt.total_scale + equation_scale + term_scale) * c * np.pow(x, k)
    # Display & save
    #print(data[:10])
    torch.save(data.astype('float64'), open('traindata.pt', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=100, help='number of datasets')
    parser.add_argument('--length', type=int, default=100, help='number of samples per dataset')
    parser.add_argument('--start', type=float, default=0, help='first sample')
    parser.add_argument('--separation', type=float, default=1, help='separation between one sample and the next')
    parser.add_argument('--shift', type=int, default=0, help='limit of random shift')
    parser.add_argument('--total_scale', type=float, default=1, help='component of scale that applies to all equations')
    parser.add_argument('--equation_scale', type=float, default=0, help='limit of random component of scale that applies to the full equation')
    parser.add_argument('--term_scale', type=float, default=0, help='limit of random component of scale that applies to one term in the equation')
    opt = parser.parse_args()
    
    # temporary assignments
    opt.shift = 4
    coefficients = [0, 0, 1]

    generate_polynomial(opt, coefficients)
