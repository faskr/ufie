import numpy as np

def generate_polynomial(opt, coefficients):
    # Generate input data
    stop = opt.start + opt.samples * opt.separation
    shift = np.random.randint(-opt.shift, opt.shift + 1, opt.datasets).reshape(opt.datasets, 1)
    x = np.array(range(opt.start, stop, opt.separation)) + shift
    # Generate output data
    equation_scale = opt.equation_scale * np.random.rand(opt.datasets).reshape(opt.datasets, 1)
    data = np.dstack([x, np.zeros(x.shape)])
    for k, c in enumerate(coefficients):
        term_scale = opt.term_scale * np.random.rand(opt.datasets).reshape(opt.datasets, 1)
        data[:, :, 1] += (opt.base_scale + equation_scale + term_scale) * c * np.pow(x, k)
    #print(data[:10])
    return data