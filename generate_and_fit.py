import sys
import json
from generate_data import *
from ufie import *

# Purpose 1: interpolate and extrapolate function given its data points (simple regression)
# Purpose 2: interpolate and extrapolate function given its data points and those from similar functions (hence generating many polynomials)
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    with open(sys.argv[1], 'r') as input_file:
        configs = json.load(input_file)

    data = generate_polynomials(configs['data'])
    ufie = UFIE(configs['data']['coefficients'], data, configs['model'])
    ufie.converge()

# Tasks
# - incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
# - what about customizing placement of activation functions, as well as choosing optimization/loss function?
# - what about putting x_k and y_(k-1) on separate networks?
# - maybe show truth functions for general case (in that case, just provide the raw input data itself instead of calculating truth functions)