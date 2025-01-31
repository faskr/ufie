import sys
import json
from generate_data import *
from ufie import *

# Purpose 1: interpolate and extrapolate function given its data points (simple regression - generalization to other data points)
# Purpose 2: interpolate and extrapolate function given its data points and those from similar functions (hence generating many polynomials - generalization to other polynomials)
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    with open(sys.argv[1], 'r') as input_file:
        configs = json.load(input_file)

    data = generate_polynomials(configs['data'])
    ufie = UFIE(data, configs['model'])
    ufie.converge()

# Ideas
# - Incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
# - Customize placement of activation functions, as well as choice of optimization and/or loss function
# - Put x_k and y_(k-1) on separate networks