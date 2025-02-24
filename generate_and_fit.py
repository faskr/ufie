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

    data_s, data_g = generate_polynomials(configs['data'])
    ufie = UFIE(data_s, data_g, configs['model'])
    ufie.converge()

# Priority Tasks
# - Make data_g an optional parameter
# - Exclude specific dataset (at index 0) from test data, and make it a third category by itself, so that the test data will be more comparable to the training data, and the loss of the ultimately desired prediction is shown
# - Test simpler, nested loop implementation of training data creation
# - Make results directory part of repo to avoid error in creating pdfs after initial clone
# - Todos
# - Test specific-general-implementation with different configs
# - Compare master to x_k, y_k => y_k+1
# Stretch
# - Incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
#   - Create a library just for parsing functions using the protocol in docs/protocol_idea.txt, and for creating data from them
#   - Use this library to parse an input function and generate the function
#   - When I combine all my major STEM projects together, that library should be a shared library
# - Customize placement of activation functions, as well as choice of optimization and/or loss function
# - Put x_k and y_(k-1) on separate networks