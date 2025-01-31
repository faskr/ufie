import argparse
import torch
from generate_functions import *
from ufie import *

np.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=int, default=100, help='number of datasets')
    parser.add_argument('--samples', type=int, default=100, help='number of samples per dataset')
    parser.add_argument('--start', type=float, default=0, help='first sample')
    parser.add_argument('--separation', type=float, default=1, help='separation between one sample and the next')
    parser.add_argument('--shift', type=int, default=0, help='limit of random shift')
    parser.add_argument('--base_scale', type=float, default=1, help='component of scale that applies to all equations')
    parser.add_argument('--equation_scale', type=float, default=0, help='limit of random component of scale that applies to the full equation')
    parser.add_argument('--term_scale', type=float, default=0, help='limit of random component of scale that applies to one term in the equation')
    parser.add_argument('--mode', type=str, default='xy', help='whether to use x, y, or both x and y for input to the network')
    parser.add_argument('--depth', type=int, default=1, help='nn depth (hidden layers)')
    parser.add_argument('--breadth', type=int, default=40, help='nn breadth')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--test_size', type=int, default=5, help='how many of the datasets are test sets')
    parser.add_argument('--steps', type=int, default=1000, help='steps to run')
    parser.add_argument('--prediction_size', type=int, default=100, help='number of future inputs to predict the outputs for')
    opt = parser.parse_args()
    coefficients = [0, 0, 1]
    #opt.shift = 10
    #opt.equation_scale = 10
    #opt.depth = 3
    print('Generating data...')
    data = generate_polynomial(opt, coefficients)
    torch.save(data.astype('float64'), open('traindata.pt', 'wb'))
    ufie = UFIE(opt)
    ufie.converge()

# Tasks
# - use inputs.json instead of command line parameters
# - test different polynomials with different configurations
# - see if network using only x_k or only y_(k-1) can be improved
# - implement desired plots
# - incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)

# Universal Function Interpolator and Extrapolator
# Receives:
# - list of polynomial coefficients along with training data parameters (e.g. how many samples, interval of each sample)
# - collection of training parameters
# - desired prediction length
# Does the following:
# - generates polynomials for training and testing
# - trains on a subset of the generated data while testing on the other subset (uses both x_k and y_(k-1) as inputs to predict y_k)
# - approximates and predicts the function that has the given coefficients
# Outputs:
# - a plot (as a subplot) of
#   - the generated function in the approximation zone and prediction zone
#   - the approximation of the function (generated using the testing values, which were generated using the given coefficients)
#   - the prediction of the function (generated using the approximation)
# - plots of the training and testing convergence (as subplots in the same figure as the plot above)
#   - loss over iterations
#   - loss over time