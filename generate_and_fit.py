import sys
import json
from generate_data import *
from ufie import *

# Use case 1: interpolate and extrapolate function given its data points (simple regression - generalization to other data points)
# Use case 2: interpolate and extrapolate function given its data points and those from similar functions (hence generating many polynomials - generalization to other polynomials)
    # Interpolate from data of function
    # Extrapolate from data of similar functions (which can include the function itself)
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)

    with open(sys.argv[1], 'r') as input_file:
        configs = json.load(input_file)

    data_s, data_g = generate_polynomials(configs['data'])
    ufie = UFIE(configs['model'], data_s, data_g)
    # ufie = UFIE(configs['model'], data_g[0, :, :], num_specific_samples=100)
    ufie.converge()

# Priority Tasks
# - In theory, training interp and extrap together is bad for extrap, because the functions are discontinuous, which is bad for the fit
#   - Instead, try separating data into two meta-datasets, one for interp and one for extrap; interp is just the specific dataset in the 
#     interp zone, tiled repeatedly into a small matrix (to build up interp learning), and extrap is the general datasets in the extrap 
#     zone; this is not much different, except that interp and extrap won't be connected to each other
#   - *In theory*, this should have lower loss, at least for the extrapolation zone if not overall
# - Test simpler, nested loop implementation of training data creation
# - Make results directory part of repo to avoid error in creating pdfs after initial clone
# - Todos
# - Test specific-general-implementation with different configs
# - Compare master to x_k, y_k => y_k+1
# - Merge whatever I think is good to merge to master
# Stretch
# - Incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
#   - Create a library just for parsing functions using the protocol in docs/protocol_idea.txt, and for creating data from them
#   - Use this library to parse an input function and generate the function
#   - When I combine all my major STEM projects together, that library should be a shared library
# - Customize placement of activation functions, as well as choice of optimization and/or loss function
# - Put x_k and y_(k-1) on separate networks