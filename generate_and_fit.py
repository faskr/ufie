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

    data = generate_polynomials_old(configs['data'])
    ufie = UFIE(data, configs['model'])
    ufie.converge()

# Priority Tasks
# - Have interface that receives a single dataset to fit, optionally along with other datasets in the zone of extrapolation, used to offer another means of generalizing that zone
# 	- Generalize the other datasets by sliding a range of prior y values as input through the extrapolation zone, and using that along with current x (or equivalent range of x's?) to predict subsequent y values
# 	- This way, the model is trained in the interpolation zone using the fitted dataset, and in the extrapolation zone using the other datasets
# - Consider using/allowing x_k-1, y_k-1 => y_k instead, or x_k, y_k => y_k+1
# - Incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
# Stretch
# - Customize placement of activation functions, as well as choice of optimization and/or loss function
# - Put x_k and y_(k-1) on separate networks