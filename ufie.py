from __future__ import print_function
import argparse
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
import os
import numpy as np
from plots import *
import time

np.random.seed(0)
torch.manual_seed(0)

class FunctionModel(nn.Module):
    def __init__(self, depthH=1, breadth=40, mode='xy'):
        super(FunctionModel, self).__init__()
        if mode not in ['x', 'y', 'xy']:
            ValueError("Mode needs to be 'x', 'y', or 'xy'")
        self.depthH = depthH
        self.mode = mode
        self.linearI = nn.Linear(2, breadth) if mode == 'xy' else nn.Linear(1, breadth)
        self.linearH = nn.Linear(breadth, breadth)
        self.linearO = nn.Linear(breadth, 1)
        self.relu = nn.ReLU()

    def network(self, input):
        h = self.relu(self.linearI(input))
        for i in range(self.depthH - 1):
            h = self.relu(self.linearH(h))
        output = self.relu(self.linearO(h))
        return output

    def forward(self, x_interp, y_prev_interp, x_extrap=torch.empty((1, 0))):
        assert x_interp.size(1) == y_prev_interp.size(1)
        if self.mode == 'x':
            input = x_interp[:, :, None]
        elif self.mode == 'y':
            input = y_prev_interp[:, :, None]
        elif self.mode == 'xy':
            input = torch.cat([x_interp[:, :, None], y_prev_interp[:, :, None]], dim=2)
        output_tensor = self.network(input)[:, :, 0]
        output_list = list(output_tensor.split(1, dim=1))
        for i in range(x_extrap.size(1)): # if we should predict the future
            if self.mode == 'x':
                input = x_extrap[:, i, None]
            elif self.mode == 'y':
                input = output_list[-1]
            elif self.mode == 'xy':
                input = torch.cat([x_extrap[:, i, None], output_list[-1]], dim=1)
            output_list += [self.network(input)]
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor

class UFIE:
    def __init__(self, coefficients, opt):
        self.datasets = opt.datasets
        self.samples = opt.samples
        self.start = opt.start
        self.separation = opt.separation
        self.shift = opt.shift
        self.base_scale = opt.base_scale
        self.equation_scale = opt.equation_scale
        self.term_scale = opt.term_scale
        self.mode = opt.mode
        self.depth = opt.depth
        self.breadth = opt.breadth
        self.lr = opt.lr
        self.test_size = opt.test_size
        self.steps = opt.steps
        self.prediction_size = opt.prediction_size
        # Get the data
        if opt.regenerate_data or not os.path.isfile('traindata.pt'):
            print('Generating data...')
            data = self.generate_polynomial(coefficients)
            torch.save(data.astype('float64'), open('traindata.pt', 'wb'))
        else:
            data = torch.load('traindata.pt')
        # load data and make training set
        self.x_interp_train = torch.from_numpy(data[self.test_size:, 1:, 0])
        self.x_interp_test = torch.from_numpy(data[:self.test_size, 1:, 0])
        self.y_prev_interp_train = torch.from_numpy(data[self.test_size:, :-1, 1])
        self.y_prev_interp_test = torch.from_numpy(data[:self.test_size, :-1, 1])
        self.target_train = torch.from_numpy(data[self.test_size:, 1:, 1])
        self.target_test = torch.from_numpy(data[:self.test_size, 1:, 1])
        step = data[0, -1, 0] - data[0, -2, 0]
        start = data[0, -1, 0] + step
        stop = start + opt.prediction_size * step
        shift = data[:self.test_size, -1, 0].reshape(self.test_size, 1) + step - start
        self.x_extrap = torch.from_numpy(np.arange(start, stop, step) + shift)
        # build the model
        self.model = FunctionModel(opt.depth, opt.breadth, mode=self.mode)
        self.model.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        #optimizer = optim.LBFGS(model.parameters(), lr=opt.lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)
        self.recorded_steps = []
        self.train_losses = []
        self.test_losses = []
        self.step_size = 25
    
    def generate_polynomial(self, coefficients):
        # Generate input data
        stop = self.start + self.samples * self.separation
        shift = np.random.randint(-self.shift, self.shift + 1, self.datasets).reshape(self.datasets, 1)
        x = np.array(range(self.start, stop, self.separation)) + shift
        # Generate output data
        equation_scale = self.equation_scale * np.random.rand(self.datasets).reshape(self.datasets, 1)
        data = np.dstack([x, np.zeros(x.shape)])
        for k, c in enumerate(coefficients):
            term_scale = self.term_scale * np.random.rand(self.datasets).reshape(self.datasets, 1)
            data[:, :, 1] += (self.base_scale + equation_scale + term_scale) * c * np.pow(x, k)
        #print(data[:10])
        return data

    def calculate_error(self):
        self.optimizer.zero_grad()
        out = self.model(self.x_interp_train, self.y_prev_interp_train)
        loss = self.criterion(out, self.target_train)
        if self.iteration % self.step_size == 0:
            print('loss:', loss.item())
            self.recorded_steps.append(self.iteration)
            self.train_losses.append(loss.item())
        loss.backward()
        return loss

    def predict(self):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = self.model(self.x_interp_test, self.y_prev_interp_test, self.x_extrap)
            loss = self.criterion(pred[:, :-self.prediction_size], self.target_test)
            y = pred.detach().numpy()
            return y, loss
    
    def converge(self):
        live_plots = LivePlots()
        start_time = time.time()
        exe_times = []
        #begin to train
        for self.iteration in range(1, self.steps + 1):
            if self.iteration % self.step_size == 0:
                print('STEP: ', self.iteration)
            self.optimizer.step(self.calculate_error)
            y, loss = self.predict()
            # outputs
            if self.iteration % self.step_size == 0:
                print('test loss:', loss.item())
                self.test_losses.append(loss.item())
                exe_times.append(time.time() - start_time)
            if self.iteration % (4 * self.step_size) == 0:
                live_plots.draw_plots(y, self.iteration, self.x_interp_train.size(1), self.prediction_size, self.test_size, self.recorded_steps, exe_times, self.train_losses, self.test_losses)
        live_plots.save('results/%.4f_%dx%d_%.2flr_%ds.pdf' % (self.test_losses[-1], self.depth, self.breadth, self.lr, self.steps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate_data', type=bool, default=False, help='regenerate training data if already present')
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
    ufie = UFIE(coefficients, opt)
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