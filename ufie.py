from __future__ import print_function
import argparse
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim
from os import path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_polynomial(opt, coefficients):
    # set random seed to 0
    np.random.seed(0)
    # Generate input data
    stop = opt.start + opt.length * opt.separation
    shift = np.random.randint(-opt.shift, opt.shift + 1, opt.number).reshape(opt.number, 1)
    x = np.array(range(opt.start, stop, opt.separation)) + shift
    # Generate output data
    equation_scale = opt.equation_scale * np.random.rand(opt.number).reshape(opt.number, 1)
    data = np.dstack([x, np.zeros(x.shape)])
    for k, c in enumerate(coefficients):
        term_scale = opt.term_scale * np.random.rand(opt.number).reshape(opt.number, 1)
        data[:, :, 1] += (opt.total_scale + equation_scale + term_scale) * c * np.pow(x, k)
    #print(data[:10])
    return data

class Net(nn.Module):
    def __init__(self, depthH=1, breadth=40):
        super(Net, self).__init__()
        self.depthH = depthH
        self.linearI = nn.Linear(2, breadth)
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
        input = torch.cat([x_interp[:, :, None], y_prev_interp[:, :, None]], dim=2)
        output_tensor = self.network(input)[:, :, 0]
        output_list = list(output_tensor.split(1, dim=1))
        for i in range(x_extrap.size(1)): # if we should predict the future
            input = torch.cat([x_extrap[:, i, None], output_list[-1]], dim=1)
            output_list += [self.network(input)]
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor

def draw_prediction(i, y, input_size, future):
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(input_size), yi[:input_size], color, linewidth = 2.0)
        plt.plot(np.arange(input_size, input_size + future), yi[input_size:], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.savefig('predict%d.pdf'%i)
    plt.close()

def draw_convergence(train_losses, test_losses, recorded_steps, depthH, breadth, lr, steps):
    plt.figure()
    plt.title('Convergence (train=%d, test=%d)' % (train_losses[-1], test_losses[-1]))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot(recorded_steps, train_losses, 'b')
    plt.plot(recorded_steps, test_losses, 'r')
    plt.yscale('log')
    plt.legend(['Training', 'Testing'])
    plt.savefig('convergence_%d_%dx%d_%.2flr_%ds.pdf' % (test_losses[-1], depthH, breadth, lr, steps))

def converge(data, depthH=1, breadth=40, lr=0.01, steps=1000, future=100):
    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    x_interp_train = torch.from_numpy(data[3:, 1:, 0])
    x_interp_test = torch.from_numpy(data[:3, 1:, 0])
    y_prev_interp_train = torch.from_numpy(data[3:, :-1, 1])
    y_prev_interp_test = torch.from_numpy(data[:3, :-1, 1])
    target_train = torch.from_numpy(data[3:, 1:, 1])
    target_test = torch.from_numpy(data[:3, 1:, 1])
    step = data[0, -1, 0] - data[0, -2, 0]
    start = data[0, -1, 0] + step
    stop = start + future * step
    shift = data[:3, -1, 0].reshape(3, 1) + step - start
    x_extrap = torch.from_numpy(np.arange(start, stop, step) + shift)
    # build the model
    model = Net(depthH, breadth)
    model.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    #optimizer = optim.LBFGS(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recorded_steps = []
    train_losses = []
    test_losses = []
    step_size = 25
    def calculate_error():
        optimizer.zero_grad()
        out = model(x_interp_train, y_prev_interp_train)
        loss = criterion(out, target_train)
        if i % step_size == 0:
            print('loss:', loss.item())
            recorded_steps.append(i)
            train_losses.append(loss.item())
        loss.backward()
        return loss
    def predict():
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = model(x_interp_test, y_prev_interp_test, x_extrap)
            loss = criterion(pred[:, :-future], target_test)
            y = pred.detach().numpy()
            return y, loss

    #begin to train
    for i in range(1, steps + 1):
        if i % step_size == 0:
            print('STEP: ', i)
        optimizer.step(calculate_error)
        y, loss = predict()
        # outputs
        if i % step_size == 0:
            print('test loss:', loss.item())
            test_losses.append(loss.item())
        if i % (4 * step_size) == 0:
            draw_prediction(i, y, x_interp_train.size(1), future)
    draw_convergence(train_losses, test_losses, recorded_steps, depthH, breadth, lr, steps)


depthH = 1
breadth = 40
lr = 0.01
steps = 1000
future = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regenerate_data', type=bool, default=False, help='regenerate training data if already present')
    parser.add_argument('--number', type=int, default=100, help='number of datasets')
    parser.add_argument('--length', type=int, default=100, help='number of samples per dataset')
    parser.add_argument('--start', type=float, default=0, help='first sample')
    parser.add_argument('--separation', type=float, default=1, help='separation between one sample and the next')
    parser.add_argument('--shift', type=int, default=0, help='limit of random shift')
    parser.add_argument('--total_scale', type=float, default=1, help='component of scale that applies to all equations')
    parser.add_argument('--equation_scale', type=float, default=0, help='limit of random component of scale that applies to the full equation')
    parser.add_argument('--term_scale', type=float, default=0, help='limit of random component of scale that applies to one term in the equation')
    parser.add_argument('--depth', type=int, default=depthH, help='nn depth (hidden layers)')
    parser.add_argument('--breadth', type=int, default=breadth, help='nn breadth')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--steps', type=int, default=steps, help='steps to run')
    parser.add_argument('--future', type=int, default=future, help='number of future inputs to predict the outputs for')
    opt = parser.parse_args()
    if opt.regenerate_data or not path.isfile('traindata.pt'):
        print('Generating data...')
        #opt.shift = 4
        coefficients = [0, 0, 1]
        data = generate_polynomial(opt, coefficients)
        torch.save(data.astype('float64'), open('traindata.pt', 'wb'))
    else:
        data = torch.load('traindata.pt')
    converge(data, opt.depth, opt.breadth, opt.lr, opt.steps, opt.future)

# Tasks
# - organize code better (move plotting functions into a different module, & use modules/classes to organize whatever else)
# - have more convenient plots (e.g. real-time updating, more precision in some numbers, outputs put into a sub-folder)
# - implement desired plots
# - have option to change number or proportion of train vs. test data
# - use inputs.json instead of command line parameters
# - incorporate non-polynomial functions (trigonometric, exponential, logarithmic, etc.)
# - might possibly need to have an option to disable either x_k or y_(k-1) parameter for certain functions

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