import numpy as np
import time
from generate_functions import *
from function_model import *
from plots import *

class UFIE:
    def __init__(self, opt):
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
        # load data and make training set
        data = torch.load('traindata.pt')
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
