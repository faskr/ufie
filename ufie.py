import numpy as np
import time
import torch.optim as optim
#import torch.nn.functional as F
from generate_data import *
from function_model import *
from plots import *

class UFIE:
    def __init__(self, data, configs):
        self.mode = configs['mode']
        self.depth = configs['depth']
        self.breadth = configs['breadth']
        self.lr = configs['lr']
        self.test_size = configs['test_size']
        self.steps = configs['steps']
        self.extrapolations = configs['extrapolations']
        self.interpolations = data.shape[1] - self.extrapolations - 1
        # load data and make training set
        self.x_interp_train = torch.from_numpy(data[self.test_size:, :self.interpolations, 0])
        self.x_interp_test = torch.from_numpy(data[:self.test_size, :self.interpolations, 0])
        self.y_interp_train = torch.from_numpy(data[self.test_size:, :self.interpolations, 1])
        self.y_interp_test = torch.from_numpy(data[:self.test_size, :self.interpolations, 1])
        self.y_target_train = torch.from_numpy(data[self.test_size:, 1:self.interpolations+1, 1])
        self.y_target_test = torch.from_numpy(data[:self.test_size, 1:self.interpolations+1, 1])
        self.y_total_test = torch.from_numpy(data[:self.test_size, 1:, 1])
        step = data[0, -1, 0] - data[0, -2, 0]
        start = data[0, -1, 0] + step
        stop = start + self.extrapolations * step
        shift = data[:self.test_size, -1, 0].reshape(self.test_size, 1) + step - start
        self.x_extrap = torch.from_numpy(np.arange(start, stop, step) + shift)
        # build the model
        self.model = FunctionModel(self.depth, self.breadth, mode=self.mode)
        self.model.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        #optimizer = optim.LBFGS(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.descent_steps = []
        self.train_losses = []
        self.test_losses = []
        self.step_size = 25

    def calculate_error(self):
        self.optimizer.zero_grad()
        out = self.model(self.x_interp_train, self.y_interp_train)
        loss = self.criterion(out, self.y_target_train)
        if self.iteration % self.step_size == 0:
            print('train loss:', loss.item())
            self.train_losses.append(loss.item())
        loss.backward()
        return loss

    def predict(self):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = self.model(self.x_interp_test, self.y_interp_test, self.x_extrap)
            loss = self.criterion(pred[:, :-self.extrapolations], self.y_target_test)
            if self.iteration % self.step_size == 0:
                print('test loss:', loss.item())
                self.test_losses.append(loss.item())
            y = pred.detach().numpy()
            return y
    
    def converge(self):
        live_plots = LivePlots(self.test_size)
        exe_times = []
        start_time = time.time()
        for self.iteration in range(1, self.steps + 1):
            if self.iteration % self.step_size == 0:
                print('STEP:', self.iteration)
                self.descent_steps.append(self.iteration)
            self.optimizer.step(self.calculate_error) # train
            y = self.predict() # predict
            # outputs
            if self.iteration % self.step_size == 0:
                exe_times.append(time.time() - start_time)
                if self.iteration % (4 * self.step_size) == 0:
                    live_plots.draw_plots(self.iteration, self.y_total_test, y, self.interpolations, self.extrapolations, self.descent_steps, exe_times, self.train_losses, self.test_losses)
        live_plots.save('results/%.4f_%dx%d_%.2flr_%dsteps.pdf' % (self.test_losses[-1], self.depth, self.breadth, self.lr, self.steps))
