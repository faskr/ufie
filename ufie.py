import numpy as np
import time
import torch.optim as optim
#import torch.nn.functional as F
from generate_data import *
from function_model import *
from plots import *

class UFIE:
    def __init__(self, data_s, data_g, configs):
        self.mode = configs['mode']
        self.model_y_inputs = configs['model_y_inputs']
        self.depth = configs['depth']
        self.breadth = configs['breadth']
        self.lr = configs['lr']
        self.test_size = configs['test_size']
        self.steps = configs['steps']
        self.sample_boundary = data_s.shape[0]
        self.total_samples = data_g.shape[1]
        self.interpolations = self.sample_boundary - self.model_y_inputs
        self.extrapolations = self.total_samples - self.sample_boundary
        # load data and make training set
        tiled_data_s = np.tile(data_s[:, 1], data_g.shape[0])
        tiled_data = np.concat((tiled_data_s, data_g[:, :, 1]), axis=1)
        self.x_train = torch.from_numpy(data_g[self.test_size:, self.model_y_inputs:, 0])
        self.x_test = torch.from_numpy(data_g[:self.test_size, self.model_y_inputs:, 0])
        y_prev_train = np.zeros((self.x_train.size(0), self.total_samples-1, self.model_y_inputs)) # y_prev doesn't include the last sample, so use total_samples - 1
        y_prev_test = np.zeros((self.x_test.size(0), self.total_samples-1, self.model_y_inputs))
        # Probably faster, more difficult way:
        for y_i in range(self.total_samples-1):
            # Fill a diagonal of values from the same sample
            y_prev = tiled_data[:, y_i]
            sample_indices = range(max(y_i+1-self.model_y_inputs, 0), y_i+1)
            input_indices = range(y_i, max(y_i-self.model_y_inputs, -1), -1)
            y_prev_train[:, sample_indices, input_indices] = y_prev[self.test_size:]
            y_prev_test[:, sample_indices, input_indices] = y_prev[:self.test_size]
        # Probably slower, easier way:
        #y_starts = range(self.interpolations + self.extrapolations)
        #y_inputs = range(self.model_y_inputs)
        #for y_start in y_starts:
        #    for y_input in y_inputs:
        #        y_s_i = tiled_data[:, y_start + y_input]
        #        y_prev_train[:, y_start, y_input] = y_s_i[self.test_size:]
        #        y_prev_test[:, y_start, y_input] = y_s_i[:self.test_size]
        self.y_prev_train = torch.from_numpy(y_prev_train)
        self.y_prev_test = torch.from_numpy(y_prev_test)

        self.y_target_train = torch.from_numpy(tiled_data[self.test_size:, self.model_y_inputs:])
        self.y_target_test = torch.from_numpy(tiled_data[:self.test_size, self.model_y_inputs:])
        step = data_g[0, -1, 0] - data_g[0, -2, 0]
        start = data_g[0, -1, 0] + step
        stop = start + self.extrapolations * step
        shift = data_g[:self.test_size, -1, 0].reshape(self.test_size, 1) + step - start
        self.x_extrap = torch.from_numpy(np.arange(start, stop, step) + shift)
        # build the model
        self.model = FunctionModel(self.depth, self.breadth, mode=self.mode, y_length=self.model_y_inputs)
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
        out = self.model(self.x_train, self.y_prev_train)
        loss = self.criterion(out, self.y_target_train)
        if self.iteration % self.step_size == 0:
            print('train loss:', loss.item())
            self.train_losses.append(loss.item())
        loss.backward()
        return loss

    def predict(self):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = self.model(self.x_test, self.y_prev_test, self.x_extrap)
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
            # TODO: can't this be in the if statement below?
            y = self.predict() # predict
            # outputs
            if self.iteration % self.step_size == 0:
                exe_times.append(time.time() - start_time)
                if self.iteration % (4 * self.step_size) == 0:
                    live_plots.draw_plots(self.iteration, self.y_target_test, y, self.interpolations, self.extrapolations, self.descent_steps, exe_times, self.train_losses, self.test_losses)
        live_plots.save('results/%.4f_%dx%d_%.2flr_%dsteps.pdf' % (self.test_losses[-1], self.depth, self.breadth, self.lr, self.steps))
