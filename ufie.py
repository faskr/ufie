import numpy as np
import time
import torch.optim as optim
#import torch.nn.functional as F
from generate_data import *
from function_model import *
from plots import *

class UFIE:
    def __init__(self, configs, data_s, data_g=None, sample_boundary=None):
        self.trained_extrap = isinstance(data_g, np.ndarray) # whether extrapolation is trained or not
        if not self.trained_extrap and not sample_boundary:
            raise ValueError("Either data_g or sample_boundary must be provided")
        self.mode = configs['mode']
        self.model_y_inputs = configs['model_y_inputs']
        self.depth = configs['depth']
        self.breadth = configs['breadth']
        self.lr = configs['lr']
        self.steps = configs['steps']
        if self.trained_extrap:
            self.test_size = configs['test_size']
            self.sample_boundary = data_s.shape[0]
            self.total_samples = data_g.shape[1]
        else:
            self.test_size = 1
            self.sample_boundary = sample_boundary
            self.total_samples = data_s.shape[0]
        self.interpolations = self.sample_boundary - self.model_y_inputs
        self.extrapolations = self.total_samples - self.sample_boundary
        # load data and make training set
        if self.trained_extrap:
            tiled_data_s = np.tile(data_s[:, 1], (data_g.shape[0], 1))
            tiled_data = np.concat((tiled_data_s, data_g[:, self.sample_boundary:, 1]), axis=1)
            self.x = torch.from_numpy(data_g[:, self.model_y_inputs:, 0])
            self.x_train = self.x[self.test_size:, :]
            self.x_test = self.x[:self.test_size, :]
        else:
            tiled_data = data_s[None, :, 1]
            self.x = torch.from_numpy(data_s[None, self.model_y_inputs:, 0])
            self.x_train = self.x[:, :self.sample_boundary-self.model_y_inputs]
            self.x_test = self.x[:, self.sample_boundary-self.model_y_inputs:]
        model_samples = self.total_samples - self.model_y_inputs
        y_prev = np.zeros((self.x.size(0), model_samples, self.model_y_inputs))
        # Probably faster, more complex way:
        for y_i in range(model_samples + self.model_y_inputs - 1): # exclude the sample in the range to slide the diagonal across
            # For each dataset, fill a diagonal of values with a single sample
            y_prev_values = tiled_data[:, y_i]
            sample_indices = range(max(y_i+1-self.model_y_inputs, 0), min(y_i+1, model_samples))
            input_indices = range(min(y_i, self.model_y_inputs-1), max(y_i-model_samples, -1), -1)
            y_prev[:, sample_indices, input_indices] = y_prev_values[:, None]
        # Probably slower, simpler way:
        #y_starts = range(self.interpolations + self.extrapolations)
        #y_inputs = range(self.model_y_inputs)
        #for y_start in y_starts:
        #    for y_input in y_inputs:
        #        y_s_i = tiled_data[:, y_start + y_input]
        #        y_prev_train[:, y_start, y_input] = y_s_i[self.test_size:]
        #        y_prev_test[:, y_start, y_input] = y_s_i[:self.test_size]
        if self.trained_extrap:
            self.y_prev_train = torch.from_numpy(y_prev[self.test_size:, :, :])
            self.y_prev_test = torch.from_numpy(y_prev[:self.test_size, :, :])
            self.y_target_train = torch.from_numpy(tiled_data[self.test_size:, self.model_y_inputs:])
            self.y_target_test = torch.from_numpy(tiled_data[:self.test_size, self.model_y_inputs:])
            self.y_target = self.y_target_test
        else:
            self.y_prev_train = torch.from_numpy(y_prev[:, :self.sample_boundary-self.model_y_inputs, :])
            self.y_prev_test = torch.from_numpy(y_prev[:, self.sample_boundary-self.model_y_inputs:, :])
            self.y_target_train = torch.from_numpy(tiled_data[:, self.model_y_inputs:self.sample_boundary])
            self.y_target_test = torch.from_numpy(tiled_data[:, self.sample_boundary:])
            self.y_target = torch.cat((self.y_target_train, self.y_target_test), dim=1)
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
        self.train_out = self.model(self.x_train, self.y_prev_train)
        loss = self.criterion(self.train_out, self.y_target_train)
        if self.iteration % self.step_size == 0:
            print('train loss:', loss.item())
            self.train_losses.append(loss.item())
        loss.backward()
        return loss

    def predict(self):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            pred = self.model(self.x_test, self.y_prev_test)
            loss = self.criterion(pred, self.y_target_test)
            #loss = self.criterion(pred[:, :-self.extrapolations], self.y_target_test[:, :self.interpolations])
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
            prediction = self.predict() # predict
            if self.trained_extrap:
                y = prediction
            else:
                y = np.concat((self.train_out.detach().numpy(), prediction), axis=1)
            # outputs
            if self.iteration % self.step_size == 0:
                exe_times.append(time.time() - start_time)
                if self.iteration % (4 * self.step_size) == 0:
                    live_plots.draw_plots(self.iteration, self.y_target, y, self.interpolations, self.extrapolations, self.descent_steps, exe_times, self.train_losses, self.test_losses)
        live_plots.save('results/%.4f_%dx%d_%.2flr_%dsteps.pdf' % (self.test_losses[-1], self.depth, self.breadth, self.lr, self.steps))
