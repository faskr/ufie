import numpy as np
import time
import torch.optim as optim
#import torch.nn.functional as F
from generate_data import *
from function_model import *
from plots import *

class UFIE:
    def __init__(self, configs, data_s, data_g=None, num_specific_samples=None):
        self.trained_extrap = isinstance(data_g, np.ndarray) # whether extrapolation is trained or not
        if not self.trained_extrap and not num_specific_samples:
            raise ValueError("Either data_g or num_specific_samples must be provided")
        self.mode = configs['mode']
        self.model_y_inputs = configs['model_y_inputs']
        self.depth = configs['depth']
        self.breadth = configs['breadth']
        self.lr = configs['lr']
        self.steps = configs['steps']
        if self.trained_extrap:
            self.test_size = configs['test_size']
            self.num_specific_samples = data_s.shape[0]
            self.total_samples = data_g.shape[1]
        else:
            self.test_size = 1
            self.num_specific_samples = num_specific_samples
            self.total_samples = data_s.shape[0]
        self.interpolations = self.num_specific_samples - self.model_y_inputs
        self.extrapolations = self.total_samples - self.num_specific_samples
        # load data and make training set
        if self.trained_extrap:
            tiled_data_s = np.tile(data_s[:, 1], (data_g.shape[0], 1))
            tiled_data_g = data_g[:, self.interpolations:, 1]
            self.x = torch.from_numpy(data_g[:, self.model_y_inputs:, 0])
            self.x_train = self.x[self.test_size:, :]
            self.x_test = self.x[:self.test_size, :]
            y_prev_g = np.zeros((self.x.size(0), self.extrapolations, self.model_y_inputs))
            for y_i in range(self.extrapolations + self.model_y_inputs - 1): # exclude the sample in the range to slide the diagonal across
                # For each dataset, fill a diagonal of values with a single sample
                y_prev_values = tiled_data_g[:, y_i]
                sample_indices = range(max(y_i+1-self.model_y_inputs, 0), min(y_i+1, self.extrapolations))
                input_indices = range(min(y_i, self.model_y_inputs-1), max(y_i-self.extrapolations, -1), -1)
                y_prev_g[:, sample_indices, input_indices] = y_prev_values[:, None]
        else:
            tiled_data_s = data_s[None, :, 1]
            self.x = torch.from_numpy(data_s[None, self.model_y_inputs:, 0])
            self.x_train = self.x[:, :self.interpolations]
            self.x_test = self.x[:, self.interpolations:]
        model_samples = self.total_samples - self.model_y_inputs
        # Probably faster, more complex way:
        y_prev_s = np.zeros((self.x.size(0), self.interpolations, self.model_y_inputs))
        for y_i in range(self.num_specific_samples - 1): # exclude the sample in the range to slide the diagonal across
            # For each dataset, fill a diagonal of values with a single sample
            y_prev_values = tiled_data_s[:, y_i]
            sample_indices = range(max(y_i+1-self.model_y_inputs, 0), min(y_i+1, self.interpolations))
            input_indices = range(min(y_i, self.model_y_inputs-1), max(y_i-self.interpolations, -1), -1)
            y_prev_s[:, sample_indices, input_indices] = y_prev_values[:, None]
        # Probably slower, simpler way:
        #y_starts = range(self.interpolations + self.extrapolations)
        #y_inputs = range(self.model_y_inputs)
        #for y_start in y_starts:
        #    for y_input in y_inputs:
        #        y_s_i = tiled_data[:, y_start + y_input]
        #        y_prev_train[:, y_start, y_input] = y_s_i[self.test_size:]
        #        y_prev_test[:, y_start, y_input] = y_s_i[:self.test_size]
        if self.trained_extrap:
            self.y_prev_train_s = torch.from_numpy(y_prev_s[self.test_size:, :, :])
            self.y_prev_test_s = torch.from_numpy(y_prev_s[:self.test_size, :, :])
            self.y_target_train_s = torch.from_numpy(tiled_data_s[self.test_size:, self.model_y_inputs:])
            self.y_target_test_s = torch.from_numpy(tiled_data_s[:self.test_size, self.model_y_inputs:])
            self.y_target_s = self.y_target_test_s
            self.y_prev_train_g = torch.from_numpy(y_prev_g[self.test_size:, :, :])
            self.y_prev_test_g = torch.from_numpy(y_prev_g[:self.test_size, :, :])
            self.y_target_train_g = torch.from_numpy(tiled_data_g[self.test_size:, self.model_y_inputs:])
            self.y_target_test_g = torch.from_numpy(tiled_data_g[:self.test_size, self.model_y_inputs:])
            self.y_target_g = self.y_target_test_g
        else:
            self.y_prev_train_s = torch.from_numpy(y_prev_s[:, :self.interpolations, :])
            self.y_prev_test_s = torch.from_numpy(y_prev_s[:, self.interpolations:, :])
            self.y_target_train_s = torch.from_numpy(tiled_data_s[:, self.model_y_inputs:self.num_specific_samples])
            self.y_target_test_s = torch.from_numpy(tiled_data_s[:, self.num_specific_samples:])
            self.y_target_s = torch.cat((self.y_target_train_s, self.y_target_test_s), dim=1)
        # build the model
        self.model = FunctionModel(self.depth, self.breadth, mode=self.mode, y_length=self.model_y_inputs)
        self.model.double()
        self.criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        #optimizer = optim.LBFGS(self.model.parameters(), lr=self.lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.descent_steps = []
        self.train_losses = []
        self.specific_losses = []
        self.test_losses = []
        self.step_size = 25

    def calculate_error(self):
        self.optimizer.zero_grad()
        self.train_out_s = self.model(self.x_train, self.y_prev_train_s)
        loss = self.criterion(self.train_out_s, self.y_target_train_s)
        if self.trained_extrap:
            self.train_out_g = self.model(self.x_train, self.y_prev_train_g)
            loss += self.criterion(self.train_out_g, self.y_target_train_g)
        if self.iteration % self.step_size == 0:
            print('train loss:', loss.item())
            self.train_losses.append(loss.item())
        loss.backward()
        return loss

    def predict(self):
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            if self.trained_extrap:
                pred_i = self.model(self.x_test[:, :self.interpolations], self.y_prev_test_s[:, :self.interpolations, :])
                pred_e = self.model(self.x_test[:, self.interpolations:], self.y_prev_test_g[:, self.interpolations:, :])
                loss_si = self.criterion(pred_i, self.y_target_test_s)
                loss_se = self.criterion(pred_e[0, :], self.y_target_test_g[0, :])
                loss_ge = self.criterion(pred_e[1:, :], self.y_target_test_g[1:, :])
            else:
                pred_e = self.model(self.x_test[:, self.interpolations:], self.y_prev_test_s)
                loss_se = self.criterion(pred_e, self.y_target_test_s)

            if self.iteration % self.step_size == 0:
                if self.trained_extrap:
                    print('specific loss:', loss_s.item())
                    self.specific_losses.append(loss_s.item())
                else:
                    self.specific_losses.append(loss.item())
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
                    live_plots.draw_plots(self.iteration, self.y_target, y, self.interpolations, self.extrapolations, self.descent_steps, 
                                          exe_times, self.train_losses, self.test_losses, self.specific_losses)
        live_plots.save('results/%.4f_%dx%d_%.2flr_%dsteps.pdf' % (self.test_losses[-1], self.depth, self.breadth, self.lr, self.steps))
