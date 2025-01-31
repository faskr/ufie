import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torch.optim as optim

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