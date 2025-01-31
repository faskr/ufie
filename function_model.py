import torch
import torch.nn as nn

class FunctionModel(nn.Module):
    def __init__(self, depthH=1, breadth=40, mode='xy'):
        super(FunctionModel, self).__init__()
        match mode:
            case 'x':
                self.make_input = self.make_input_x
            case 'y':
                self.make_input = self.make_input_y
            case 'xy':
                self.make_input = self.make_input_xy
            case _:
                ValueError("Mode needs to be 'x', 'y', or 'xy'")
        self.depthH = depthH
        self.linearI = nn.Linear(2, breadth) if mode == 'xy' else nn.Linear(1, breadth)
        self.linearH = nn.Linear(breadth, breadth)
        self.linearO = nn.Linear(breadth, 1)
        self.relu = nn.ReLU()

    def make_input_x(self, x, y_prev):
        return x

    def make_input_y(self, x, y_prev):
        return y_prev

    def make_input_xy(self, x, y_prev):
        return torch.cat([x, y_prev], dim=x.dim() - 1)

    def network(self, x, y_prev):
        input = self.make_input(x, y_prev)
        h = self.linearI(input)
        for i in range(self.depthH - 1):
            h = self.relu(self.linearH(h))
        output = self.linearO(h)
        return output

    def forward(self, x_interp, y_prev_interp, x_extrap=torch.empty((1, 0))):
        assert x_interp.size(1) == y_prev_interp.size(1)
        # Interpolate
        output_tensor = self.network(x_interp[:, :, None], y_prev_interp[:, :, None])[:, :, 0]
        output_list = list(output_tensor.split(1, dim=1))
        # Extrapolate if there are extrapolation inputs
        for i in range(x_extrap.size(1)):
            output_list += [self.network(x_extrap[:, i, None], output_list[-1])]
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor