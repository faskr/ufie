import torch
import torch.nn as nn

class FunctionModel(nn.Module):
    def __init__(self, depthH=1, breadth=40, mode='xy', x_length=1, y_length=1):
        super(FunctionModel, self).__init__()
        match mode:
            case 'x':
                self.make_input = self.make_input_x
                inputs = x_length
            case 'y':
                self.make_input = self.make_input_y
                inputs = y_length
            case 'xy':
                self.make_input = self.make_input_xy
                inputs = x_length + y_length
            case _:
                ValueError("Mode needs to be 'x', 'y', or 'xy'")
        self.y_length = y_length
        self.depthH = depthH
        self.linearI = nn.Linear(inputs, breadth)
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

    def forward(self, x_interp, y_prev_interp):
        assert x_interp.size(1) == y_prev_interp.size(1)
        # Interpolate
        if y_prev_interp.dim() == 2:
            y_prev_interp = y_prev_interp[:, :, None]
        output_tensor = self.network(x_interp[:, :, None], y_prev_interp)[:, :, 0]
        #output_list = list(output_tensor.split(1, dim=1))
        # Extrapolate if there are extrapolation inputs
        # TODO: use general data for extrapolation; edit: actually, I think this loop will just be deleted and this function will simply be the above
        #for i in range(x_extrap.size(1)):
        #    output_tensor = torch.cat([output_tensor, self.network(x_extrap[:, i, None], output_tensor[:, -self.y_length:])], dim=1)
        #output_tensor = torch.cat(output_list, dim=1)
        return output_tensor