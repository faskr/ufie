import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class LivePlot:
    def __init__(self):
        self.figure_created = False

    def draw_prediction(self, y, known_points, predictions, num_functions, title=None, path=None):
        matplotlib.use('TkAgg')
        if not self.figure_created:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(16,8))
            plt.xlabel('x')
            plt.ylabel('y')
            self.plots = []
            for i in range(num_functions):
                self.plots += [self.ax.plot(np.arange(known_points), y[i][:known_points]) +
                               self.ax.plot(np.arange(known_points, known_points + predictions), y[i][known_points:], ':')]
                self.plots[-1][1].set_color(self.plots[-1][0].get_color())
            self.figure_created = True
        else:
            for i in range(num_functions):
                self.plots[i][0].set_xdata(np.arange(known_points))
                self.plots[i][0].set_ydata(y[i][:known_points])
                self.plots[i][1].set_xdata(np.arange(known_points, known_points + predictions))
                self.plots[i][1].set_ydata(y[i][known_points:])
        if title:
            plt.title(title)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if path:
            plt.savefig(path)

def draw_convergence(train_losses, test_losses, recorded_steps, title, path):
    matplotlib.use('Agg')
    plt.figure()
    plt.title(title)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.plot(recorded_steps, train_losses, 'b')
    plt.plot(recorded_steps, test_losses, 'r')
    plt.legend(['Training', 'Testing'])
    plt.savefig(path)
