import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class LivePlots:
    def __init__(self, predictions):
        self.predictions = predictions
        self.figure_created = False

    def create_figure(self, coefficients, y, known_points, predicted_points, descent_steps, exe_times, train_losses, test_losses):
        # figure
        plt.ion()
        gs = gridspec.GridSpec(2, 2)
        self.fig = plt.figure(figsize=(12, 8))
        # truth vs. fit plot
        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        x = np.arange(1, known_points + predicted_points + 1)
        y_true = np.zeros(x.shape)
        for k, c in enumerate(coefficients):
            y_true += c * np.pow(x, k)
        self.ax1.plot(x, y_true)
        self.generalized_plots = [self.ax1.plot(x[:known_points], y[0][:known_points]) +
                                  self.ax1.plot(x[known_points:], y[0][known_points:], ':')]
        self.generalized_plots[0][1].set_color(self.generalized_plots[0][0].get_color())
        self.ax1.legend(['Truth', 'Fit', 'Prediction'])
        # generalized fit and prediction plot
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        for i in range(1, self.predictions):
            self.generalized_plots += [self.ax2.plot(np.arange(1, known_points + 1), y[i][:known_points]) +
                                       self.ax2.plot(np.arange(known_points + 1, known_points + predicted_points + 1), y[i][known_points:], ':')]
            self.generalized_plots[-1][1].set_color(self.generalized_plots[-1][0].get_color())
        # loss plot (steps)
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax3.set_xlabel('step')
        self.ax3.set_ylabel('loss')
        self.ax3.set_yscale('log')
        self.train_plot_steps, = self.ax3.plot(descent_steps, train_losses, 'b')
        self.test_plot_steps, = self.ax3.plot(descent_steps, test_losses, 'r')
        self.ax3.legend(['Training', 'Testing'])
        # loss plot (time)
        self.ax4 = self.fig.add_subplot(gs[1, 1])
        self.ax4.set_xlabel('seconds')
        self.ax4.set_ylabel('loss')
        self.ax4.set_yscale('log')
        self.train_plot_time, = self.ax4.plot(exe_times, train_losses, 'b')
        self.test_plot_time, = self.ax4.plot(exe_times, test_losses, 'r')
        self.ax4.legend(['Training', 'Testing'])
        # figure is created
        self.figure_created = True

    def update_figure(self, y, known_points, predicted_points, descent_steps, exe_times, train_losses, test_losses):
        # fit and prediction plots
        for i in range(self.predictions):
            self.generalized_plots[i][0].set_xdata(np.arange(1, known_points + 1))
            self.generalized_plots[i][0].set_ydata(y[i][:known_points])
            self.generalized_plots[i][1].set_xdata(np.arange(known_points + 1, known_points + predicted_points + 1))
            self.generalized_plots[i][1].set_ydata(y[i][known_points:])
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        # loss plot (steps)
        self.train_plot_steps.set_xdata(descent_steps)
        self.train_plot_steps.set_ydata(train_losses)
        self.test_plot_steps.set_xdata(descent_steps)
        self.test_plot_steps.set_ydata(test_losses)
        self.ax3.relim()
        self.ax3.autoscale_view()
        # loss plot (time)
        self.train_plot_time.set_xdata(exe_times)
        self.train_plot_time.set_ydata(train_losses)
        self.test_plot_time.set_xdata(exe_times)
        self.test_plot_time.set_ydata(test_losses)
        self.ax4.relim()
        self.ax4.autoscale_view()

    def draw_plots(self, iteration, coefficients, y, known_points, predicted_points, descent_steps, exe_times, train_losses, test_losses):
        if not self.figure_created:
            self.create_figure(coefficients, y, known_points, predicted_points, descent_steps, exe_times, train_losses, test_losses)
        else:
            self.update_figure(y, known_points, predicted_points, descent_steps, exe_times, train_losses, test_losses)
        self.ax1.set_title('Truth vs. fit & prediction - iteration %d' % iteration)
        self.ax2.set_title('Generalized fit & prediction - iteration %d' % iteration)
        self.ax3.set_title('Convergence (train=%.4f, test=%.4f)' % (train_losses[-1], test_losses[-1]))
        self.ax4.set_title('Convergence (train=%.4f, test=%.4f)' % (train_losses[-1], test_losses[-1]))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path)