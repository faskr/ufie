import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class LivePlots:
    def __init__(self):
        self.figure_created = False

    def create_figure(self, y, known_points, predictions, num_functions, steps, exe_times, train_losses, test_losses):
        # figure
        plt.ion()
        gs = gridspec.GridSpec(2, 2)
        self.fig = plt.figure(figsize=(18, 9))
        plt.get_current_fig_manager().window.state('zoomed')
        # prediction plot
        self.ax1 = self.fig.add_subplot(gs[0, :])
        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.plots = []
        for i in range(num_functions):
            self.plots += [self.ax1.plot(np.arange(known_points), y[i][:known_points]) +
                           self.ax1.plot(np.arange(known_points, known_points + predictions), y[i][known_points:], ':')]
            self.plots[-1][1].set_color(self.plots[-1][0].get_color())
        # loss plot (steps)
        self.ax2 = self.fig.add_subplot(gs[1, 0])
        self.ax2.set_xlabel('step')
        self.ax2.set_ylabel('loss')
        self.ax2.set_yscale('log')
        self.train_plot_steps, = self.ax2.plot(steps, train_losses, 'b')
        self.test_plot_steps, = self.ax2.plot(steps, test_losses, 'r')
        self.ax2.legend(['Training', 'Testing'])
        # loss plot (time)
        self.ax3 = self.fig.add_subplot(gs[1, 1])
        self.ax3.set_xlabel('seconds')
        self.ax3.set_ylabel('loss')
        self.ax3.set_yscale('log')
        self.train_plot_time, = self.ax3.plot(exe_times, train_losses, 'b')
        self.test_plot_time, = self.ax3.plot(exe_times, test_losses, 'r')
        self.ax3.legend(['Training', 'Testing'])
        # figure is created
        self.figure_created = True

    def update_figure(self, y, known_points, predictions, num_functions, steps, exe_times, train_losses, test_losses):
        # prediction plot
        for i in range(num_functions):
            self.plots[i][0].set_xdata(np.arange(known_points))
            self.plots[i][0].set_ydata(y[i][:known_points])
            self.plots[i][1].set_xdata(np.arange(known_points, known_points + predictions))
            self.plots[i][1].set_ydata(y[i][known_points:])
        self.ax1.relim()
        self.ax1.autoscale_view()
        # loss plot (steps)
        self.train_plot_steps.set_xdata(steps)
        self.train_plot_steps.set_ydata(train_losses)
        self.test_plot_steps.set_xdata(steps)
        self.test_plot_steps.set_ydata(test_losses)
        self.ax2.relim()
        self.ax2.autoscale_view()
        # loss plot (time)
        self.train_plot_time.set_xdata(exe_times)
        self.train_plot_time.set_ydata(train_losses)
        self.test_plot_time.set_xdata(exe_times)
        self.test_plot_time.set_ydata(test_losses)
        self.ax3.relim()
        self.ax3.autoscale_view()

    def draw_plots(self, y, iteration, known_points, predictions, num_functions, steps, exe_times, train_losses, test_losses):
        if not self.figure_created:
            self.create_figure(y, known_points, predictions, num_functions, steps, exe_times, train_losses, test_losses)
        else:
            self.update_figure(y, known_points, predictions, num_functions, steps, exe_times, train_losses, test_losses)
        self.ax1.set_title('Predict future values (shown as dashlines) - iteration %d' % iteration)
        self.ax2.set_title('Convergence (train=%.4f, test=%.4f)' % (train_losses[-1], test_losses[-1]))
        self.ax3.set_title('Convergence (train=%.4f, test=%.4f)' % (train_losses[-1], test_losses[-1]))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save(self, path):
        self.fig.savefig(path)