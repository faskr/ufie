import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import sleep

class LivePrediction:
    def __init__(self):
        self.figure_created = False
    
    def draw(self, plot, yi, input_size, future):
        plot[0].set_xdata(np.arange(input_size))
        plot[0].set_ydata(yi[:input_size])
        plot[1].set_xdata(np.arange(input_size, input_size + future))
        plot[1].set_ydata(yi[input_size:])
        plot[1].set_color(plot[0].get_color())
        plot[1].set_linestyle(':')
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        #plt.plot(np.arange(input_size), yi[:input_size], color, linewidth = 2.0)
        #plt.plot(np.arange(input_size, input_size + future), yi[input_size:], color + ':', linewidth = 2.0)
        #plt.pause(0.05)

    def draw_prediction(self, i, y, input_size, future, test_size):
        if not self.figure_created:
            plt.ion()
            #plt.figure(figsize=(30,10))
            self.fig, self.ax = plt.subplots()
            self.plots = []
            for test in range(test_size):
                self.plots += [self.ax.plot([], []) + self.ax.plot([], [])]
            self.figure_created = True
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values) - Iteration %d'%i)
        plt.xlabel('x')
        plt.ylabel('y')
        for test in range(test_size):
            self.draw(self.plots[test], y[test], input_size, future)
        #plt.draw()
        plt.savefig('results/prediction_%d.pdf'%i)
        #plt.close()

def draw_convergence(train_losses, test_losses, recorded_steps, depthH, breadth, lr, steps):
    plt.figure()
    plt.title('Convergence (train=%.4f, test=%.4f)' % (train_losses[-1], test_losses[-1]))
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.plot(recorded_steps, train_losses, 'b')
    plt.plot(recorded_steps, test_losses, 'r')
    plt.yscale('log')
    plt.legend(['Training', 'Testing'])
    plt.savefig('results/convergence_%.4f_%dx%d_%.2flr_%ds.pdf' % (test_losses[-1], depthH, breadth, lr, steps))
